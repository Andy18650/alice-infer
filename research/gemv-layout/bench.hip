#include "kernels.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace gemv_layout;
#define HIP(call) do { const auto e=(call); if(e!=hipSuccess) throw std::runtime_error(std::string(#call)+": "+hipGetErrorString(e)); } while(0)

Geometry geometry(uint32_t n,uint32_t k,uint32_t splits) {
  if(!n || !splits) throw std::runtime_error("rows/splits must be positive");
  const uint64_t tiles=(uint64_t(n)+rows_per_tile-1)/rows_per_tile;
  const uint64_t steps=(uint64_t(k)+4*uint64_t(splits)-1)/(4*uint64_t(splits));
  const uint64_t jobs=tiles*splits;
  constexpr auto limit=std::numeric_limits<uint32_t>::max();
  if(tiles*rows_per_tile>limit || jobs>limit || steps*4*splits>limit ||
     jobs*steps>std::numeric_limits<size_t>::max()/(threads*sizeof(uint4)))
    throw std::runtime_error("packed geometry overflow");
  return {n,k,uint32_t(tiles),splits,uint32_t(steps),uint32_t(tiles*rows_per_tile),uint32_t(jobs)};
}
const char* name(Format f) {
  switch(f) { case Format::Interleaved:return "interleaved";case Format::Adjacent:return "adjacent";case Format::Column:return "column"; }
  throw std::runtime_error("invalid format");
}
float widen(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits)<<16); }

// Independent forward-scatter CPU packer: GPU initializer uses inverse indexing.
std::vector<uint32_t> pack_cpu(Format f,Geometry g,std::span<const uint16_t> weights) {
  if(weights.size()!=size_t(g.rows)*g.cols) throw std::runtime_error("invalid logical weight extent");
  std::vector<uint32_t> p(g.records()*4,0);
  for(uint32_t row=0;row<g.rows;++row) for(uint32_t k=0;k<g.cols;++k) {
    size_t index;
    if(f==Format::Column) index=size_t(k)*(g.padded_rows/2)+row/2;
    else {
      const uint32_t split=k/(g.steps*4),step=(k/4)%g.steps;
      const uint32_t job=split*g.row_tiles+row/rows_per_tile;
      const size_t tile=f==Format::Adjacent?size_t(job)*g.steps+step:size_t(step)*g.jobs+job;
      index=(tile*threads+(row%rows_per_tile)/2)*4+k%4;
    }
    p[index]|=uint32_t(weights[size_t(row)*g.cols+k])<<((row&1)*16);
  }
  return p;
}

struct Reference { std::vector<double> dot,expanded; uint32_t checksum=0; };
Reference reference(uint32_t n,uint32_t k,uint32_t seed,const std::vector<float>& x) {
  Reference ref{std::vector<double>(n),std::vector<double>(n),0};
  const uint32_t workers=std::min(n,12u);
  std::vector<uint32_t> checksums(workers,0);
  std::vector<std::jthread> pool;
  for(uint32_t w=0;w<workers;++w) pool.emplace_back([&,w] {
    for(uint32_t row=w;row<n;row+=workers) {
      double dot=0,sum=0;
      for(uint32_t col=0;col<k;++col) {
        const uint16_t bits=weight_bits(row,col,seed);
        const double weight=widen(bits);
        dot+=weight*double(x[col]); sum+=weight;
        checksums[w]+=uint32_t(bits)<<((row&1)*16);
      }
      ref.dot[row]=dot; ref.expanded[row]=sum;
    }
  });
  pool.clear(); // join before publishing results or timing GPU work
  for(auto sum:checksums) ref.checksum+=sum;
  return ref;
}

struct Deleter { template<class T> void operator()(T* p) const { if(p) (void)hipFree(p); } };
template<class T> using Device=std::unique_ptr<T,Deleter>;
template<class T> Device<T> allocate(size_t count) {
  T* p=nullptr; HIP(hipMalloc(&p,std::max<size_t>(count,4)*sizeof(T))); return Device<T>(p);
}

struct Case {
  Format format; int batch,grid; Geometry g;
  std::array<const void*,3> functions;
  std::array<int,3> registers,capacity;
  const void* fused{};
  int fused_registers{},fused_capacity{};
  size_t fused_lds{};
};
template<Format F,int Batch>
void add(std::vector<Case>& cases,Geometry g,int target) {
  Case c{F,Batch,target?int(std::min<uint32_t>(target,g.jobs)):int(g.jobs),g,
         {reinterpret_cast<const void*>(partial<F,Batch,Mode::Read>),
          reinterpret_cast<const void*>(partial<F,Batch,Mode::Expand>),
          reinterpret_cast<const void*>(partial<F,Batch,Mode::Gemv>)},{},{}};
  for(size_t i=0;i<c.functions.size();++i) {
    hipFuncAttributes attrs{}; HIP(hipFuncGetAttributes(&attrs,c.functions[i]));
    if(attrs.localSizeBytes || attrs.sharedSizeBytes) throw std::runtime_error("unexpected scratch/LDS in candidate");
    c.registers[i]=attrs.numRegs;
    HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&c.capacity[i],c.functions[i],threads,0));
  }
  c.fused=reinterpret_cast<const void*>(last_arriver<F,Batch>);
  hipFuncAttributes fused_attrs{};HIP(hipFuncGetAttributes(&fused_attrs,c.fused));
  if(fused_attrs.localSizeBytes) throw std::runtime_error("scratch in last-arriver candidate");
  c.fused_registers=fused_attrs.numRegs;c.fused_lds=fused_attrs.sharedSizeBytes;
  HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&c.fused_capacity,c.fused,threads,0));
  cases.push_back(c);
}
void add_formats(std::vector<Case>& cases,Geometry g,int grid,int batch) {
  if(batch==4) { add<Format::Interleaved,4>(cases,g,grid);add<Format::Adjacent,4>(cases,g,grid);add<Format::Column,4>(cases,g,grid); }
  else { add<Format::Interleaved,8>(cases,g,grid);add<Format::Adjacent,8>(cases,g,grid);add<Format::Column,8>(cases,g,grid); }
}

struct Context {
  hipStream_t stream{}; hipEvent_t begin{},end{};
  Device<uint4> weights;
  Device<float> x,partials,y;
  Device<uint32_t> sink,counters,padded_counters;
  std::vector<float> host_y;
  explicit Context(const std::vector<Case>& cases) {
    size_t records=0,pwords=0,kwords=0,outputs=0,checksum_words=0,tiles=0;
    for(const auto& c:cases) {
      records=std::max(records,c.g.records());pwords=std::max(pwords,size_t(c.g.splits)*c.g.padded_rows);
      kwords=std::max(kwords,size_t(c.g.padded_cols())); outputs=std::max(outputs,size_t(c.g.padded_rows));
      checksum_words=std::max(checksum_words,size_t(c.g.jobs)*threads);tiles=std::max(tiles,size_t(c.g.row_tiles));
    }
    weights=allocate<uint4>(records);x=allocate<float>(kwords);partials=allocate<float>(pwords);
    y=allocate<float>(outputs);sink=allocate<uint32_t>(checksum_words);counters=allocate<uint32_t>(tiles);padded_counters=allocate<uint32_t>(tiles*64);host_y.resize(outputs);
    HIP(hipStreamCreate(&stream));HIP(hipEventCreate(&begin));HIP(hipEventCreate(&end));
  }
  ~Context() { (void)hipEventDestroy(begin);(void)hipEventDestroy(end);(void)hipStreamDestroy(stream); }
  void upload_x(const Case& c,std::span<const float> hx) {
    if(hx.size()!=c.g.cols) throw std::runtime_error("invalid activation extent");
    std::vector<float> padded(hx.begin(),hx.end());padded.resize(c.g.padded_cols(),0);
    if(!padded.empty()) HIP(hipMemcpyAsync(x.get(),padded.data(),padded.size()*4,hipMemcpyHostToDevice,stream));
    HIP(hipStreamSynchronize(stream)); // pageable source lifetime ends here
  }
  void prepare(const Case& c,uint32_t seed,const std::vector<float>& hx) {
    upload_x(c,hx);
    switch(c.format) {
      case Format::Interleaved:initialize<Format::Interleaved><<<832,256,0,stream>>>(weights.get(),c.g,seed);break;
      case Format::Adjacent:initialize<Format::Adjacent><<<832,256,0,stream>>>(weights.get(),c.g,seed);break;
      case Format::Column:initialize<Format::Column><<<832,256,0,stream>>>(weights.get(),c.g,seed);break;
    }
    reset_counters<<<(c.g.row_tiles+255)/256,256,0,stream>>>(counters.get(),c.g.row_tiles,c.g.splits,1);
    reset_counters<<<(c.g.row_tiles+255)/256,256,0,stream>>>(padded_counters.get(),c.g.row_tiles,c.g.splits,64);
    HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
  }
  void main(const Case& c,Mode mode) {
    auto* w=weights.get();auto* xp=x.get();auto* output=c.g.splits==1?y.get():partials.get();auto* sums=sink.get();
    Geometry g=c.g;void* args[]={&w,&xp,&output,&sums,&g};
    HIP(hipLaunchKernel(c.functions[size_t(mode)],dim3(c.grid),dim3(threads),args,0,stream));
  }
  void reduce(const Case& c) {
    if(c.g.splits>1) combine<<<(c.g.rows+255)/256,256,0,stream>>>(partials.get(),y.get(),c.g);
    HIP(hipGetLastError());
  }
  void fused(const Case& c,bool spaced=false) {
    if(c.g.splits==1) {main(c,Mode::Gemv);return;} // No counters/fences for unsplit.
    auto* w=weights.get();auto* xp=x.get();auto* p=partials.get();auto* out=y.get();auto* count=spaced?padded_counters.get():counters.get();
    uint32_t stride=spaced?64:1;
    Geometry g=c.g;void* args[]={&w,&xp,&p,&out,&count,&g,&stride};
    HIP(hipLaunchKernel(c.fused,dim3(c.grid),dim3(threads),args,0,stream));
  }
  void check_counters(const Case& c,bool spaced=false) {
    const size_t stride=spaced?64:1;
    std::vector<uint32_t> host(c.g.row_tiles*stride);
    HIP(hipMemcpyAsync(host.data(),spaced?padded_counters.get():counters.get(),host.size()*4,hipMemcpyDeviceToHost,stream));HIP(hipStreamSynchronize(stream));
    for(uint32_t tile=0;tile<c.g.row_tiles;++tile) if(host[size_t(tile)*stride]!=c.g.splits)
      throw std::runtime_error("counter not restored after last-arriver operation");
  }
  void launch(const Case& c,const std::string& phase) {
    if(phase=="gemv_last" || phase=="gemv_last_spaced") {fused(c,phase=="gemv_last_spaced");return;}
    if(phase=="combine") { reduce(c);return; }
    const Mode mode=phase=="read"?Mode::Read:phase=="expand"?Mode::Expand:Mode::Gemv;
    main(c,mode);
    if(phase=="gemv_full") reduce(c);
  }
  double validate(const Case& c,const Reference& ref,Mode mode) {
    if(mode==Mode::Read) {
      std::vector<uint32_t> h(size_t(c.g.jobs)*threads);
      HIP(hipMemcpyAsync(h.data(),sink.get(),h.size()*4,hipMemcpyDeviceToHost,stream));HIP(hipStreamSynchronize(stream));
      uint32_t sum=0;for(auto v:h) sum+=v;
      if(sum!=ref.checksum) throw std::runtime_error("read checksum mismatch");
      return 0;
    }
    HIP(hipMemcpyAsync(host_y.data(),y.get(),c.g.rows*4,hipMemcpyDeviceToHost,stream));HIP(hipStreamSynchronize(stream));
    const auto& gold=mode==Mode::Expand?ref.expanded:ref.dot;
    double max_error=0;
    for(uint32_t row=0;row<c.g.rows;++row) {
      const double err=std::abs(double(host_y[row])-gold[row]);max_error=std::max(max_error,err);
      if(!std::isfinite(host_y[row]) || err>2e-4+3e-5*std::abs(gold[row]))
        throw std::runtime_error("numeric mismatch row="+std::to_string(row)+" error="+std::to_string(err));
    }
    return max_error;
  }
};

void validate_case(Context& ctx,const Case& c,const Reference& ref) {
  HIP(hipMemsetAsync(ctx.partials.get(),0xff,size_t(c.g.splits)*c.g.padded_rows*4,ctx.stream));
  HIP(hipMemsetAsync(ctx.y.get(),0xff,c.g.padded_rows*4,ctx.stream));
  ctx.main(c,Mode::Read);ctx.validate(c,ref,Mode::Read);
  for(auto mode:{Mode::Expand,Mode::Gemv}) {ctx.main(c,mode);ctx.reduce(c);ctx.validate(c,ref,mode);}
  std::vector<float> first(c.g.rows);
  HIP(hipMemcpy(first.data(),ctx.y.get(),first.size()*4,hipMemcpyDeviceToHost));
  ctx.main(c,Mode::Gemv);ctx.reduce(c);HIP(hipStreamSynchronize(ctx.stream));
  HIP(hipMemcpy(ctx.host_y.data(),ctx.y.get(),first.size()*4,hipMemcpyDeviceToHost));
  if(std::memcmp(first.data(),ctx.host_y.data(),first.size()*sizeof(float))) throw std::runtime_error("non-deterministic output");
  HIP(hipMemsetAsync(ctx.y.get(),0xff,c.g.padded_rows*4,ctx.stream));
  for(bool spaced:{false,true}) {
    ctx.fused(c,spaced);ctx.validate(c,ref,Mode::Gemv);ctx.check_counters(c,spaced);
    if(std::memcmp(first.data(),ctx.host_y.data(),first.size()*sizeof(float))) throw std::runtime_error("fused/reference reduction-tree disagreement");
  }
}

void stress_fused(Context& ctx,const Case& c,const Reference& ref,std::span<const float> hx,bool spaced) {
  for(int i=0;i<16;++i) ctx.fused(c,spaced); // No counter initialization between invocations.
  ctx.validate(c,ref,Mode::Gemv);ctx.check_counters(c,spaced);
  std::vector<float> negative(hx.begin(),hx.end());for(auto& v:negative)v=-v;
  auto neg_ref=ref;for(auto& v:neg_ref.dot)v=-v;
  ctx.upload_x(c,negative); // Change data without resetting counters or re-packing weights.
  HIP(hipMemsetAsync(ctx.partials.get(),0xff,size_t(c.g.splits)*c.g.padded_rows*4,ctx.stream));
  HIP(hipMemsetAsync(ctx.y.get(),0xff,c.g.padded_rows*4,ctx.stream));
  for(int i=0;i<16;++i) ctx.fused(c,spaced);
  ctx.validate(c,neg_ref,Mode::Gemv);ctx.check_counters(c,spaced);
  ctx.upload_x(c,hx);ctx.fused(c,spaced);ctx.validate(c,ref,Mode::Gemv);ctx.check_counters(c,spaced);
}

void selftest() {
  for(auto [n,k]:{std::pair{1u,0u},{1u,1u},{3u,17u},{513u,37u},{1025u,259u}}) {
    for(uint32_t splits:{1u,3u,7u}) {
      std::vector<Case> cases;add_formats(cases,geometry(n,k,splits),7,4);add_formats(cases,geometry(n,k,splits),7,8);
      Context ctx(cases);
      for(int pattern=0;pattern<4;++pattern) {
        const uint32_t seed=pattern==1?0:123;
        std::vector<float> hx(k);
        for(uint32_t j=0;j<k;++j) hx[j]=pattern==3?0:pattern==2?((j&1)?-1.00390625f:1.00390625f):1.00390625f+float(j%11)*0.00017f;
        const auto ref=reference(n,k,seed,hx);
        std::vector<uint16_t> logical(size_t(n)*k);
        for(uint32_t row=0;row<n;++row) for(uint32_t col=0;col<k;++col)
          logical[size_t(row)*k+col]=weight_bits(row,col,seed);
        for(const auto& c:cases) {
          ctx.prepare(c,seed,hx);
          const auto packed=pack_cpu(c.format,c.g,logical);
          std::vector<uint32_t> actual(packed.size());
          if(!actual.empty()) HIP(hipMemcpy(actual.data(),ctx.weights.get(),actual.size()*4,hipMemcpyDeviceToHost));
          if(actual!=packed) throw std::runtime_error("CPU/GPU packing disagreement");
          validate_case(ctx,c,ref);for(bool spaced:{false,true}) stress_fused(ctx,c,ref,hx,spaced);
        }
      }
    }
  }
  // More CTAs than can reside; also one CTA serially processing every job.
  // Neither path waits for another CTA, so the scheduling order must progress.
  {
    std::vector<Case> cases;const auto g=geometry(513,259,1025);
    add<Format::Interleaved,4>(cases,g,0);add<Format::Interleaved,4>(cases,g,1);
    std::vector<float> hx(g.cols,1.00390625f);const auto ref=reference(g.rows,g.cols,123,hx);
    Context ctx(cases);
    for(const auto& c:cases) {ctx.prepare(c,123,hx);validate_case(ctx,c,ref);for(bool spaced:{false,true}) stress_fused(ctx,c,ref,hx,spaced);}
  }
  std::fprintf(stderr,"packing, odd N/K, empty K, split tails, FP32-x precision, zero/cancellation, determinism, last-arriver reuse/changing-input/oversubscription tests passed\n");
}

struct Shape { const char* name;uint32_t n,k; };
int main(int argc,char** argv) try {
  if(argc<2 || argc>5) throw std::runtime_error("usage: bench OUT [quick|sweep|test] [ROUNDS=9] [ITERS=12]");
  const std::string out=argv[1],suite=argc>2?argv[2]:"quick";
  const int rounds=argc>3?std::stoi(argv[3]):9,iters=argc>4?std::stoi(argv[4]):12;
  if((suite!="quick" && suite!="sweep" && suite!="test") || rounds<3 || rounds%2==0 || (iters!=1 && iters!=12)) throw std::runtime_error("invalid suite/rounds/iterations");
  std::filesystem::create_directories(out);HIP(hipSetDevice(0));
  hipDeviceProp_t prop{};HIP(hipGetDeviceProperties(&prop,0));
  if(prop.warpSize!=64 || prop.multiProcessorCount!=104) throw std::runtime_error("requires 104-CU wave64 GPU");
  char pci[64];HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
  {std::ofstream f(out+"/device.txt");f<<"device=0 PCI="<<pci<<" name="<<prop.name<<" CUs="<<prop.multiProcessorCount<<" wave="<<prop.warpSize<<" BF16_weights FP32_x_accum_y no_x_quantization strong_publication="<<GEMV_LAST_STRONG_PUBLICATION<<'\n';}
  selftest(); if(suite=="test") return 0;
  std::vector<Shape> shapes={{"tp4_o",2048,8192},{"tp4_gup",14336,8192},{"stream104",53248,8192}};
  if(suite=="sweep") shapes={{"8b_qkv",6144,4096},{"8b_o",4096,4096},{"8b_gup",28672,4096},{"8b_down",4096,14336},
                            {"tp4_qkv",2560,8192},{"tp4_o",2048,8192},{"tp4_gup",14336,8192},{"tp4_down",8192,7168},{"stream104",53248,8192}};
  std::ofstream raw(out+"/raw.csv"),resources(out+"/resources.csv"),fused_resources(out+"/fused-resources.csv");
  raw<<"shape,N,K,format,batch,splits,grid,jobs,padded_bytes,useful_bytes,phase,round,iters,ms,wall_ms,useful_GB_s,packed_GB_s,max_abs_error\n"<<std::setprecision(12);
  resources<<"shape,format,batch,splits,grid,jobs,mode,regs,capacity_CTAs_per_CU\n";
  fused_resources<<"shape,format,batch,splits,grid,jobs,regs,capacity_CTAs_per_CU,lds_bytes\n";
  std::mt19937 random(0x90a);
  for(const auto& shape:shapes) {
    std::vector<float> hx(shape.k);std::uniform_real_distribution<float> dist(-1.f,1.f);
    for(auto& value:hx) value=dist(random);
    const auto ref=reference(shape.n,shape.k,789,hx);
    std::vector<Case> cases;
    for(int target:{80,104}) {
      const uint32_t tiles=(shape.n+511)/512,fill=(target+tiles-1)/tiles;
      const uint32_t balanced=uint32_t(target)/std::gcd(uint32_t(target),tiles);
      std::vector<uint32_t> splits={1,fill,2*fill,balanced};
      std::sort(splits.begin(),splits.end());splits.erase(std::unique(splits.begin(),splits.end()),splits.end());
      for(auto split:splits) for(int batch:{4,8}) {
        const auto g=geometry(shape.n,shape.k,split);
        add_formats(cases,g,target,batch);
        add_formats(cases,g,0,batch); // Natural task grid: no persistent-grid tail.
      }
    }
    // Natural-job column-major control, akin to the current 256-K partition.
    add<Format::Column,4>(cases,geometry(shape.n,shape.k,(shape.k+255)/256),0);
    const auto key=[](const Case& c) {return std::tuple{c.format,c.batch,c.g.splits,c.grid};};
    std::sort(cases.begin(),cases.end(),[&](const Case& a,const Case& b){return key(a)<key(b);});
    cases.erase(std::unique(cases.begin(),cases.end(),[&](const Case& a,const Case& b){return key(a)==key(b);}),cases.end());
    Context ctx(cases);
    for(const auto& c:cases) for(size_t mode=0;mode<3;++mode)
      resources<<shape.name<<','<<name(c.format)<<','<<c.batch<<','<<c.g.splits<<','<<c.grid<<','<<c.g.jobs<<','<<mode<<','<<c.registers[mode]<<','<<c.capacity[mode]<<'\n';
    for(const auto& c:cases) fused_resources<<shape.name<<','<<name(c.format)<<','<<c.batch<<','<<c.g.splits<<','<<c.grid<<','<<c.g.jobs
      <<','<<c.fused_registers<<','<<c.fused_capacity<<','<<c.fused_lds<<'\n';
    resources.flush();fused_resources.flush();
    std::vector<size_t> order(cases.size());std::iota(order.begin(),order.end(),0);
    for(int round=0;round<rounds;++round) {
      std::shuffle(order.begin(),order.end(),random);
      for(size_t index:order) {
        const auto& c=cases[index];ctx.prepare(c,789,hx);validate_case(ctx,c,ref);
        std::vector<std::string> phases={"read","expand","gemv_partial","gemv_full","gemv_last","gemv_last_spaced"};
        if(c.g.splits>1) phases.push_back("combine");
        std::shuffle(phases.begin(),phases.end(),random);
        for(const auto& phase:phases) {
          if(phase=="combine") {ctx.main(c,Mode::Gemv);HIP(hipStreamSynchronize(ctx.stream));}
          ctx.launch(c,phase);ctx.launch(c,phase);HIP(hipStreamSynchronize(ctx.stream));
          const auto start=std::chrono::steady_clock::now();HIP(hipEventRecord(ctx.begin,ctx.stream));
          for(int i=0;i<iters;++i) ctx.launch(c,phase);
          HIP(hipEventRecord(ctx.end,ctx.stream));HIP(hipEventSynchronize(ctx.end));
          float elapsed;HIP(hipEventElapsedTime(&elapsed,ctx.begin,ctx.end));
          const double ms=double(elapsed)/iters;
          const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iters;
          const Mode mode=phase=="read"?Mode::Read:phase=="expand"?Mode::Expand:Mode::Gemv;
          if(phase=="expand" || phase=="gemv_partial") ctx.reduce(c);
          const double error=ctx.validate(c,ref,mode);
          if(phase=="gemv_last" || phase=="gemv_last_spaced") ctx.check_counters(c,phase=="gemv_last_spaced");
          const size_t useful=size_t(shape.n)*shape.k*2,packed=c.g.records()*16;
          // Combine does not read weights: report its time, never fake weight bandwidth.
          const double useful_bw=phase=="combine"?0:useful/(ms*1e6);
          const double packed_bw=phase=="combine"?0:packed/(ms*1e6);
          raw<<shape.name<<','<<shape.n<<','<<shape.k<<','<<name(c.format)<<','<<c.batch<<','<<c.g.splits<<','<<c.grid<<','<<c.g.jobs
             <<','<<packed<<','<<useful<<','<<phase<<','<<round<<','<<iters<<','<<ms<<','<<wall<<','<<useful_bw<<','<<packed_bw<<','<<error<<'\n';
        }
      }
      raw.flush();std::fprintf(stderr,"%s round %d/%d (%zu cases)\n",shape.name,round+1,rounds,cases.size());
    }
  }
  return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
