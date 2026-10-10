// A/B/C: same flat input, different per-instruction addresses.
// Block/thread baseline; separate wave-indexed comparison control.
// Native gfx90a HIP/C++20, uint32_t input and built-in uint4 loads/checksums.
#include "wave-control.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while(0)

struct Case {
  std::string pattern;
  int chunks,grid;
  const void* function;
  int registers,capacity;
  size_t scratch;
};
template<class Reader,int Chunks>
void add(std::vector<Case>& cases,int grid) {
  const void* function=reinterpret_cast<const void*>(read_tiles<Reader,Chunks>);
  hipFuncAttributes attributes{}; HIP(hipFuncGetAttributes(&attributes,function));
  int capacity; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&capacity,function,256,0));
  cases.push_back({Reader::name,Chunks,grid,function,attributes.numRegs,capacity,attributes.localSizeBytes});
}
template<class Reader,int Chunks>
void add_order_pair(std::vector<Case>& cases,int grid) {
  add<Reader,Chunks>(cases,grid);
  cases.back().pattern+="_grid";
  const void* fn=reinterpret_cast<const void*>(read_adjacent_tiles<Reader,Chunks>);
  hipFuncAttributes attributes{}; HIP(hipFuncGetAttributes(&attributes,fn));
  int capacity; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&capacity,fn,256,0));
  cases.push_back({std::string(Reader::name)+"_adjacent",Chunks,grid,fn,attributes.numRegs,capacity,attributes.localSizeBytes});
}

template<class Reader,int Chunks>
void add_body_pair(std::vector<Case>& cases,int grid) {
  auto append=[&](const void* fn,const char* order) {
    hipFuncAttributes attributes{}; HIP(hipFuncGetAttributes(&attributes,fn));
    int capacity; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&capacity,fn,256,0));
    cases.push_back({std::string(Reader::name)+order,Chunks,grid,fn,attributes.numRegs,capacity,attributes.localSizeBytes});
  };
  append(reinterpret_cast<const void*>(read_window_bodies<Reader,Chunks,false>),"_grid");
  append(reinterpret_cast<const void*>(read_window_bodies<Reader,Chunks,true>),"_adjacent");
}

template<int Chunks> void add_abc(std::vector<Case>& cases,int grid) {
  add<A,Chunks>(cases,grid); add<B,Chunks>(cases,grid); add<C,Chunks>(cases,grid);
}

template<class Reader,int Chunks>
void add_index_pair(std::vector<Case>& cases,int grid) {
  const void* fn=reinterpret_cast<const void*>(read_wave_tiles<Reader,Chunks>);
  hipFuncAttributes attributes{}; HIP(hipFuncGetAttributes(&attributes,fn));
  int capacity; HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&capacity,fn,256,0));
  cases.push_back({std::string(Reader::name)+"_wave",Chunks,grid,fn,attributes.numRegs,capacity,attributes.localSizeBytes});
  add<Reader,Chunks>(cases,grid);
  cases.back().pattern+="_block";
}

struct Context {
  hipDeviceProp_t device{};
  hipStream_t stream{};
  hipEvent_t start{},end{};
  uint32_t* input{}; uint32_t* sink{};
  Context(const std::string& out) {
    HIP(hipSetDevice(0)); HIP(hipGetDeviceProperties(&device,0));
    if(device.warpSize!=64 || device.multiProcessorCount!=104) throw std::runtime_error("requires 104-CU wave64 device");
    HIP(hipStreamCreate(&stream)); HIP(hipEventCreate(&start)); HIP(hipEventCreate(&end));
    HIP(hipMalloc(&input,size_t(4)<<30)); HIP(hipMalloc(&sink,832*4*4));
    char pci[64]; HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
    std::ofstream meta(out+"/device.txt");
    meta<<"device=0 PCI="<<pci<<" name="<<device.name<<" arch="<<device.gcnArchName
        <<" CUs="<<device.multiProcessorCount<<" wave="<<device.warpSize
        <<" pointer="<<static_cast<void*>(input)<<" input=uint32_t vector=uint4 generator=affine_mod2^32 baseline_indexing=block scheduling_barriers="<<RESEARCH_SCHED_BARRIERS<<" per_load_barriers="<<RESEARCH_LOAD_BARRIERS<<'\n'; meta.close();
    initialize<<<832,256,0,stream>>>(input,(size_t(4)<<30)/4);
    HIP(hipGetLastError()); HIP(hipStreamSynchronize(stream));
  }
  ~Context() {
    (void)hipFree(sink); (void)hipFree(input); (void)hipEventDestroy(start);
    (void)hipEventDestroy(end); (void)hipStreamDestroy(stream);
  }
  void launch(const Case& c,size_t bytes,size_t offset=0,size_t sink_offset=0) {
    const size_t capacity=size_t(4)<<30;
    if(offset%16 || bytes%4 || offset>capacity || bytes>capacity-offset)
      throw std::runtime_error("invalid input slice");
    if(sink_offset+size_t(c.grid)*4>832*4) throw std::runtime_error("invalid output slice");
    uint32_t* slice=input+offset/4; uint32_t* output=sink+sink_offset;
    size_t count=bytes/4; void* args[]={&slice,&count,&output};
    HIP(hipLaunchKernel(c.function,dim3(c.grid),dim3(256),args,0,stream));
  }
  void validate(const Case& c,size_t bytes,size_t offset=0,size_t sink_offset=0) {
    std::vector<uint32_t> host(c.grid*4);
    HIP(hipMemcpyAsync(host.data(),sink+sink_offset,host.size()*4,hipMemcpyDeviceToHost,stream)); HIP(hipStreamSynchronize(stream));
    uint32_t sum=0; for(auto value:host) sum+=value;
    const uint32_t expected=expected_sum(bytes/4)+uint32_t(bytes/4)*uint32_t(offset/4)*multiplier;
    if(sum!=expected) throw std::runtime_error("checksum mismatch: "+c.pattern);
  }
  void launch_bodies(const Case& c,size_t window_bytes) {
    if(window_bytes%16 || window_bytes>((size_t(4)<<30)/4) || size_t(c.grid)*4*4>832*4)
      throw std::runtime_error("invalid four-body input/output extent");
    launch(c,window_bytes);
  }
  double measure(const Case& c,size_t bytes,size_t offset=0) {
    HIP(hipEventRecord(start,stream));
    for(int i=0;i<12;++i) launch(c,bytes,offset);
    HIP(hipGetLastError()); HIP(hipEventRecord(end,stream)); HIP(hipEventSynchronize(end));
    float ms; HIP(hipEventElapsedTime(&ms,start,end)); return double(ms)/12;
  }
};

double unix_seconds() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void window_status(const std::string& out,const std::string& stage,const Case& c,size_t bytes,size_t offset=0) {
  { std::ofstream f(out+"/status.tmp"); f<<stage<<','<<c.pattern<<','<<bytes<<','<<offset<<'\n'; }
  std::filesystem::rename(out+"/status.tmp",out+"/status.txt");
}

void run_windows(Context& context,const std::vector<Case>& cases,const std::string& out,int rounds,
                 const std::vector<Case>& bodies={},int timed_iters=12) {
  constexpr size_t gib=size_t(1)<<30;
  // Verify offset-dependent checksums with short inputs: full GiB affine
  // sums can coincide modulo 2^32, so those alone cannot validate offsets.
  for(const auto& c:cases) for(size_t offset:{size_t(0),gib,2*gib,3*gib})
    for(size_t bytes:{size_t(0),size_t(4),size_t(68),size_t(4160)}) {
      context.launch(c,bytes,offset); context.validate(c,bytes,offset);
    }
  if(!bodies.empty()) {
    if(bodies.size()!=cases.size()) throw std::runtime_error("unmatched body-control cases");
    std::ofstream resources(out+"/body-resources.csv");
    resources<<"pattern,chunks,grid,block,reported_regs,capacity_CTAs_per_CU,private_bytes\n";
    for(size_t i=0;i<bodies.size();++i) {
      const auto& c=bodies[i];
      if(c.pattern!=cases[i].pattern || c.chunks!=cases[i].chunks || c.grid!=cases[i].grid || c.scratch)
        throw std::runtime_error("invalid/scratch-using body-control case");
      resources<<c.pattern<<','<<c.chunks<<','<<c.grid<<",256,"<<c.registers<<','<<c.capacity<<','<<c.scratch<<'\n';
      const size_t batch_bytes=size_t(c.grid)*256*4*sizeof(uint32_t)*c.chunks;
      for(size_t bytes:{size_t(0),size_t(16),size_t(64),size_t(4160),size_t((1<<20)+64),batch_bytes-16,batch_bytes+16}) {
        context.launch_bodies(c,bytes);
        for(size_t j=0;j<4;++j) context.validate(c,bytes,j*bytes,j*size_t(c.grid)*4);
      }
    }
    std::fprintf(stderr,"%zu body controls passed separate-output/tail checks\n",bodies.size());
  }
  struct Work { size_t index,bytes,offset; const char* kind; };
  std::vector<Work> work;
  for(size_t i=0;i<cases.size();++i) {
    if(bodies.empty()) for(size_t offset:{size_t(0),gib,2*gib,3*gib}) work.push_back({i,gib,offset,"window"});
    work.push_back({i,4*gib,0,"full"});
    work.push_back({i,4*gib,0,"split4"});
    if(!bodies.empty()) work.push_back({i,4*gib,0,"body4"});
  }
  std::ofstream raw(out+"/windows-raw.csv"),warm(out+"/warmups.csv"),curve(out+"/warm-bandwidth.csv");
  raw<<"stage,pattern,chunks,grid,block,kind,bytes,offset_bytes,round,iters,ms,GB_s,start_unix_s,end_unix_s,wall_ms\n"<<std::setprecision(12);
  warm<<"stage,wall_s,launches,start_unix_s,end_unix_s\n"<<std::setprecision(12);
  curve<<"elapsed_s,pattern,chunks,grid,bytes,ms,GB_s,unix_s\n"<<std::setprecision(12);
  std::mt19937 random(0x90a);
  for(const std::string stage:{"brief","sustained"}) {
    const double warm_start=unix_seconds();
    const auto start=std::chrono::steady_clock::now();
    size_t launches=0;
    if(stage=="brief") {
      window_status(out,"brief_warmup",cases.front(),gib);
      for(int i=0;i<250;++i) context.launch(cases.front(),gib);
      launches=250;
      HIP(hipStreamSynchronize(context.stream));
    } else {
      while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<30) {
        const auto& c=cases[(launches/12)%cases.size()];
        window_status(out,"sustained_warmup",c,4*gib);
        const double ms=context.measure(c,4*gib);
        context.validate(c,4*gib); launches+=12;
        curve<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
             <<','<<c.pattern<<','<<c.chunks<<','<<c.grid<<','<<4*gib<<','<<ms<<','<<4*gib/(ms*1e6)
             <<','<<unix_seconds()<<'\n'; curve.flush();
      }
    }
    warm<<stage<<','<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
        <<','<<launches<<','<<warm_start<<','<<unix_seconds()<<'\n'; warm.flush();
    for(int round=0;round<rounds;++round) {
      std::shuffle(work.begin(),work.end(),random);
      for(const auto& w:work) {
        const auto& c=cases[w.index];
        window_status(out,stage+"_"+w.kind,c,w.bytes,w.offset);
        const bool split=std::string(w.kind)=="split4",body=std::string(w.kind)=="body4";
        // Same 4 GiB payload; different kernel/body boundaries.
        auto launch_work=[&] {
          if(split) for(size_t j=0;j<4;++j) context.launch(c,gib,j*gib,j*size_t(c.grid)*4);
          else if(body) context.launch_bodies(bodies[w.index],gib);
          else context.launch(c,w.bytes,w.offset);
        };
        launch_work(); launch_work();
        HIP(hipStreamSynchronize(context.stream)); // Warmup is outside both timing intervals.
        const double begin=unix_seconds();
        const auto wall_start=std::chrono::steady_clock::now();
        HIP(hipEventRecord(context.start,context.stream));
        for(int i=0;i<timed_iters;++i) launch_work();
        HIP(hipGetLastError()); HIP(hipEventRecord(context.end,context.stream)); HIP(hipEventSynchronize(context.end));
        float elapsed; HIP(hipEventElapsedTime(&elapsed,context.start,context.end));
        const double ms=double(elapsed)/timed_iters;
        const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall_start).count()/timed_iters;
        if(split || body) for(size_t j=0;j<4;++j) context.validate(c,gib,j*gib,j*size_t(c.grid)*4);
        else context.validate(c,w.bytes,w.offset);
        raw<<stage<<','<<c.pattern<<','<<c.chunks<<','<<c.grid<<",256,"<<w.kind<<','<<w.bytes<<','<<w.offset
           <<','<<round<<','<<timed_iters<<','<<ms<<','<<w.bytes/(ms*1e6)<<','<<begin<<','<<unix_seconds()<<','<<wall_ms<<'\n';
      }
      raw.flush(); std::fprintf(stderr,"%s round %d/%d\n",stage.c_str(),round+1,rounds);
    }
  }
  window_status(out,"done",cases.front(),0);
}

int main(int argc,char** argv) try {
  if(argc<2 || argc>5) throw std::runtime_error("usage: probe OUT [quick|sweep|blocks|small-grid|inflight|order|order-matrix|windows|body-control|index] [ROUNDS=9] [WINDOW_ITERS=12]");
  const std::string out=argv[1],phase=argc>2?argv[2]:"quick";
  const int rounds=argc>3?std::stoi(argv[3]):9;
  if((phase!="quick" && phase!="sweep" && phase!="blocks" && phase!="small-grid" && phase!="inflight" && phase!="order" && phase!="order-matrix" && phase!="windows" && phase!="body-control" && phase!="index") || rounds<3 || rounds%2==0) throw std::runtime_error("invalid phase/rounds");
  const int timed_iters=argc>4?std::stoi(argv[4]):12;
  if((timed_iters!=1 && timed_iters!=12) || (timed_iters!=12 && phase!="windows" && phase!="body-control"))
    throw std::runtime_error("iteration override is 1/12 and only for window/body controls");
  std::filesystem::create_directories(out);
  Context context(out);
  std::vector<Case> cases;
  if(phase=="index") {
    for(int grid:{104,208,416}) { add_index_pair<A,4>(cases,grid); add_index_pair<B,4>(cases,grid); }
  } else if(phase=="windows" || phase=="body-control") {
    add_order_pair<A,4>(cases,104); add_order_pair<B,4>(cases,104);
    add_order_pair<B,16>(cases,104);
  } else if(phase=="order-matrix") {
    for(int grid:{104,80,64}) {
      add_order_pair<A,4>(cases,grid); add_order_pair<B,4>(cases,grid);
      add_order_pair<A,8>(cases,grid); add_order_pair<B,8>(cases,grid);
      add_order_pair<A,16>(cases,grid); add_order_pair<B,16>(cases,grid);
    }
  } else if(phase=="order") {
    for(int grid:{104,80}) { add_order_pair<A,4>(cases,grid); add_order_pair<B,4>(cases,grid); }
    for(int grid:{64,32}) {
      add_order_pair<A,4>(cases,grid); add_order_pair<B,4>(cases,grid);
      add_order_pair<A,8>(cases,grid); add_order_pair<B,8>(cases,grid);
      add_order_pair<A,16>(cases,grid); add_order_pair<B,16>(cases,grid);
    }
  } else if(phase=="inflight") {
    add<A,4>(cases,104); add<B,4>(cases,104);
    for(int grid:{64,32}) {
      add<A,4>(cases,grid); add<B,4>(cases,grid);
      add<A,8>(cases,grid); add<B,8>(cases,grid);
      add<A,16>(cases,grid); add<B,16>(cases,grid);
    }
  } else if(phase=="small-grid") {
    for(int grid:{104,80,64,32}) { add<A,4>(cases,grid); add<B,4>(cases,grid); }
  } else if(phase=="blocks") {
    for(int grid:{104,208,416}) { add<A,4>(cases,grid); add<B,4>(cases,grid); }
  } else {
    add_abc<1>(cases,104);
    if(phase=="sweep") {
      add_abc<2>(cases,104); add_abc<4>(cases,104); add_abc<8>(cases,104);
      add_abc<1>(cases,832); add_abc<2>(cases,832); add_abc<4>(cases,832); add_abc<8>(cases,832);
    }
  }
  std::ofstream resources(out+"/resources.csv");
  resources<<"pattern,chunks,grid,block,reported_regs,capacity_CTAs_per_CU,private_bytes\n";
  for(const auto& c:cases) {
    resources<<c.pattern<<','<<c.chunks<<','<<c.grid<<",256,"<<c.registers<<','<<c.capacity<<','<<c.scratch<<'\n';
    if(c.scratch) throw std::runtime_error("scratch-using candidate: "+c.pattern);
    const size_t batch_bytes=size_t(c.grid)*256*4*sizeof(uint32_t)*c.chunks;
    for(size_t bytes:{size_t(0),size_t(4),size_t(64),size_t(68),size_t(4160),
                      size_t((1<<20)+64),batch_bytes-4,batch_bytes+4}) {
      context.launch(c,bytes); HIP(hipGetLastError()); context.validate(c,bytes);
    }
  }
  std::fprintf(stderr,"%zu cases passed all short/tail checks\n",cases.size());
  if(phase=="windows" || phase=="body-control") {
    std::vector<Case> bodies;
    if(phase=="body-control") {
      add_body_pair<A,4>(bodies,104); add_body_pair<B,4>(bodies,104); add_body_pair<B,16>(bodies,104);
    }
    run_windows(context,cases,out,rounds,bodies,timed_iters); return 0;
  }
  for(int i=0;i<250;++i) context.launch(cases.front(),size_t(1)<<30);
  HIP(hipGetLastError()); HIP(hipStreamSynchronize(context.stream));
  struct Work { size_t index,bytes; };
  std::vector<Work> work;
  for(size_t i=0;i<cases.size();++i) {
    work.push_back({i,size_t(1)<<30});
    if(phase!="quick") { work.push_back({i,size_t(2)<<30}); work.push_back({i,size_t(4)<<30}); }
  }
  std::ofstream raw(out+"/raw.csv");
  raw<<"pattern,chunks,grid,block,bytes,round,iters,ms,GB_s\n"<<std::setprecision(10);
  std::mt19937 random(0x90a);
  for(int round=0;round<rounds;++round) {
    std::shuffle(work.begin(),work.end(),random);
    for(auto w:work) {
      const auto& c=cases[w.index]; context.launch(c,w.bytes); context.launch(c,w.bytes);
      const double ms=context.measure(c,w.bytes); context.validate(c,w.bytes);
      raw<<c.pattern<<','<<c.chunks<<','<<c.grid<<",256,"<<w.bytes<<','<<round<<",12,"<<ms<<','<<w.bytes/(ms*1e6)<<'\n';
    }
    raw.flush(); std::fprintf(stderr,"round %d/%d\n",round+1,rounds);
  }
  return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
