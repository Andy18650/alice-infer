#include "api.hpp"
#include <rocblas/rocblas.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace granularity;
#define HIP(x) do { auto hip_status=(x); if(hip_status!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(hip_status)); } while(0)
#define BLAS(x) do { auto blas_status=(x); if(blas_status!=rocblas_status_success) throw std::runtime_error(std::string(#x)+": rocBLAS status "+std::to_string(int(blas_status))); } while(0)

__host__ __device__ uint32_t hash32(uint32_t x) {
  x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
__host__ __device__ uint16_t bits(size_t i,uint32_t seed) {
  const uint32_t h=hash32(uint32_t(i)^hash32(uint32_t(i>>32))^seed);
  return uint16_t(0x3e80u|(h&127u)|((h>>16)&0x8000u));
}
float widen(uint16_t v) { return std::bit_cast<float>(uint32_t(v)<<16); }
__global__ void initialize(uint16_t* p,size_t count,int pk,int k,uint32_t seed) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)
    p[i]=int(i%pk)<k?bits(i,seed):0;
}
__host__ __device__ size_t canonical_b(size_t i,int stride,int raw_k) {
  return (i/stride)*raw_k+i%stride;
}
__global__ void initialize_b(uint16_t* p,size_t count,int pk,int k,int stride,int raw_k) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
    const int t=int(i%stride);
    p[i]=(t<raw_k && t%pk<k)?bits(canonical_b(i,stride,raw_k),456u):0;
  }
}
__global__ void initialize_packed(uint16_t* p,size_t count,int n,int raw_k,int pk,int k,WeightFormat format) {
  const int nt=(n+31)/32,kt=(raw_k+127)/128;
  const size_t one=size_t(nt)*kt*4096;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
    const size_t matrix=i/one,local=i%one,tile=local/4096;
    const int inner_n=int(local%4096)/128,inner_k=int(local%128);
    const int tn=format==WeightFormat::TileN?int(tile/kt):int(tile%nt);
    const int tk=format==WeightFormat::TileN?int(tile%kt):int(tile/nt);
    const int row=tn*32+inner_n,col=tk*128+inner_k;
    p[i]=(row<n && col<raw_k && col%pk<k)?bits((matrix*n+row)*raw_k+col,456u):0;
  }
}
__global__ void reduce(const float* partial,float* out,int m,int n,int pn,int q) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(m)*n;i+=size_t(gridDim.x)*blockDim.x) {
    size_t offset=(i/n)*pn+i%n; float sum=0;
    for(int s=0;s<q;++s) sum+=partial[size_t(s)*m*pn+offset];
    out[offset]=sum;
  }
}
__global__ void gather(const float* source,const size_t* offsets,float* values,size_t count) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)
    values[i]=source[offsets[i]];
}
__global__ void fill_reader(uint32_t* p,size_t count) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)
    p[i]=uint32_t(i)*0x9e3779b9u+17u;
}
__global__ void reader(const uint4* p,size_t records,uint32_t* sink) {
  uint32_t sum=0; const size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t stride=size_t(gridDim.x)*blockDim.x;
  for(size_t base=t;base<records;base+=stride*4) {
    uint4 v[4]{};
    #pragma unroll
    for(int j=0;j<4;++j) if(base+size_t(j)*stride<records) v[j]=p[base+size_t(j)*stride];
    __builtin_amdgcn_sched_barrier(0);
    #pragma unroll
    for(int j=0;j<4;++j) sum+=v[j].x+v[j].y+v[j].z+v[j].w;
  }
  sink[t]=sum;
}
__global__ void empty() {}
using U16x4=uint16_t __attribute__((ext_vector_type(4)));
using F32x4=float __attribute__((ext_vector_type(4)));
__global__ void mfma_reference(float* output,int iterations) {
  const U16x4 a={0x3b80,0x3b80,0x3b80,0x3b80}; // BF16 2^-8
  F32x4 c0={0,0,0,0},c1={1,1,1,1},c2={2,2,2,2},c3={3,3,3,3};
  #pragma clang loop unroll(disable)
  for(int i=0;i<iterations;++i) {
    c0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a,a,c0,0,0,0);
    c1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a,a,c1,0,0,0);
    c2=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a,a,c2,0,0,0);
    c3=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a,a,c3,0,0,0);
  }
  const auto c=c0+c1+c2+c3;
  output[size_t(blockIdx.x)*blockDim.x+threadIdx.x]=c[0]+c[1]+c[2]+c[3];
}

struct Free { template<class T> void operator()(T* p) const { if(p) (void)hipFree(p); } };
template<class T> struct Buffer {
  std::unique_ptr<T,Free> p; size_t size=0;
  explicit Buffer(size_t n=0):size(n) { if(n) {T* raw=nullptr; HIP(hipMalloc(&raw,n*sizeof(T)));p.reset(raw);} }
  T* get() const { return p.get(); }
};
struct Options {
  std::string phase="quick",axis="independent",cache="warm",mode="both",impl="all",shape_file,k_launch="separate";
  int rounds=3,repeats=16,q=1,streams=1,m=0,n=0,k=0,bpad=0;
  WeightFormat weights=WeightFormat::Column;
};
Options parse(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    const std::string a=argv[i]; if(i+1==argc) throw std::runtime_error("missing option value: "+a);
    const std::string v=argv[++i];
    if(a=="--phase") o.phase=v; else if(a=="--axis") o.axis=v; else if(a=="--cache") o.cache=v;
    else if(a=="--mode") o.mode=v; else if(a=="--impl") o.impl=v; else if(a=="--shapes") o.shape_file=v;
    else if(a=="--rounds") o.rounds=std::stoi(v); else if(a=="--repeats") o.repeats=std::stoi(v);
    else if(a=="--q") o.q=std::stoi(v); else if(a=="--streams") o.streams=std::stoi(v);
    else if(a=="--m") o.m=std::stoi(v); else if(a=="--n") o.n=std::stoi(v); else if(a=="--k") o.k=std::stoi(v);
    else if(a=="--k-launch") o.k_launch=v;
    else if(a=="--b-pad") o.bpad=std::stoi(v);
    else if(a=="--weights") {
      if(v=="column") o.weights=WeightFormat::Column;
      else if(v=="tile-n") o.weights=WeightFormat::TileN;
      else if(v=="tile-k") o.weights=WeightFormat::TileK;
      else throw std::runtime_error("unknown weight format");
    }
    else throw std::runtime_error("unknown option: "+a);
  }
  if(o.rounds<1 || o.repeats<1 || o.repeats>1024 || o.q<1 || o.q>128 || o.streams<1 || o.streams>16)
    throw std::runtime_error("invalid rounds/repeats/q/streams");
  if(o.cache!="warm" && o.cache!="evict") throw std::runtime_error("cache must be warm or evict");
  if(o.mode!="both" && o.mode!="eager" && o.mode!="graph") throw std::runtime_error("invalid mode");
  if(o.axis!="independent" && o.axis!="m" && o.axis!="n" && o.axis!="k") throw std::runtime_error("invalid axis");
  if(o.bpad<0 || o.bpad>4096 || o.bpad%16) throw std::runtime_error("B padding must be a multiple of 16 BF16 elements, <=4096");
  if(o.weights!=WeightFormat::Column && o.bpad!=0) throw std::runtime_error("tiled weights do not use affine B-row padding");
  if(o.k_launch!="separate" && o.k_launch!="fused") throw std::runtime_error("invalid K launch mode");
  if(o.k_launch=="fused" && (o.axis!="k" || o.streams!=1)) throw std::runtime_error("fused K producer requires axis=k, streams=1");
  return o;
}
struct Layout {
  int m,n,k,pk,pn,q,lda,ldb,ldc,bpad;
  std::string axis;
  WeightFormat weights;
  int full_n,full_k;
  size_t asize,bsize,csize,outsize;
  Layout(int M,int N,int K,int Q,std::string AX,int BP=0,WeightFormat WF=WeightFormat::Column):m(M),n(N),k(K),q(Q),bpad(BP),axis(std::move(AX)),weights(WF) {
    if(m<1 || n<1 || k<1 || m>32768 || n>32768 || k>32768) throw std::runtime_error("invalid matrix dimension");
    pk=(k+15)/16*16; pn=(n+15)/16*16; lda=ldb=pk; ldc=pn;
    asize=size_t(m)*pk; bsize=size_t(n)*pk; csize=size_t(m)*pn; outsize=csize;
    if(axis=="independent") {asize*=q;bsize*=q;csize*=q;outsize=csize;}
    else if(axis=="m") {asize*=q;csize*=q;outsize=csize;}
    else if(axis=="n") {bsize*=q;csize*=q;outsize=csize;ldc=pn*q;}
    else if(axis=="k") {lda=ldb=pk*q;asize*=q;bsize*=q;csize*=q;}
    full_n=n*(axis=="n"?q:1);full_k=ldb;
    ldb+=bpad;bsize=size_t(n)*ldb*((axis=="n" || axis=="independent")?q:1);
    if(weights!=WeightFormat::Column) bsize=packed_elements(full_n,full_k)*(axis=="independent"?q:1);
  }
  Args args(int i,const uint16_t* a,const uint16_t* b,float* c) const {
    size_t ao=0,bo=0,co=0;
    if(axis=="independent") {ao=size_t(i)*m*pk;bo=size_t(i)*n*ldb;co=size_t(i)*m*pn;}
    else if(axis=="m") {ao=size_t(i)*m*pk;co=size_t(i)*m*pn;}
    else if(axis=="n") {bo=size_t(i)*n*ldb;co=size_t(i)*pn;}
    else {ao=bo=size_t(i)*pk;co=size_t(i)*m*pn;}
    PackedInfo packed{full_n,full_k,axis=="n"?i*n:0,axis=="k"?i*pk:0,weights};
    if(weights!=WeightFormat::Column) bo=axis=="independent"?size_t(i)*packed_elements(full_n,full_k):0;
    return {a+ao,b+bo,c+co,m,n,pk,lda,ldb,ldc,packed};
  }
  size_t logical_b_index(int piece,int col,int t) const {
    const int global_n=col+((axis=="n" || axis=="independent")?piece*n:0);
    const int global_k=t+(axis=="k"?piece*pk:0);
    return size_t(global_n)*full_k+global_k;
  }
  double flops() const {return 2.0*m*n*k*q;}
  double compulsory() const {
    if(axis=="m") return 2.0*m*k*q+2.0*n*k+4.0*m*n*q;
    if(axis=="n") return 2.0*m*k+2.0*n*k*q+4.0*m*n*q;
    if(axis=="k") return 2.0*m*k*q+2.0*n*k*q+4.0*m*n;
    return q*(2.0*m*k+2.0*n*k+4.0*m*n);
  }
  double modeled() const {return compulsory()+(axis=="k" && q>1?8.0*m*n*q:0.0);}
};
struct Runtime {
  hipStream_t control{}; hipEvent_t begin{},end{},gate{};
  std::vector<hipStream_t> work;
  std::vector<hipEvent_t> done;
  std::vector<rocblas_handle> handles;
  std::vector<Buffer<char>> workspace;
  Buffer<uint4> eviction{(256ull<<20)/16};
  Buffer<uint32_t> sink{104*256};
  Runtime(int streams) {
    HIP(hipStreamCreateWithFlags(&control,hipStreamNonBlocking));
    HIP(hipEventCreate(&begin));HIP(hipEventCreate(&end));
    HIP(hipEventCreateWithFlags(&gate,hipEventDisableTiming));
    for(int i=0;i<streams;++i) {
      hipStream_t s=control; if(streams>1) HIP(hipStreamCreateWithFlags(&s,hipStreamNonBlocking)); work.push_back(s);
      hipEvent_t e{};HIP(hipEventCreateWithFlags(&e,hipEventDisableTiming));done.push_back(e);
      rocblas_handle h{};BLAS(rocblas_create_handle(&h));BLAS(rocblas_set_stream(h,s));
      BLAS(rocblas_set_atomics_mode(h,rocblas_atomics_not_allowed));
      workspace.emplace_back(64ull<<20);BLAS(rocblas_set_workspace(h,workspace.back().get(),workspace.back().size));handles.push_back(h);
    }
    fill_reader<<<104,256,0,control>>>(reinterpret_cast<uint32_t*>(eviction.get()),eviction.size*4);
    HIP(hipGetLastError());HIP(hipStreamSynchronize(control));
  }
  ~Runtime() {
    for(auto h:handles) (void)rocblas_destroy_handle(h);
    for(auto e:done) (void)hipEventDestroy(e);
    if(work.size()>1) for(auto s:work) (void)hipStreamDestroy(s);
    (void)hipEventDestroy(gate);(void)hipEventDestroy(begin);(void)hipEventDestroy(end);(void)hipStreamDestroy(control);
  }
  void evict() {reader<<<104,256,0,control>>>(eviction.get(),eviction.size/4,sink.get());HIP(hipStreamSynchronize(control));}
  void launch_blas(const Args& a,size_t stream) {
    const float alpha=1,beta=0;
    // Row-major C is column-major C^T: B^T[N,K] * A^T[K,M].
    BLAS(rocblas_gemm_ex(handles[stream],rocblas_operation_transpose,rocblas_operation_none,
        a.n,a.m,a.k,&alpha,a.b,rocblas_datatype_bf16_r,a.ldb,
        a.a,rocblas_datatype_bf16_r,a.lda,&beta,a.c,rocblas_datatype_f32_r,a.ldc,
        a.c,rocblas_datatype_f32_r,a.ldc,rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0));
  }
};
struct Matrices {
  Layout l; Buffer<uint16_t> a,b; Buffer<float> c,result;
  bool fused_split;
  Buffer<size_t> indices;Buffer<float> samples;
  std::vector<size_t> offsets;std::vector<double> gold;
  mutable std::vector<uint32_t> previous;
  Matrices(Layout layout,Runtime& r,bool full,bool fused=false):l(std::move(layout)),a(l.asize),b(l.bsize),c(l.csize+256),result(l.outsize+256),fused_split(fused && reduction()) {
    initialize<<<104,256,0,r.control>>>(a.get(),a.size,l.pk,l.k,123u);
    if(l.weights==WeightFormat::Column) initialize_b<<<104,256,0,r.control>>>(b.get(),b.size,l.pk,l.k,l.ldb,l.ldb-l.bpad);
    else initialize_packed<<<104,256,0,r.control>>>(b.get(),b.size,l.full_n,l.full_k,l.pk,l.k,l.weights);
    const size_t count=size_t(l.m)*l.n*(reduction()?1:l.q);
    const size_t checks=full?count:std::min<size_t>(count,128);
    for(size_t j=0;j<checks;++j) {
      const size_t logical=checks==count?j:(j==0?0:j==1?count-1:size_t(hash32(uint32_t(j)))*count/(uint64_t(1)<<32));
      const int piece=int(logical/(size_t(l.m)*l.n));const size_t cell=logical%(size_t(l.m)*l.n);
      const int row=int(cell/l.n),col=int(cell%l.n);double ref=0;
      for(int split=reduction()?0:piece;split<(reduction()?l.q:piece+1);++split) {
        const auto x=l.args(split,a.get(),b.get(),c.get());
        const size_t ao=size_t(x.a-a.get())+size_t(row)*x.lda;
        for(int t=0;t<l.k;++t) ref+=double(widen(bits(ao+t,123u)))*double(widen(bits(l.logical_b_index(split,col,t),456u)));
      }
      size_t offset=size_t(row)*l.pn+col;
      if(!reduction()) {const auto x=l.args(piece,a.get(),b.get(),c.get());offset=size_t(x.c-c.get())+size_t(row)*x.ldc+col;}
      offsets.push_back(offset);gold.push_back(ref);
    }
    indices=Buffer<size_t>(offsets.size());samples=Buffer<float>(offsets.size());
    HIP(hipMemcpyAsync(indices.get(),offsets.data(),indices.size*sizeof(size_t),hipMemcpyHostToDevice,r.control));
    HIP(hipGetLastError());HIP(hipStreamSynchronize(r.control));
  }
  bool reduction() const {return l.axis=="k" && l.q>1;}
  float* output() const {return reduction()?result.get():c.get();}
  void poison(Runtime& r) {HIP(hipMemsetAsync(c.get(),0xff,c.size*4,r.control));HIP(hipMemsetAsync(result.get(),0xff,result.size*4,r.control));HIP(hipStreamSynchronize(r.control));}
  void operation(Runtime& r,const Kernel* kernel) {
    if(fused_split) {
      if(!kernel) throw std::runtime_error("rocBLAS control has no fused producer");
      const auto args=l.args(0,a.get(),b.get(),c.get());
      if(l.weights==WeightFormat::Column) kernel->launch_split(args,l.q,l.pk,l.pk,size_t(l.m)*l.pn,r.control);
      else kernel->launch_packed(args,l.q,l.pk,size_t(l.m)*l.pn,r.control);
      reduce<<<std::min<size_t>((size_t(l.m)*l.n+255)/256,104*8),256,0,r.control>>>(c.get(),result.get(),l.m,l.n,l.pn,l.q);
      HIP(hipGetLastError());return;
    }
    if(r.work.size()>1) {
      HIP(hipEventRecord(r.gate,r.control));
      for(auto s:r.work) HIP(hipStreamWaitEvent(s,r.gate,0));
    }
    for(int i=0;i<l.q;++i) {
      const auto args=l.args(i,a.get(),b.get(),c.get());size_t s=size_t(i)%r.work.size();
      if(kernel) {
        if(l.weights==WeightFormat::Column) kernel->launch(args,r.work[s]);
        else kernel->launch_packed(args,1,0,0,r.work[s]);
      } else r.launch_blas(args,s);
    }
    if(r.work.size()>1) {
      for(size_t s=0;s<r.work.size();++s) {HIP(hipEventRecord(r.done[s],r.work[s]));HIP(hipStreamWaitEvent(r.control,r.done[s],0));}
    }
    if(reduction()) reduce<<<std::min<size_t>((size_t(l.m)*l.n+255)/256,104*8),256,0,r.control>>>(c.get(),result.get(),l.m,l.n,l.pn,l.q);
    HIP(hipGetLastError());
  }
  double validate(Runtime& r) const {
    HIP(hipStreamSynchronize(r.control));
    std::array<float,256> output_guard{};
    HIP(hipMemcpy(output_guard.data(),output()+l.outsize,output_guard.size()*4,hipMemcpyDeviceToHost));
    for(float v:output_guard)
      if(std::bit_cast<uint32_t>(v)!=0xffffffffu) throw std::runtime_error("output guard overwritten");
    if(reduction()) {
      std::array<float,256> guard{};HIP(hipMemcpy(guard.data(),c.get()+l.csize,guard.size()*4,hipMemcpyDeviceToHost));
      for(auto v:guard) if(std::bit_cast<uint32_t>(v)!=0xffffffffu) throw std::runtime_error("partial guard overwritten");
    }
    gather<<<std::min<size_t>((indices.size+255)/256,104),256,0,r.control>>>(output(),indices.get(),samples.get(),indices.size);
    HIP(hipGetLastError());HIP(hipStreamSynchronize(r.control));
    std::vector<float> values(samples.size);HIP(hipMemcpy(values.data(),samples.get(),values.size()*4,hipMemcpyDeviceToHost));
    std::vector<uint32_t> current;current.reserve(values.size());
    for(float v:values) current.push_back(std::bit_cast<uint32_t>(v));
    if(!previous.empty() && current!=previous) throw std::runtime_error("within-implementation nondeterminism");
    previous=std::move(current);
    double max_error=0;
    for(size_t j=0;j<gold.size();++j) {
      const double err=std::abs(double(values[j])-gold[j]);max_error=std::max(max_error,err);
      if(!std::isfinite(values[j]) || err>2e-4+3e-5*std::abs(gold[j]))
        throw std::runtime_error("FP64 oracle mismatch output_offset="+std::to_string(offsets[j])+" err="+std::to_string(err));
    }
    return max_error;
  }
};
struct Graph {
  hipGraph_t graph{};hipGraphExec_t exec{};
  Graph(Runtime& r,Matrices& a,const Kernel* k) {
    HIP(hipStreamBeginCapture(r.control,hipStreamCaptureModeGlobal));a.operation(r,k);
    HIP(hipStreamEndCapture(r.control,&graph));HIP(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  }
  ~Graph(){if(exec)(void)hipGraphExecDestroy(exec);if(graph)(void)hipGraphDestroy(graph);}
};
struct Time {double gpu,cpu;};
template<class Launch> Time measure(Runtime& r,Launch launch,int repetitions,const std::string& cache) {
  if(cache=="evict") r.evict(); // completed before both timers; not a proven hardware cache flush
  const auto start=std::chrono::steady_clock::now();HIP(hipEventRecord(r.begin,r.control));
  for(int i=0;i<repetitions;++i) launch();
  HIP(hipEventRecord(r.end,r.control));HIP(hipEventSynchronize(r.end));
  const auto finish=std::chrono::steady_clock::now();float ms=0;HIP(hipEventElapsedTime(&ms,r.begin,r.end));
  return {double(ms)*1000/repetitions,std::chrono::duration<double,std::micro>(finish-start).count()/repetitions};
}
void header() {
  std::puts("phase,m,n,k,pk,axis,q,streams,impl,cache,mode,k_launch,bpad,weights,round,repeats,gpu_us,cpu_us,tflops,compulsory_bytes,modeled_bytes,padded_storage_bytes,ctas,producer_launches,threads,registers,lds_bytes,scratch_bytes,blocks_per_cu,tile_flops,weight_bytes,weight_GBps,max_abs_error");
}
void report(const Options& o,const Matrices& a,const Kernel* k,const char* mode,int round,int reps,Time t,double error,int streams) {
  const auto& l=a.l; const auto grid=k?size_t((l.m+k->tm-1)/k->tm)*((l.n+k->tn-1)/k->tn)*l.q:0;
  const double tile_flops=k && std::string(k->name).starts_with("ck_")?2.0*grid*k->tm*k->tn*((l.pk+k->tk-1)/k->tk*k->tk):0;
  const size_t storage=(l.asize+l.bsize)*2+(l.csize+(a.reduction()?l.outsize:0))*4;
  const auto* attrs=k?(l.weights!=WeightFormat::Column?&k->packed_attributes:a.fused_split?&k->split_attributes:&k->attributes):nullptr;
  const char* format=l.weights==WeightFormat::Column?"column":l.weights==WeightFormat::TileN?"tile-n":"tile-k";
  const double weight_bytes=2.0*l.n*l.k*(l.axis=="m"?1:l.q);
  std::printf("%s,%d,%d,%d,%d,%s,%d,%d,%s,%s,%s,%s,%d,%s,%d,%d,%.6f,%.6f,%.6f,%.0f,%.0f,%zu,%zu,%d,%d,%d,%zu,%zu,%d,%.0f,%.0f,%.6f,%.9g\n",
      o.phase.c_str(),l.m,l.n,l.k,l.pk,l.axis.c_str(),l.q,streams,k?k->name:"rocblas",
      o.cache.c_str(),mode,a.fused_split?"fused":"separate",l.bpad,format,round,reps,t.gpu,t.cpu,l.flops()/t.gpu/1e6,l.compulsory(),l.modeled(),storage,grid,
      a.fused_split?1:l.q,k?k->threads:0,attrs?attrs->numRegs:0,attrs?attrs->sharedSizeBytes:0,attrs?attrs->localSizeBytes:0,
      k?(l.weights!=WeightFormat::Column?k->packed_blocks_per_cu:a.fused_split?k->split_blocks_per_cu:k->blocks_per_cu):0,tile_flops,weight_bytes,weight_bytes/t.gpu/1000,error);
  std::fflush(stdout);
}
std::vector<std::array<int,3>> shapes(const Options& o) {
  if(!o.shape_file.empty()) {
    std::ifstream file(o.shape_file);if(!file) throw std::runtime_error("cannot read shapes file");
    std::vector<std::array<int,3>> s;std::string line;
    while(std::getline(file,line)) {if(line.empty() || line[0]=='#') continue;std::replace(line.begin(),line.end(),',',' ');std::array<int,3> v{};if(std::sscanf(line.c_str(),"%d %d %d",&v[0],&v[1],&v[2])!=3) throw std::runtime_error("bad shape line");s.push_back(v);}return s;
  }
  if(o.m && o.n && o.k) return {{{o.m,o.n,o.k}}};
  if(o.phase=="test") return {{{1,1,1},{1,16,1},{3,20,65},{16,64,64},{31,36,97},{31,33,97},{64,128,128},{128,256,256}}};
  if(o.phase=="roof") return {{{1024,4096,4096},{2048,4096,8192},{4096,8192,8192},{8192,8192,8192}}};
  if(o.phase=="quick") return {{{16,256,256},{64,256,256},{256,256,256},{64,1024,1024},{256,1024,1024},{1024,1024,1024},{256,4096,4096},{1024,4096,4096}}};
  if(o.phase=="sweep") {
    std::vector<std::array<int,3>> s;
    for(int m:{1,2,4,8,16,32,64,128,256,512,1024,2048}) for(int n:{16,32,64,128,256,512,1024,2048,4096,8192}) for(int k:{16,32,64,128,256,512,1024,2048,4096,8192}) s.push_back({m,n,k});return s;
  }
  throw std::runtime_error("unknown phase (use test/roof/quick/sweep or explicit m,n,k)");
}
void arithmetic_reference(Runtime& r,int rounds) {
  Buffer<float> output{832*256};
  const auto start=std::chrono::steady_clock::now();
  do {
    for(int i=0;i<8;++i) mfma_reference<<<208,256,0,r.control>>>(output.get(),4096);
    HIP(hipStreamSynchronize(r.control));
  } while(std::chrono::steady_clock::now()-start<std::chrono::seconds(2));
  for(int blocks:{104,208,416,832}) for(int round=0;round<rounds;++round) {
    const auto t=measure(r,[&]{mfma_reference<<<blocks,256,0,r.control>>>(output.get(),4096);},16,"warm");
    std::vector<float> values(size_t(blocks)*256);HIP(hipMemcpy(values.data(),output.get(),values.size()*4,hipMemcpyDeviceToHost));
    for(float v:values) if(v!=40.0f) throw std::runtime_error("MFMA register-reference mismatch");
    // One 16x16x16 instruction per wave = 8192 FLOPs, four independent chains.
    const double flops=double(blocks)*4*4096*4*8192;
    std::fprintf(stderr,"MFMA_REFERENCE blocks=%d round=%d tflops=%.3f gpu_us=%.3f\n",blocks,round,flops/t.gpu/1e6,t.gpu);
  }
}
void packing_tests(Runtime& r,const std::vector<Kernel>& kernels) {
  for(auto f:{WeightFormat::TileN,WeightFormat::TileK}) {
    for(auto dims:std::vector<std::array<int,2>>{{1,1},{17,65},{32,128},{67,259},{65,256}}) {
      const int n=dims[0],k=dims[1],pk=(k+15)/16*16;
      const int nt=(n+31)/32,kt=(pk+127)/128;
      Buffer<uint16_t> device{packed_elements(n,pk)};
      initialize_packed<<<104,256,0,r.control>>>(device.get(),device.size,n,pk,pk,k,f);
      HIP(hipGetLastError());HIP(hipStreamSynchronize(r.control));
      std::vector<uint16_t> expected(device.size,0),actual(device.size);
      // Independent CPU forward scatter; GPU initializer traverses packed
      // addresses and inverts the mapping. No shared offset helper is used.
      for(int row=0;row<n;++row) for(int col=0;col<k;++col) {
        const int nr=row/32,kr=col/128,ni=row-nr*32,ki=col-kr*128;
        const size_t block=f==WeightFormat::TileN?size_t(nr)*kt+kr:size_t(kr)*nt+nr;
        expected[(block*32+ni)*128+ki]=bits(size_t(row)*pk+col,456u);
      }
      HIP(hipMemcpy(actual.data(),device.get(),device.size*2,hipMemcpyDeviceToHost));
      if(actual!=expected) throw std::runtime_error("independent CPU/GPU packed bytes mismatch");
    }
    Matrices parent(Layout(3,64,1008,1,"k",0,f),r,true);
    std::vector<uint16_t> before(parent.b.size),after(parent.b.size);
    HIP(hipMemcpy(before.data(),parent.b.get(),before.size()*2,hipMemcpyDeviceToHost));
    Buffer<float> partial{size_t(63)*3*64+256};
    for(const auto& kernel:kernels) if(std::string(kernel.name).starts_with("ck_")) for(int q:{1,2,3,7,9,21,63}) {
      auto args=parent.l.args(0,parent.a.get(),parent.b.get(),partial.get());args.k=1008/q;
      if(!kernel.supported(args)) throw std::runtime_error("reuse shape unexpectedly unsupported");
      parent.previous.clear();parent.poison(r);HIP(hipMemsetAsync(partial.get(),0xff,partial.size*4,r.control));
      kernel.launch_packed(args,q,args.k,3*64,r.control);
      reduce<<<1,256,0,r.control>>>(partial.get(),parent.c.get(),3,64,64,q);
      parent.validate(r);
    }
    HIP(hipMemcpy(after.data(),parent.b.get(),after.size()*2,hipMemcpyDeviceToHost));
    if(before!=after) throw std::runtime_error("packed weights mutated across split schedules");
    std::fprintf(stderr,"PACK_PASS format=%s independent tails/bytes; same immutable blob reused for q=1,2,3,7,9,21,63 across all kernels\n",f==WeightFormat::TileN?"tile-n":"tile-k");
  }
}
void reader_reference(Runtime& r,int rounds) {
  for(int i=0;i<3;++i) reader<<<104,256,0,r.control>>>(r.eviction.get(),r.eviction.size,r.sink.get());
  HIP(hipStreamSynchronize(r.control));
  for(int round=0;round<rounds;++round) {
    const auto t=measure(r,[&]{reader<<<104,256,0,r.control>>>(r.eviction.get(),r.eviction.size,r.sink.get());},16,"warm");
    std::vector<uint32_t> sums(r.sink.size);HIP(hipMemcpy(sums.data(),r.sink.get(),sums.size()*4,hipMemcpyDeviceToHost));
    uint32_t actual=std::accumulate(sums.begin(),sums.end(),uint32_t(0));
    const uint64_t words=r.eviction.size*4;
    const uint32_t expected=uint32_t(words*(words-1)/2)*0x9e3779b9u+uint32_t(words)*17u;
    if(actual!=expected) throw std::runtime_error("reader checksum mismatch");
    std::fprintf(stderr,"READ_REFERENCE round=%d useful_GBps=%.3f gpu_us=%.3f bytes=%zu\n",round,double(r.eviction.size*16)/t.gpu/1000,t.gpu,r.eviction.size*16);
  }
  for(int round=0;round<rounds;++round) {
    const auto t=measure(r,[&]{empty<<<1,64,0,r.control>>>();},16,"warm");
    std::fprintf(stderr,"EMPTY_REFERENCE round=%d gpu_us=%.3f cpu_us=%.3f\n",round,t.gpu,t.cpu);
  }
}
int main(int argc,char** argv) {
  try {
    const auto o=parse(argc,argv);HIP(hipSetDevice(0));hipDeviceProp_t props{};HIP(hipGetDeviceProperties(&props,0));
    if(props.warpSize!=64 || std::string(props.gcnArchName).find("gfx90a")==std::string::npos) throw std::runtime_error("requires gfx90a wave64");
    char bus[64]{};HIP(hipDeviceGetPCIBusId(bus,sizeof(bus),0));
    size_t version_size=0;BLAS(rocblas_get_version_string_size(&version_size));
    std::vector<char> version(version_size);BLAS(rocblas_get_version_string(version.data(),version.size()));
    std::fprintf(stderr,"device=0 pci=%s arch=%s CUs=%d clocks_khz=%d/%d rocblas=%s model=none context=none BF16xBF16->FP32\n",bus,props.gcnArchName,props.multiProcessorCount,props.clockRate,props.memoryClockRate,version.data());
    std::vector<Kernel> kernels;register_16x16(kernels);register_16x32(kernels);register_16x16_k128(kernels);register_16x32_k128(kernels);register_16x32_k256(kernels);register_16x64(kernels);register_32x32(kernels);register_64x64(kernels);register_64x128(kernels);register_128x128(kernels);register_128x256(kernels);register_256x256(kernels);
    register_simt(kernels);
    const int streams=std::min(o.streams,o.q);Runtime r(streams);
    if(o.phase=="pack-test") {packing_tests(r,kernels);return 0;}
    if(o.phase=="roof") {arithmetic_reference(r,o.rounds);reader_reference(r,o.rounds);}
    else if(o.phase!="test") {
      const auto warm_start=std::chrono::steady_clock::now();
      do { for(int i=0;i<16;++i) reader<<<104,256,0,r.control>>>(r.eviction.get(),r.eviction.size,r.sink.get());HIP(hipStreamSynchronize(r.control)); }
      while(std::chrono::steady_clock::now()-warm_start<std::chrono::seconds(2));
    }
    header();auto all_shapes=shapes(o);std::mt19937 rng(1431);std::shuffle(all_shapes.begin(),all_shapes.end(),rng);
    for(const auto shape:all_shapes) {
      Matrices a(Layout(shape[0],shape[1],shape[2],o.q,o.axis,o.bpad,o.weights),r,o.phase=="test",o.k_launch=="fused");
      std::vector<const Kernel*> candidates;
      if(o.impl=="all" || o.impl=="rocblas") candidates.push_back(nullptr);
      for(const auto& k:kernels) if(o.impl=="all" || o.impl==k.name || (o.impl=="ck" && std::string(k.name).starts_with("ck_"))) candidates.push_back(&k);
      if(candidates.empty()) throw std::runtime_error("unknown implementation");
      std::shuffle(candidates.begin(),candidates.end(),rng);
      for(const auto* k:candidates) {
        if(a.l.weights!=WeightFormat::Column && !k) {std::fprintf(stderr,"UNSUPPORTED rocblas tiled weights (no timed repack/fallback)\n");continue;}
        if(a.fused_split && !k) {std::fprintf(stderr,"UNSUPPORTED rocblas fused producer (separate-launch control only)\n");continue;}
        bool supported=true;for(int i=0;i<a.l.q;++i) if(k && !k->supported(a.l.args(i,a.a.get(),a.b.get(),a.c.get()))) supported=false;
        if(!supported) {std::fprintf(stderr,"UNSUPPORTED %s M=%d N=%d K=%d\n",k->name,a.l.m,a.l.n,a.l.k);continue;}
        a.previous.clear();a.poison(r);for(int i=0;i<3;++i) a.operation(r,k);const double error=a.validate(r);
        std::unique_ptr<Graph> graph;
        if(o.mode!="eager") {
          graph=std::make_unique<Graph>(r,a,k);a.poison(r);
          HIP(hipGraphLaunch(graph->exec,r.control));a.validate(r); // capture itself never executes the work
        }
        for(int round=0;round<o.rounds;++round) {
          std::vector<int> repetition_counts{1};
          if(o.repeats>1 && o.cache=="warm" && o.phase!="test") repetition_counts.push_back(o.repeats);
          for(int reps:repetition_counts) {
            if(o.mode!="graph" || o.phase=="test") {auto t=measure(r,[&]{a.operation(r,k);},reps,o.cache);report(o,a,k,"eager",round,reps,t,error,streams);}
            if(graph) {auto t=measure(r,[&]{HIP(hipGraphLaunch(graph->exec,r.control));},reps,o.cache);report(o,a,k,"graph",round,reps,t,error,streams);}
          }
        }
        a.validate(r); // final timed output, no tolerance-free cross-implementation claim
      }
    }
    return 0;
  } catch(const std::exception& e) {std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
}
