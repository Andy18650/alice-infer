// PERFORMANCE ONLY: each B-only hot-loop MFMA is replaced by s_nop 0.
// B preparation and operand lifetimes stay; no output readback or numerical tests.
#ifndef OMIT_PAIR_WAIT
#define OMIT_PAIR_WAIT 0
#endif
#if OMIT_PAIR_WAIT
#define WAIT8 ""
#define WAIT16 ""
#else
#define WAIT8 "s_waitcnt vmcnt(%7)\n\t"
#define WAIT16 "s_waitcnt vmcnt(%4)\n\t"
#endif
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while(0)
using H4=_Float16 __attribute__((ext_vector_type(4)));
using U2=uint32_t __attribute__((ext_vector_type(2)));
using F16=float __attribute__((ext_vector_type(16)));
constexpr int kA=13,kB=17,kPadding=18;
constexpr int kUnroll=32;

__device__ __forceinline__ uint32_t load8(const uint32_t* p) {
  uint32_t x; asm volatile("global_load_dword %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=&v"(x) : "v"(p) : "memory"); return x;
}
__device__ __forceinline__ H4 load16(const U2* p) {
  U2 x; asm volatile("global_load_dwordx2 %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=&v"(x) : "v"(p) : "memory"); return __builtin_bit_cast(H4,x);
}
__device__ __forceinline__ void issue_b8(const uint32_t* bp,uint32_t& b) {
  asm volatile("global_load_dword %0, %1, off" : "=&v"(b) : "v"(bp) : "memory");
}
__device__ __forceinline__ void issue_b16(const U2* bp,U2& b) {
  asm volatile("global_load_dwordx2 %0, %1, off" : "=&v"(b) : "v"(bp) : "memory");
}
__device__ __forceinline__ H4 convert(uint32_t w,H4 keep) {
  uint32_t lo,hi;
  asm volatile("v_perm_b32 %0, 0, %2, %3\n\tv_perm_b32 %1, 0, %2, %4\n\t"
      "v_pk_ashrrev_i16 %0, 1, %0 op_sel_hi:[0,1]\n\t"
      "v_pk_ashrrev_i16 %1, 1, %1 op_sel_hi:[0,1]\n\t"
      "v_and_b32 %0, %5, %0\n\tv_and_b32 %1, %5, %1\n\t"
      "v_pk_mul_f16 %0, %0, %6\n\tv_pk_mul_f16 %1, %1, %6"
      : "=&v"(lo),"=&v"(hi) : "v"(w),"s"(0x01040004u),"s"(0x03040204u),"s"(0xbfffbfffu),"s"(0x5c005c00u),"v"(keep));
  return __builtin_bit_cast(H4,(U2{lo,hi}));
}
// Optional wait and first B reads stay together; no A copies are needed.
template<int Depth> __device__ __forceinline__ H4 take_b8(uint32_t b,H4 keepa,H4 keepb) {
  uint32_t lo,hi;
  asm volatile(WAIT8 "v_perm_b32 %0, 0, %2, %3\n\tv_perm_b32 %1, 0, %2, %4\n\t"
      "v_pk_ashrrev_i16 %0, 1, %0 op_sel_hi:[0,1]\n\t"
      "v_pk_ashrrev_i16 %1, 1, %1 op_sel_hi:[0,1]\n\t"
      "v_and_b32 %0, %5, %0\n\tv_and_b32 %1, %5, %1\n\t"
      "v_pk_mul_f16 %0, %0, %6\n\tv_pk_mul_f16 %1, %1, %6"
      : "=&v"(lo),"=&v"(hi)
      : "v"(b),"s"(0x01040004u),"s"(0x03040204u),"s"(0xbfffbfffu),"s"(0x5c005c00u),"n"(Depth-1),"v"(keepa),"v"(keepb) : "memory");
  return __builtin_bit_cast(H4,(U2{lo,hi}));
}
template<int Depth> __device__ __forceinline__ H4 take_b16(U2 b,H4 keepa,H4 keepb) {
  uint32_t lo,hi;
  asm volatile(WAIT16 "v_mov_b32 %0, %2\n\tv_mov_b32 %1, %3"
               : "=&v"(lo),"=&v"(hi)
               : "v"(b[0]),"v"(b[1]),"n"(Depth-1),"v"(keepa),"v"(keepb) : "memory");
  return __builtin_bit_cast(H4,(U2{lo,hi}));
}
template<int D,class T> __device__ __forceinline__ void drain_b(const T* b) {
#define P(I) "v"(b[I])
  if constexpr(D==1) asm volatile("s_waitcnt vmcnt(0)" : : P(0) : "memory");
  if constexpr(D==2) asm volatile("s_waitcnt vmcnt(0)" : : P(0),P(1) : "memory");
  if constexpr(D==4) asm volatile("s_waitcnt vmcnt(0)" : : P(0),P(1),P(2),P(3) : "memory");
  if constexpr(D==8) asm volatile("s_waitcnt vmcnt(0)" : : P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7) : "memory");
  if constexpr(D==16) asm volatile("s_waitcnt vmcnt(0)" : : P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7),P(8),P(9),P(10),P(11),P(12),P(13),P(14),P(15) : "memory");
#undef P
}

// Cache pre-read stays in THIS dispatch, before the core-clock interval.
// Repeating a separate dispatch did not establish a hot-cache control.
template<int Format> __device__ __forceinline__ void warm_inputs(
    const U2* a,const uint32_t* b8,const U2* b16,int records) {
  uint32_t checksum=0;
#pragma clang loop unroll(disable)
  for(int s=0;s<records;++s) {
    U2 x=__builtin_bit_cast(U2,load16(a+size_t(s)*64));
    checksum^=x[0]^x[1];
    if constexpr(Format==1) checksum^=load8(b8+size_t(s)*64);
    else { U2 y=__builtin_bit_cast(U2,load16(b16+size_t(s)*64)); checksum^=y[0]^y[1]; }
  }
  asm volatile("" : : "v"(checksum) : "memory");
  __builtin_amdgcn_sched_barrier(0);
}

// Format 0=FP16 B, 1=FP8 B. A is resident; only B has a raw load ring.
// Fixed fragment-zero B, grid=(1,1), block=256, unroll32.
template<int Format,int FixedK,int Depth,int Unroll>
__global__ __launch_bounds__(256) void fixed_kernel(
    const U2* activation,const uint32_t* stream8,const U2* stream16,
    float* output,uint64_t* stamps,int steps,int warm) {
  static_assert(FixedK==1 && Depth>0 && Unroll%Depth==0);
  int tid=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
  int lane=threadIdx.x%64,wave=tid/64;
  int nt=blockIdx.x*4+threadIdx.x/64;
  size_t base=size_t(nt)*(steps+kPadding)*64+lane;
  size_t abase=size_t(blockIdx.y)*(steps+kPadding)*64+lane;
  if(warm) warm_inputs<Format>(activation+abase,stream8+base,stream16+base,FixedK?1:steps+kPadding);
  H4 a=load16(activation+abase),b[2]; U2 q16[Depth]; uint32_t q8[Depth];
  constexpr int second=0;
  if constexpr(Format==0) { b[0]=load16(stream16+base); b[1]=load16(stream16+base+second); }
  else { b[0]=convert(load8(stream8+base),a); b[1]=convert(load8(stream8+base+second),b[0]); }
  F16 c={};
  asm volatile("s_waitcnt vmcnt(0)" : "+a"(c) : "v"(a),"v"(b[0]),"v"(b[1]));
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for(int r=0;r<Depth-1;++r) {
    if constexpr(Format==0) issue_b16(stream16+base,q16[r]);
    else issue_b8(stream8+base,q8[r]);
  }
  __builtin_amdgcn_sched_barrier(0);
  uint64_t begin=clock64(),wall_begin=wall_clock64();
#pragma clang loop unroll(disable)
  for(int i=0;i<steps;i+=Unroll) {
#pragma unroll
    for(int j=0;j<Unroll;++j) {
      // Dummy operands preserve compiler-visible liveness, not hardware reads.
      asm volatile("s_nop 0" : "+a"(c) : "v"(a),"v"(b[j%2]) : "memory");
      __builtin_amdgcn_sched_barrier(0);
      int tail=(j+Depth-1)%Depth,head=j%Depth;
      if constexpr(Format==0) issue_b16(stream16+base,q16[tail]);
      else issue_b8(stream8+base,q8[tail]);
      __builtin_amdgcn_sched_barrier(0);
      if constexpr(Format==0) b[j%2]=take_b16<Depth>(q16[head],a,b[(j+1)%2]);
      else b[j%2]=take_b8<Depth>(q8[head],a,b[(j+1)%2]);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  if constexpr(Format==0) drain_b<Depth>(q16);
  else drain_b<Depth>(q8);
  __builtin_amdgcn_sched_barrier(0);
  uint64_t end=clock64(),wall_end=wall_clock64();
  if(lane==0) { stamps[4*wave]=begin; stamps[4*wave+1]=end; stamps[4*wave+2]=wall_begin; stamps[4*wave+3]=wall_end; }
  for(int r=0;r<16;++r) output[tid*16+r]=c[r];
}

// Pure resident FP16 MFMA anchor: same four-wave CTA, no timed memory or
// conversion. Performance anchor only; computed output is not inspected.
__global__ __launch_bounds__(256) void resident_kernel(
    const U2* activation,const U2* stream16,float* output,uint64_t* stamps,int steps,int warm) {
  int tid=threadIdx.x,lane=tid%64,wave=tid/64;
  size_t base=size_t(wave)*(steps+kPadding)*64+lane;
  if(warm) warm_inputs<0>(activation+lane,nullptr,stream16+base,1);
  H4 a=load16(activation+lane),b=load16(stream16+base);
  F16 c={};
  asm volatile("s_waitcnt vmcnt(0)" : "+a"(c) : "v"(a),"v"(b));
  __builtin_amdgcn_sched_barrier(0);
  uint64_t begin=clock64(),wall_begin=wall_clock64();
#pragma clang loop unroll(disable)
  for(int i=0;i<steps;i+=kUnroll) {
#pragma unroll
    for(int j=0;j<kUnroll;++j) c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,b,c,0,0,0);
  }
  uint64_t end=clock64(),wall_end=wall_clock64();
  if(lane==0) { stamps[4*wave]=begin; stamps[4*wave+1]=end; stamps[4*wave+2]=wall_begin; stamps[4*wave+3]=wall_end; }
  for(int r=0;r<16;++r) output[tid*16+r]=c[r];
}

using U4=uint32_t __attribute__((ext_vector_type(4)));
__host__ __device__ inline U4 eviction_value(size_t t) {
  uint32_t x=uint32_t(t)*1664525u+1013904223u;
  return U4{x,x^0x9e3779b9u,(x<<13)|(x>>19),x+uint32_t(t>>8)};
}
__global__ void fill_eviction(U4* p,size_t n) {
  size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t<n) p[t]=eviction_value(t);
}
// Read every 16-byte record. The wide load and consumption are explicit;
// this kernel is always completed BEFORE the measured single-CTA launch.
__global__ void evict_cache(const U4* p,size_t n,uint32_t* sink) {
  size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x,stride=size_t(gridDim.x)*blockDim.x;
  uint32_t sum=0;
#pragma clang loop unroll(disable)
  for(size_t t=tid;t<n;t+=stride) {
    U4 x;
    asm volatile("global_load_dwordx4 %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=&v"(x) : "v"(p+t) : "memory");
    sum^=x[0]^x[1]^x[2]^x[3];
  }
  sink[tid]=sum;
}

// One actual A row stream and four private B column streams, all K-varying.
__global__ void fill_data(const U2* as,const uint32_t* bs8,const U2* bs16,
                         U2* ao,uint32_t* bo8,U2* bo16,size_t nb,size_t na,
                         int pitch,int ae,int be) {
  size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t<na) {
    int mt=int(t/(size_t(pitch)*64)),s=int(t/64%pitch),lane=int(t%64);
    int ap=(5*s+s/kA+ae*7+3*mt)%kA; ao[t]=as[ap*64+lane];
  }
  if(t<nb) {
    int lane=int(t%64),wave=int(t/(size_t(pitch)*64)),s=int(t/64%pitch);
    int bp=(s+s/kB+wave*7+be*3)%kB;
    bo8[t]=bs8[bp*64+lane]; bo16[t]=bs16[bp*64+lane];
  }
}
float decode(uint8_t x) {
  int e=(x>>3)&15,m=x&7; float v=e?std::ldexp(float(8+m),e-10):std::ldexp(float(m),-9); return x&128?-v:v;
}
template<class T> struct Device {
  T* p=nullptr;
  explicit Device(size_t n) { HIP(hipMalloc(&p,n*sizeof(T))); }
  ~Device() { if(p) (void)hipFree(p); }
  Device(const Device&)=delete;
  void put(const std::vector<T>& v) { HIP(hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice)); }
  std::vector<T> get(size_t n) const { std::vector<T> v(n); HIP(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost)); return v; }
};
using Launch=void(*)(const U2*,const uint32_t*,const U2*,float*,uint64_t*,int,int,int);
struct Case { int format,fixed_k,depth; Launch launch; };
template<int F,int K,int D> void launch(const U2* a,const uint32_t* q,const U2* h,float* y,uint64_t* st,int steps,int blocks,int warm) {
  (void)blocks;
  fixed_kernel<F,K,D,kUnroll><<<1,256>>>(a,q,h,y,st,steps,warm); HIP(hipGetLastError());
}
void launch_resident(const U2* a,const uint32_t* q,const U2* h,float* y,uint64_t* st,int steps,int blocks,int warm) {
  (void)q; (void)blocks;
  resident_kernel<<<1,256>>>(a,h,y,st,steps,warm); HIP(hipGetLastError());
}
std::vector<Case> cases() {
  std::vector<Case> v;
#define ADD(F,K,D) v.push_back({F,K,D,launch<F,K,D>})
#define DEPTHS(F,K) ADD(F,K,1); ADD(F,K,2); ADD(F,K,4); ADD(F,K,8); ADD(F,K,16)
  DEPTHS(0,1); DEPTHS(1,1);
#undef DEPTHS
#undef ADD
  v.push_back({-1,1,0,launch_resident});
  return v;
}
struct Events {
  hipEvent_t begin{},end{};
  Events() { HIP(hipEventCreate(&begin)); HIP(hipEventCreate(&end)); }
  ~Events() { (void)hipEventDestroy(begin); (void)hipEventDestroy(end); }
};

int main(int argc,char** argv) try {
  int steps=argc>1?std::stoi(argv[1]):4096,rounds=argc>2?std::stoi(argv[2]):9;
  unsigned seed=argc>3?unsigned(std::stoul(argv[3])):1729;
  int evict_mib=argc>4?std::stoi(argv[4]):256;
  if(steps<=0 || steps>8192 || steps%32 || rounds<=0 || evict_mib<32 || evict_mib>1024) throw std::runtime_error("steps: multiple of 32, 1..8192; rounds positive; eviction MiB 32..1024");
  HIP(hipSetDevice(0)); hipDeviceProp_t prop{}; HIP(hipGetDeviceProperties(&prop,0));
  int wall_rate=0; HIP(hipDeviceGetAttribute(&wall_rate,hipDeviceAttributeWallClockRate,0));
  constexpr int blocks=1,threads=256,waves=4;
  int l2_bytes=0; HIP(hipDeviceGetAttribute(&l2_bytes,hipDeviceAttributeL2CacheSize,0));
  int evict_blocks=prop.multiProcessorCount*8,evict_threads=evict_blocks*256;
  size_t evict_records=(size_t(evict_mib)<<20)/sizeof(U4);
  size_t outputs=size_t(threads)*16,tile_records=size_t(steps+kPadding)*64;
  size_t na=tile_records,nb=tile_records*waves;
  std::fprintf(stderr,"device=%s arch=%s CUs=%d timed_grid=1x1 block=256 waves=4 steps=%d rounds=%d seed=%u unroll=%d reported_L2_bytes=%d A_bytes=%zu B8_bytes=%zu B16_bytes=%zu eviction_bytes=%zu eviction_grid=%dx256; native O3, CPU+memory NUMA0 externally, no model/context\n",prop.name,prop.gcnArchName,prop.multiProcessorCount,steps,rounds,seed,kUnroll,l2_bytes,na*8,nb*4,nb*8,evict_records*sizeof(U4),evict_blocks);
  std::vector<U2> as(kA*64),bs16(kB*64); std::vector<uint32_t> bs8(kB*64);
  for(int m=0;m<kA;++m) for(int lane=0;lane<64;++lane) {
    H4 v;
    for(int t=0;t<4;++t) {
      int row=lane%32,k=4*(lane/32)+t;
      v[t]=_Float16(((row*3+k*5+m*(2+row%3)+(m*k)%5)%7-3)*0.125f);
    }
    as[m*64+lane]=std::bit_cast<U2>(v);
  }
  for(int m=0;m<kB;++m) for(int lane=0;lane<64;++lane) {
    H4 v; uint32_t word=0;
    for(int t=0;t<4;++t) {
      int x=(lane*19+t*7+m*11)%32;
      uint8_t byte=uint8_t(x|((lane+t+m)%3==0?128:0));
      word|=uint32_t(byte)<<(8*t); v[t]=_Float16(decode(byte));
    }
    bs8[m*64+lane]=word; bs16[m*64+lane]=std::bit_cast<U2>(v);
  }
  Device<U2> das(as.size()),dbs16(bs16.size()),activation(na),b16(nb);
  Device<uint32_t> dbs8(bs8.size()),b8(nb); Device<float> out(outputs+32); Device<uint64_t> stamps(waves*4);
  das.put(as); dbs8.put(bs8); dbs16.put(bs16);
  Device<U4> trash(evict_records); Device<uint32_t> trash_sink(evict_threads);
  fill_eviction<<<(evict_records+255)/256,256>>>(trash.p,evict_records); HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
  auto evict=[&]() { evict_cache<<<evict_blocks,256>>>(trash.p,evict_records,trash_sink.p); HIP(hipGetLastError()); HIP(hipDeviceSynchronize()); };
  fill_data<<<(nb+255)/256,256>>>(das.p,dbs8.p,dbs16.p,activation.p,b8.p,b16.p,nb,na,steps+kPadding,0,0);
  HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
  auto all=cases();
  auto run=[&](const Case& c,int warm=0) { c.launch(activation.p,b8.p,b16.p,out.p+16,stamps.p,steps,blocks,warm); };
  std::fprintf(stderr,"PERFORMANCE ONLY B-only loads, MFMA replaced by s_nop 0; omit_pair_wait=%d; no output readback or numerical tests; prologue waits and final B drain retained\n",OMIT_PAIR_WAIT);
  auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(std::chrono::steady_clock::now()<until) { for(auto& c:all) if(c.format<0 || (c.format==1 && c.depth==8)) run(c); HIP(hipDeviceSynchronize()); }
  Events events; std::mt19937 rng(seed);
  // cache=0: in-kernel full-input pre-read, outside the core-clock body.
  // cache=1: completed full-device eviction reader; no in-kernel pre-read.
  // ONE probe dispatch in both. Event time includes the warm pre-read, so
  // it is launch_ms, not a comparable timed-body throughput measurement.
  std::vector<std::pair<int,int>> order;
  for(size_t i=0;i<all.size();++i) for(int cache: {0,1}) order.emplace_back(int(i),cache);
  std::puts("round,format,fixed_k,depth,cache,steps,eviction_MiB,launch_ms,body_us_mean,cycles_per_step,min_wave_cycles,max_wave_cycles,core_MHz");
  for(int round=0;round<rounds;++round) {
    std::shuffle(order.begin(),order.end(),rng);
    for(auto [index,cache]:order) {
      auto& c=all[index];
      if(cache) evict();
      HIP(hipEventRecord(events.begin)); run(c,cache==0);
      HIP(hipEventRecord(events.end)); HIP(hipEventSynchronize(events.end));
      float ms; HIP(hipEventElapsedTime(&ms,events.begin,events.end));
      auto st=stamps.get(waves*4); double cycles=0,ticks=0; std::vector<double> intervals;
      for(int w=0;w<waves;++w) { double n=double(st[4*w+1]-st[4*w]); cycles+=n; ticks+=double(st[4*w+3]-st[4*w+2]); intervals.push_back(n/steps); }
      std::sort(intervals.begin(),intervals.end());
      double body_us=ticks/waves*1000/wall_rate;
      std::printf("%d,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.3f\n",round,c.format,c.fixed_k,c.depth,cache,steps,evict_mib,ms,body_us,cycles/(waves*double(steps)),intervals.front(),intervals.back(),cycles/ticks*wall_rate/1000);
    }
    std::fflush(stdout);
  }
  std::fprintf(stderr,"COMPLETE %zu kernels x 2 cache protocols x %d rounds, one timed CTA/pass; PERFORMANCE ONLY no math checks\n",all.size(),rounds);
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
