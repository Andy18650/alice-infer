// CU load-delivery ceiling probe. The fp8-w8a16 fixed-K diagnostic measured
// ~31.96/31.68 logical load bytes per CTA core cycle with all loads cache-hot
// and concluded "each CU can receive at most ~32 bytes per cycle" as a clue.
// That probe entangled loads with MFMA issue, FP8 conversion VALUs, A copies
// and paired waits. This standalone probe removes all of that: it times pure
// load rings (issue -> ring wait -> one checksum XOR) on ONE CTA and sweeps
// load width, resident waves per CU and address duplication, with LDS as a
// contrast path and a data-varying hot control. No MFMA, no conversion, no
// engine code. Scoped down on request: single CTA, hot/LDS only, no L2/
// stream/grid modes, no 12-byte loads (the 3-dword vector type is 16 bytes
// on this toolchain and caused both a silent mis-addressing bug and register
// allocator hazards; see the hazards noted inline below).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while(0)
using U2=uint32_t __attribute__((ext_vector_type(2)));
using U4=uint32_t __attribute__((ext_vector_type(4)));
constexpr int kUnroll=32;
constexpr uint32_t kLdsTag=0x40000000u;

// Identical deterministic payload on host and device; every timed dword is
// checked against this oracle on the host after each launch.
__host__ __device__ inline uint32_t mix32(uint32_t t) {
  uint32_t x=t+0x9e3779b9u;
  x^=x>>16; x*=0x21f0aaadu; x^=x>>15; x*=0x735a2d97u; x^=x>>15;
  return x;
}
// Mode 0=hot (one 64*W-byte fragment per wave, addresses frozen like the
// fixed-K probe), 3=LDS, 4=hot2 (two alternating fragments: data varies per
// step while staying L1-hot, a stale-read control).
__host__ __device__ constexpr int region_frags(int mode) { return mode==4?2:1; }
__host__ __device__ constexpr int frag_index(int mode,int k) { return mode==4?(k&1):0; }

template<int W> struct LoadReg;
template<> struct LoadReg<4> { using type=uint32_t; };
template<> struct LoadReg<8> { using type=U2; };
template<> struct LoadReg<16> { using type=U4; };
template<int W> using Reg=typename LoadReg<W>::type;

template<int W> __device__ __forceinline__ void issue_global(const uint32_t* p,Reg<W>& r) {
  if constexpr(W==4) asm volatile("global_load_dword %0, %1, off" : "=&v"(r) : "v"(p) : "memory");
  else if constexpr(W==8) asm volatile("global_load_dwordx2 %0, %1, off" : "=&v"(r) : "v"(p) : "memory");
  else asm volatile("global_load_dwordx4 %0, %1, off" : "=&v"(r) : "v"(p) : "memory");
}
template<int W> __device__ __forceinline__ void issue_lds(uint32_t addr,Reg<W>& r) {
  if constexpr(W==4) asm volatile("ds_read_b32 %0, %1" : "=&v"(r) : "v"(addr) : "memory");
  else if constexpr(W==8) asm volatile("ds_read_b64 %0, %1" : "=&v"(r) : "v"(addr) : "memory");
  else asm volatile("ds_read_b128 %0, %1" : "=&v"(r) : "v"(addr) : "memory");
}
// The wait and the FIRST read of each ring register stay in the same asm
// block (fp8-w8a16 scheduling lesson: no compiler-inserted premature copies
// of asynchronously written registers). Every element of the slot is passed
// as an asm input even though only element 0 is XORed: with dead elements
// the register allocator reuses slot registers (including as load ADDRESS
// registers), and a pending load's data return then clobbers them
// asynchronously (observed as corruption and HSA aperture violations).
template<int B,int W> __device__ __forceinline__ void take0_global(uint32_t& acc,const Reg<W>& r) {
  if constexpr(W==4)
    asm volatile("s_waitcnt vmcnt(%1)\n\tv_xor_b32 %0, %0, %2" : "+v"(acc) : "n"(B-1),"v"(r) : "memory");
  else if constexpr(W==8)
    asm volatile("s_waitcnt vmcnt(%1)\n\tv_xor_b32 %0, %0, %2" : "+v"(acc) : "n"(B-1),"v"(r[0]),"v"(r[1]) : "memory");
  else
    asm volatile("s_waitcnt vmcnt(%1)\n\tv_xor_b32 %0, %0, %2" : "+v"(acc) : "n"(B-1),"v"(r[0]),"v"(r[1]),"v"(r[2]),"v"(r[3]) : "memory");
}
template<int B,int W> __device__ __forceinline__ void take0_lds(uint32_t& acc,const Reg<W>& r) {
  if constexpr(W==4)
    asm volatile("s_waitcnt lgkmcnt(%1)\n\tv_xor_b32 %0, %0, %2" : "+v"(acc) : "n"(B-1),"v"(r) : "memory");
  else if constexpr(W==8)
    asm volatile("s_waitcnt lgkmcnt(%1)\n\tv_xor_b32 %0, %0, %2" : "+v"(acc) : "n"(B-1),"v"(r[0]),"v"(r[1]) : "memory");
  else
    asm volatile("s_waitcnt lgkmcnt(%1)\n\tv_xor_b32 %0, %0, %2" : "+v"(acc) : "n"(B-1),"v"(r[0]),"v"(r[1]),"v"(r[2]),"v"(r[3]) : "memory");
}
// Full-width consumption variant: only used by the separate step-64 verify
// launches, never by the timed bodies.
template<int B,int W> __device__ __forceinline__ void take_full_global(uint32_t (&acc)[4],const Reg<W>& r) {
  if constexpr(W==4)
    asm volatile("s_waitcnt vmcnt(%1)\n\tv_xor_b32 %0, %0, %2" : "+v"(acc[0]) : "n"(B-1),"v"(r) : "memory");
  else if constexpr(W==8)
    asm volatile("s_waitcnt vmcnt(%2)\n\tv_xor_b32 %0, %0, %3\n\tv_xor_b32 %1, %1, %4"
                 : "+v"(acc[0]),"+v"(acc[1]) : "n"(B-1),"v"(r[0]),"v"(r[1]) : "memory");
  else
    asm volatile("s_waitcnt vmcnt(%4)\n\tv_xor_b32 %0, %0, %5\n\tv_xor_b32 %1, %1, %6\n\tv_xor_b32 %2, %2, %7\n\tv_xor_b32 %3, %3, %8"
                 : "+v"(acc[0]),"+v"(acc[1]),"+v"(acc[2]),"+v"(acc[3]) : "n"(B-1),"v"(r[0]),"v"(r[1]),"v"(r[2]),"v"(r[3]) : "memory");
}
template<int B,int W> __device__ __forceinline__ void take_full_lds(uint32_t (&acc)[4],const Reg<W>& r) {
  if constexpr(W==4)
    asm volatile("s_waitcnt lgkmcnt(%1)\n\tv_xor_b32 %0, %0, %2" : "+v"(acc[0]) : "n"(B-1),"v"(r) : "memory");
  else if constexpr(W==8)
    asm volatile("s_waitcnt lgkmcnt(%2)\n\tv_xor_b32 %0, %0, %3\n\tv_xor_b32 %1, %1, %4"
                 : "+v"(acc[0]),"+v"(acc[1]) : "n"(B-1),"v"(r[0]),"v"(r[1]) : "memory");
  else
    asm volatile("s_waitcnt lgkmcnt(%4)\n\tv_xor_b32 %0, %0, %5\n\tv_xor_b32 %1, %1, %6\n\tv_xor_b32 %2, %2, %7\n\tv_xor_b32 %3, %3, %8"
                 : "+v"(acc[0]),"+v"(acc[1]),"+v"(acc[2]),"+v"(acc[3]) : "n"(B-1),"v"(r[0]),"v"(r[1]),"v"(r[2]),"v"(r[3]) : "memory");
}
__device__ __forceinline__ void drain_global() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
__device__ __forceinline__ void drain_lds() { asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory"); }
__device__ __forceinline__ uint32_t blocking_dword(const void* p) {
  uint32_t t;
  asm volatile("global_load_dword %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=&v"(t) : "v"(p) : "memory");
  return t;
}

// One load per wave per iteration into a Batch-deep ring; the wait retires
// exactly the oldest ring entry before its single checksum XOR.
template<int Mode,int Width,int Batch,int Dup,int Full>
__global__ __launch_bounds__(512) void load_kernel(
    const uint32_t* __restrict__ buf,uint32_t* __restrict__ output,
    uint64_t* __restrict__ stamps,int steps,int warm) {
  static_assert(Batch>0 && kUnroll%Batch==0);
  static_assert(Dup>=1 && Dup<=64 && 64%Dup==0);
  constexpr int W=Width,B=Batch,RF=region_frags(Mode),W4=W/4;
  using R=Reg<W>;
  const int tid=threadIdx.x;
  const int lane=threadIdx.x&63;
  const int region=threadIdx.x>>6;
  const int goff=lane/Dup;
  __shared__ alignas(16) uint32_t smem[256];
  uint32_t acc[4]={};
  R ring[B];
  // All addressing in dword units from the raw buffer (never do pointer
  // arithmetic on 3-dword vector types here: they are 16 bytes on this
  // toolchain).
  const uint32_t* p0=buf+(size_t(region)*RF*64+size_t(goff))*W4;
  const uint32_t lds_addr=uint32_t(goff)*uint32_t(W);
  if constexpr(Mode==3) {
    for(int i=threadIdx.x;i<256;i+=blockDim.x) smem[i]=mix32(kLdsTag+uint32_t(i));
    __syncthreads();
  } else if(warm) {
    // In-dispatch cache pre-read of the active fragments, outside the
    // core-clock interval (a separate dispatch did not establish a hot
    // cache in the fp8-w8a16 rounds).
    uint32_t sum=0;
#pragma clang loop unroll(disable)
    for(int f=0;f<RF;++f) sum^=blocking_dword(p0+size_t(f)*64*W4);
    asm volatile("" : : "v"(sum) : "memory");
  }
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for(int k=0;k<B-1;++k) {
    if constexpr(Mode==3) issue_lds<W>(lds_addr,ring[k%B]);
    else issue_global<W>(p0+size_t(frag_index(Mode,k))*64*W4,ring[k%B]);
  }
  __builtin_amdgcn_sched_barrier(0);
  const uint64_t begin=clock64(),wall_begin=wall_clock64();
#pragma clang loop unroll(disable)
  for(int i=0;i<steps;i+=kUnroll) {
#pragma unroll
    for(int j=0;j<kUnroll;++j) {
      // i is always a multiple of kUnroll and Batch divides kUnroll, so the
      // ring slots are (B-1+j)%B and j%B: compile-time constants. A runtime
      // k%B index would force the whole ring array into scratch memory and
      // the compiler would then store PENDING ring registers to it.
      const int k=B-1+i+j,slot=(B-1+j)%B,cslot=j%B;
      if constexpr(Mode==3) issue_lds<W>(lds_addr,ring[slot]);
      else issue_global<W>(p0+size_t(frag_index(Mode,k))*64*W4,ring[slot]);
      __builtin_amdgcn_sched_barrier(0);
      if constexpr(Mode==3) { if constexpr(Full) take_full_lds<B,W>(acc,ring[cslot]); else take0_lds<B,W>(acc[0],ring[cslot]); }
      else { if constexpr(Full) take_full_global<B,W>(acc,ring[cslot]); else take0_global<B,W>(acc[0],ring[cslot]); }
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  if constexpr(Mode==3) drain_lds(); else drain_global();
  // Consume the remaining Batch-1 in-flight results. steps is a multiple of
  // kUnroll (host-validated) and Batch divides kUnroll, so their ring slots
  // are exactly 0..B-2, again compile-time constants. The wait and first
  // read stay in one asm block (take with immediate 0); amortized over the
  // body.
#pragma unroll
  for(int i=0;i<B-1;++i) {
    if constexpr(Mode==3) { if constexpr(Full) take_full_lds<1,W>(acc,ring[i]); else take0_lds<1,W>(acc[0],ring[i]); }
    else { if constexpr(Full) take_full_global<1,W>(acc,ring[i]); else take0_global<1,W>(acc[0],ring[i]); }
  }
  __builtin_amdgcn_sched_barrier(0);
  const uint64_t end=clock64(),wall_end=wall_clock64();
  const int gw=tid>>6;
  if(lane==0) { stamps[4*gw]=begin; stamps[4*gw+1]=end; stamps[4*gw+2]=wall_begin; stamps[4*gw+3]=wall_end; }
  for(int d=0;d<4;++d) output[size_t(tid)*4+d]=acc[d];
}

__global__ void fill_values(uint32_t* p,size_t n) {
  size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t<n) p[t]=mix32(uint32_t(t));
}
using V4=uint32_t __attribute__((ext_vector_type(4)));
__host__ __device__ inline V4 eviction_value(size_t t) {
  uint32_t x=uint32_t(t)*1664525u+1013904223u;
  return V4{x,x^0x9e3779b9u,(x<<13)|(x>>19),x+uint32_t(t>>8)};
}
__global__ void fill_eviction(V4* p,size_t n) {
  size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t<n) p[t]=eviction_value(t);
}
// Completed before every cache=1 measured launch; one explicit dwordx4 load
// per grid-stride iteration, all four words consumed.
__global__ void evict_cache(const V4* p,size_t n,uint32_t* sink) {
  size_t tid=size_t(blockIdx.x)*blockDim.x+threadIdx.x,stride=size_t(gridDim.x)*blockDim.x;
  uint32_t sum=0;
#pragma clang loop unroll(disable)
  for(size_t t=tid;t<n;t+=stride) {
    V4 x;
    asm volatile("global_load_dwordx4 %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=&v"(x) : "v"(p+t) : "memory");
    sum^=x[0]^x[1]^x[2]^x[3];
  }
  sink[tid]=sum;
}

template<class T> struct Device {
  T* p=nullptr;
  explicit Device(size_t n) { HIP(hipMalloc(&p,n*sizeof(T))); }
  ~Device() { if(p) (void)hipFree(p); }
  Device(const Device&)=delete;
  std::vector<T> get(size_t n) const { std::vector<T> v(n); HIP(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost)); return v; }
};

using Launch=void(*)(const uint32_t*,uint32_t*,uint64_t*,int,int,int,int);
struct Case { int mode,width,batch,dup,full,waves; Launch launch; };
template<int M,int W,int B,int D,int F>
void launch_case(const uint32_t* buf,uint32_t* out,uint64_t* st,int steps,int warm,int ctas,int waves) {
  (void)ctas;
  load_kernel<M,W,B,D,F><<<1,64*waves>>>(buf,out,st,steps,warm);
}
std::vector<int> caches_for(const Case& c) { return c.mode==3?std::vector<int>{0}:std::vector<int>{0,1}; }
std::vector<Case> make_cases() {
  std::vector<Case> v;
#define ADD(M,W,B,D,F,WV) v.push_back({M,W,B,D,F,WV,launch_case<M,W,B,D,F>})
#define WAVES(M,W) ADD(M,W,8,1,0,1); ADD(M,W,8,1,0,2); ADD(M,W,8,1,0,4); ADD(M,W,8,1,0,8)
  WAVES(0,4); WAVES(0,8); WAVES(0,16);
  ADD(0,4,1,1,0,4); ADD(0,4,2,1,0,4); ADD(0,4,4,1,0,4); ADD(0,4,16,1,0,4);
#define DUPS(W) ADD(0,W,8,2,0,4); ADD(0,W,8,4,0,4); ADD(0,W,8,8,0,4); ADD(0,W,8,16,0,4); ADD(0,W,8,32,0,4); ADD(0,W,8,64,0,4)
  DUPS(4); DUPS(16);
  ADD(4,4,8,1,0,4); ADD(4,16,8,1,0,4);
  ADD(3,4,8,1,0,4); ADD(3,8,8,1,0,4); ADD(3,16,8,1,0,4); ADD(3,16,8,1,0,1);
#undef DUPS
#undef WAVES
#undef ADD
  return v;
}
std::vector<Case> make_verify() {
  std::vector<Case> v;
#define ADD(M,W,B,D,WV) v.push_back({M,W,B,D,1,WV,launch_case<M,W,B,D,1>})
#define WAVES(W) ADD(0,W,8,1,1); ADD(0,W,8,1,2); ADD(0,W,8,1,4); ADD(0,W,8,1,8)
  WAVES(4); WAVES(8); WAVES(16);
  ADD(0,4,1,1,4); ADD(0,4,2,1,4); ADD(0,4,4,1,4); ADD(0,4,16,1,4);
#define DUPS(W) ADD(0,W,8,2,4); ADD(0,W,8,4,4); ADD(0,W,8,8,4); ADD(0,W,8,16,4); ADD(0,W,8,32,4); ADD(0,W,8,64,4)
  DUPS(4); DUPS(16);
  ADD(4,4,8,1,4); ADD(4,8,8,1,4); ADD(4,16,8,1,4);
  ADD(3,4,8,1,4); ADD(3,8,8,1,4); ADD(3,16,8,1,4); ADD(3,16,8,1,1);
#undef DUPS
#undef WAVES
#undef ADD
  return v;
}
struct Expect { uint32_t v[4]{}; };
// Closed form: every load repeats fragment 0 (hot/LDS) or alternates two
// fragments (hot2), so the per-dword XOR checksum is trivial in the count.
Expect expected(const Case& c,int region,int g,int steps) {
  const int W4=c.width/4,B=c.batch,n=steps+B-1;
  const int RF=region_frags(c.mode);
  Expect e;
  auto val=[&](int k,int d)->uint32_t {
    if(c.mode==3) return mix32(kLdsTag+uint32_t(g)*uint32_t(W4)+uint32_t(d));
    size_t reg=size_t(region)*RF*64+size_t(frag_index(c.mode,k))*64+size_t(g);
    return mix32(uint32_t(reg*W4+d));
  };
  if(c.mode!=4) { for(int d=0;d<W4;++d) e.v[d]=(n&1)?val(0,d):0; }
  else {
    const int n0=(n+1)/2,n1=n/2;
    for(int d=0;d<W4;++d) e.v[d]=((n0&1)?val(0,d):0)^((n1&1)?val(1,d):0);
  }
  return e;
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
  if(steps<=0 || steps>8192 || steps%kUnroll || rounds<=0 || evict_mib<32 || evict_mib>1024)
    throw std::runtime_error("steps: multiple of 32, 1..8192; rounds positive; eviction MiB 32..1024");
  HIP(hipSetDevice(0)); hipDeviceProp_t prop{}; HIP(hipGetDeviceProperties(&prop,0));
  int wall_rate=0; HIP(hipDeviceGetAttribute(&wall_rate,hipDeviceAttributeWallClockRate,0));
  int l2_bytes=0; HIP(hipDeviceGetAttribute(&l2_bytes,hipDeviceAttributeL2CacheSize,0));
  int evict_blocks=prop.multiProcessorCount*8,evict_threads=evict_blocks*256;
  size_t evict_records=(size_t(evict_mib)<<20)/sizeof(V4);
  auto cases=make_cases();
  auto verify_cases=make_verify();
  constexpr size_t kMaxWaves=8,kValueDwords=4096;   // 8 waves x 2 frags x 64 lanes x 16 B
  std::fprintf(stderr,"device=%s arch=%s CUs=%d timed_grid=1x1 block=64*waves steps=%d rounds=%d seed=%u unroll=%d reported_L2_bytes=%d eviction_bytes=%zu eviction_grid=%dx256; native O3, CPU+memory NUMA0 externally, no model/context, no MFMA/conversion\n",
               prop.name,prop.gcnArchName,prop.multiProcessorCount,steps,rounds,seed,kUnroll,l2_bytes,evict_records*sizeof(V4),evict_blocks);
  Device<uint32_t> buf(kValueDwords);
  fill_values<<<int((kValueDwords+255)/256),256>>>(buf.p,kValueDwords); HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
  {
    auto spot=buf.get(4);
    for(uint32_t t=0;t<4;++t) if(spot[t]!=mix32(t)) throw std::runtime_error("value buffer fill spot check failed");
  }
  std::fprintf(stderr,"PASS value buffer fill spot checks (mix32 oracle)\n");
  Device<V4> trash(evict_records); Device<uint32_t> trash_sink(evict_threads);
  fill_eviction<<<int((evict_records+255)/256),256>>>(trash.p,evict_records); HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
  auto evict=[&]() { evict_cache<<<evict_blocks,256>>>(trash.p,evict_records,trash_sink.p); HIP(hipGetLastError()); HIP(hipDeviceSynchronize()); };
  evict();
  {
    auto checks=trash_sink.get(evict_threads);
    for(int tid: {0,evict_threads/2,evict_threads-1}) {
      uint32_t expected_sum=0;
      for(size_t t=tid;t<evict_records;t+=evict_threads) { V4 v=eviction_value(t); expected_sum^=v[0]^v[1]^v[2]^v[3]; }
      if(checks[tid]!=expected_sum) throw std::runtime_error("eviction reader checksum mismatch");
    }
  }
  std::fprintf(stderr,"PASS eviction reader checksum controls (all four words consumed)\n");
  const size_t out_total=kMaxWaves*64*4+32;
  Device<uint32_t> out(out_total); Device<uint64_t> stamps(kMaxWaves*4);
  auto run=[&](const Case& c,int csteps,int warm) {
    HIP(hipMemset(out.p,0xff,out_total*sizeof(uint32_t)));
    c.launch(buf.p,out.p+16,stamps.p,csteps,warm,1,c.waves); HIP(hipGetLastError());
  };
  // Returns -1 for a guard failure, otherwise the count of mismatching
  // dwords across all threads (0 expected for every launch).
  auto verify=[&](const Case& c,int csteps,bool full)->int {
    auto y=out.get(out_total);
    const size_t threads=size_t(64*c.waves);
    for(int g=0;g<16;++g)
      if(y[size_t(g)]!=0xffffffffu || y[16+threads*4+size_t(g)]!=0xffffffffu) return -1;
    int bad=0;
    for(size_t tid=0;tid<threads;++tid) {
      const int region=int(tid>>6),g2=int(tid&63)/c.dup;
      Expect e=expected(c,region,g2,csteps);
      for(int d=0;d<4;++d) {
        const uint32_t want=full?(d<c.width/4?e.v[d]:0u):(d==0?e.v[0]:0u);
        if(y[16+tid*4+size_t(d)]!=want) ++bad;
      }
    }
    return bad;
  };
  // Full-width correctness gate at steps=64: every dword of every width,
  // every mode, before any timing.
  for(const auto& c:verify_cases) {
    run(c,64,1); HIP(hipDeviceSynchronize());
    const int bad=verify(c,64,true);
    if(bad!=0) { std::fprintf(stderr,"FAIL verify mode=%d width=%d batch=%d dup=%d waves=%d bad=%d\n",c.mode,c.width,c.batch,c.dup,c.waves,bad); throw std::runtime_error("verify-case checksum mismatch"); }
  }
  std::fprintf(stderr,"PASS %zu full-width step-64 verify cases (exact checksums, guards)\n",verify_cases.size());
  auto find_case=[&](int mode,int width,int waves)->int {
    for(size_t i=0;i<cases.size();++i) {
      const Case& c=cases[i];
      if(c.mode==mode&&c.width==width&&c.batch==8&&c.dup==1&&c.waves==waves) return int(i);
    }
    throw std::runtime_error("warm-up case not found");
  };
  std::vector<int> warm_cases={find_case(0,16,4),find_case(0,16,8),find_case(3,16,4)};
  auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(std::chrono::steady_clock::now()<until) {
    for(int i:warm_cases) run(cases[size_t(i)],steps,1);
    HIP(hipDeviceSynchronize());
  }
  Events events; std::mt19937 rng(seed);
  std::vector<std::pair<int,int>> order;
  for(size_t i=0;i<cases.size();++i) for(int cache:caches_for(cases[i])) order.emplace_back(int(i),cache);
  const char* mode_names[]={"hot","","","lds","hot2"};
  std::puts("round,case,mode,mode_name,width,batch,dup,waves,cache,steps,eviction_MiB,launch_ms,body_us_mean,cycles_per_load,per_cu_median_cycles,min_wave_cycles,max_wave_cycles,core_MHz,mismatch_count,valid");
  int invalid=0;
  for(int round=0;round<rounds;++round) {
    std::shuffle(order.begin(),order.end(),rng);
    for(auto [index,cache]:order) {
      const Case& c=cases[size_t(index)];
      if(cache) evict();
      HIP(hipEventRecord(events.begin));
      run(c,steps,cache==0);
      HIP(hipEventRecord(events.end)); HIP(hipEventSynchronize(events.end));
      float ms; HIP(hipEventElapsedTime(&ms,events.begin,events.end));
      const int bad=verify(c,steps,false);
      if(bad!=0) { ++invalid; std::fprintf(stderr,"INVALID round=%d case=%d mode=%d width=%d batch=%d dup=%d waves=%d cache=%d bad=%d\n",round,index,c.mode,c.width,c.batch,c.dup,c.waves,cache,bad); }
      auto st=stamps.get(kMaxWaves*4);
      const int nwav=c.waves;
      double cyc_sum=0,tick_sum=0,mn=1e300,mx=0;
      std::vector<double> intervals;
      for(int w=0;w<nwav;++w) {
        const double n=double(st[4*w+1]-st[4*w]),t=double(st[4*w+3]-st[4*w+2]);
        cyc_sum+=n; tick_sum+=t;
        const double per=n/steps;
        intervals.push_back(per);
        mn=std::min(mn,per); mx=std::max(mx,per);
      }
      std::sort(intervals.begin(),intervals.end());
      const double per_cu=(nwav&1) ? intervals[nwav/2] : (intervals[nwav/2-1]+intervals[nwav/2])/2;
      std::printf("%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3f,%d,%d\n",
                  round,index,c.mode,mode_names[c.mode],c.width,c.batch,c.dup,c.waves,cache,steps,evict_mib,
                  ms,tick_sum/nwav*1000.0/wall_rate,cyc_sum/(double(nwav)*steps),per_cu,mn,mx,
                  cyc_sum/tick_sum*double(wall_rate)/1000.0,bad,bad==0);
      std::fflush(stdout);
    }
  }
  std::fprintf(stderr,"COMPLETE %zu cases x cache protocols x %d rounds, one timed CTA/pass; exact checksums every launch, invalid_samples=%d\n",cases.size(),rounds,invalid);
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
