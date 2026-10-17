// Preheader isolation and explicit boundary conversion redistribution.
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
#include <vector>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while(0)
using H4 = _Float16 __attribute__((ext_vector_type(4)));
using U2 = uint32_t __attribute__((ext_vector_type(2)));
using F16 = float __attribute__((ext_vector_type(16)));

// Same eight-VALU conversion as the accepted probe. Earlyclobber and the
// compiler-visible previous operand prevent clobbering the ready future bank.
__device__ __forceinline__ H4 convert(uint32_t w, H4 previous) {
  uint32_t lo,hi;
  asm volatile(
      "v_perm_b32 %0, 0, %2, %3\n\t"
      "v_perm_b32 %1, 0, %2, %4\n\t"
      "v_pk_ashrrev_i16 %0, 1, %0 op_sel_hi:[0,1]\n\t"
      "v_pk_ashrrev_i16 %1, 1, %1 op_sel_hi:[0,1]\n\t"
      "v_and_b32 %0, %5, %0\n\t"
      "v_and_b32 %1, %5, %1"
      : "=&v"(lo), "=&v"(hi)
      : "v"(w), "s"(0x01040004u), "s"(0x03040204u), "s"(0xbfffbfffu), "v"(previous));
  asm volatile("v_pk_mul_f16 %0, %0, %2\n\tv_pk_mul_f16 %1, %1, %2"
               : "+v"(lo), "+v"(hi) : "s"(0x5c005c00u));
  return __builtin_bit_cast(H4,(U2{lo,hi}));
}

struct Parts { uint32_t lo,hi; };
// Begin the last conversion early. Never overwrite the final MFMA source.
// Split 1/2/3/4 moves perm-lo / both perms / plus shift-lo / both shifts.
template<int Split> __device__ __forceinline__ Parts begin_conversion(uint32_t w,H4 keep) {
  uint32_t lo,hi=0;
  if constexpr(Split==1) {
    asm volatile("v_perm_b32 %0, 0, %1, %2" : "=&v"(lo) : "v"(w), "s"(0x01040004u), "v"(keep));
  } else {
    asm volatile("v_perm_b32 %0, 0, %2, %3\n\tv_perm_b32 %1, 0, %2, %4"
                 : "=&v"(lo), "=&v"(hi) : "v"(w), "s"(0x01040004u), "s"(0x03040204u), "v"(keep));
  }
  if constexpr(Split>=3) asm volatile("v_pk_ashrrev_i16 %0, 1, %0 op_sel_hi:[0,1]" : "+v"(lo));
  if constexpr(Split>=4) asm volatile("v_pk_ashrrev_i16 %0, 1, %0 op_sel_hi:[0,1]" : "+v"(hi));
  return {lo,hi};
}
template<int Split> __device__ __forceinline__ H4 finish_conversion(Parts p,uint32_t w) {
  uint32_t lo=p.lo,hi=p.hi;
  if constexpr(Split==1) asm volatile("v_perm_b32 %0, 0, %1, %2" : "=&v"(hi) : "v"(w), "s"(0x03040204u));
  if constexpr(Split<3) asm volatile("v_pk_ashrrev_i16 %0, 1, %0 op_sel_hi:[0,1]" : "+v"(lo));
  if constexpr(Split<4) asm volatile("v_pk_ashrrev_i16 %0, 1, %0 op_sel_hi:[0,1]" : "+v"(hi));
  asm volatile("v_and_b32 %0, %2, %0\n\tv_and_b32 %1, %2, %1\n\t"
               "v_pk_mul_f16 %0, %0, %3\n\tv_pk_mul_f16 %1, %1, %3"
               : "+v"(lo), "+v"(hi) : "s"(0xbfffbfffu), "s"(0x5c005c00u));
  return __builtin_bit_cast(H4,(U2{lo,hi}));
}

// +a forces accumulator initialization before the preheader pad. The explicit
// wait remains outside the loop; no guessed NOP delay substitutes for a load wait.
template<int Prelude> __device__ __forceinline__ void preheader(F16& c,H4 a,H4 b0,H4 b1,uint32_t w,uint32_t scratch) {
#define PAD(TEXT) asm volatile(TEXT : "+a"(c), "+v"(scratch) : "v"(a), "v"(b0), "v"(b1), "v"(w))
#define X "v_xor_b32 %1, 1, %1\n\t"
  if constexpr(Prelude==1) { PAD("s_waitcnt vmcnt(0)"); }
  if constexpr(Prelude==2) { PAD("s_waitcnt vmcnt(0)\n\ts_nop 0"); }
  if constexpr(Prelude==3) { PAD("s_waitcnt vmcnt(0)\n\ts_nop 3"); }
  if constexpr(Prelude==4) { PAD("s_waitcnt vmcnt(0)\n\ts_nop 7"); }
  if constexpr(Prelude==5) { PAD("s_waitcnt vmcnt(0)\n\t" X X); }
  if constexpr(Prelude==6) { PAD("s_waitcnt vmcnt(0)\n\t" X X X X); }
  if constexpr(Prelude==7) { PAD("s_waitcnt vmcnt(0)\n\t" X X X X X X X X); }
#undef X
#undef PAD
  if constexpr(Prelude!=0) __builtin_amdgcn_sched_barrier(0);
}

// Schedule 0 raw FP16. 1..5 two-bank after-order, moving 0..4 operations.
// 6..10 four-bank lagged-target after-order, moving 0..4 operations. Lagged
// target is already dead at the penultimate MFMA, so no temporary bank copies
// are intrinsically required. Each split is compared with its own 8/8 control.
template<int Schedule,int Prelude,int Unroll>
__global__ __launch_bounds__(256) void split_kernel(
    const H4* acts,const uint32_t* packed,const U2* expanded,
    float* output,uint64_t* stamps,int steps) {
  constexpr int Banks=Schedule>=6?4:2;
  constexpr int Split=Schedule==0?0:Schedule>=6?Schedule-6:Schedule-1;
  int tid=blockIdx.x*blockDim.x+threadIdx.x, lane=threadIdx.x%64;
  H4 a=acts[lane],b[Banks]; uint32_t w=packed[lane];
  b[0]=__builtin_bit_cast(H4,expanded[lane]);
  if constexpr(Schedule!=0) {
    b[0]=convert(w,b[0]);
#pragma unroll
    for(int r=1;r<Banks;++r) b[r]=convert(w,b[r-1]);
    __builtin_amdgcn_sched_barrier(0);
  } else b[1]=b[0];
  F16 c={};
  preheader<Prelude>(c,a,b[0],b[1],w,uint32_t(lane));
  uint64_t begin=clock64(), wall_begin=wall_clock64();
#define M(J) c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,b[Schedule==0?0:(J)%Banks],c,0,0,0)
#define TARGET(J) ((Schedule>=6?((J)+Banks-1):(J))%Banks)
#define STEP(J) do { M(J); if constexpr(Schedule!=0) { \
  __builtin_amdgcn_sched_barrier(0); \
  b[TARGET(J)]=convert(w,b[(TARGET(J)+1)%Banks]); \
  __builtin_amdgcn_sched_barrier(0); } } while(0)
#pragma clang loop unroll(disable)
  for(int i=0;i<steps;i+=Unroll) {
    // Explicit body, not an automatically unrolled uniform inner loop.
    STEP(0); STEP(1);
    if constexpr(Unroll>=8) { STEP(2); STEP(3); STEP(4); STEP(5); }
    if constexpr(Unroll==32) {
      STEP(6); STEP(7); STEP(8); STEP(9); STEP(10); STEP(11);
      STEP(12); STEP(13); STEP(14); STEP(15); STEP(16); STEP(17);
      STEP(18); STEP(19); STEP(20); STEP(21); STEP(22); STEP(23);
      STEP(24); STEP(25); STEP(26); STEP(27); STEP(28); STEP(29);
    }
    STEP(Unroll-2);
    if constexpr(Split==0) { STEP(Unroll-1); }
    else {
      Parts p=begin_conversion<Split>(w,b[(Unroll-1)%Banks]);
      __builtin_amdgcn_sched_barrier(0);
      M(Unroll-1);
      __builtin_amdgcn_sched_barrier(0);
      b[TARGET(Unroll-1)]=finish_conversion<Split>(p,w);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
#undef STEP
#undef TARGET
#undef M
  uint64_t end=clock64(), wall_end=wall_clock64();
  if(lane==0) {
    int wave=tid/64;
    stamps[4*wave]=begin; stamps[4*wave+1]=end;
    stamps[4*wave+2]=wall_begin; stamps[4*wave+3]=wall_end;
  }
  for(int r=0;r<16;++r) output[tid*16+r]=c[r];
}

template<class T> struct Device {
  T* p=nullptr;
  explicit Device(size_t n) { HIP(hipMalloc(&p,n*sizeof(T))); }
  ~Device() { if(p) (void)hipFree(p); }
  Device(const Device&)=delete;
  void put(const std::vector<T>& v) { HIP(hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice)); }
  std::vector<T> get(size_t n) const { std::vector<T> v(n); HIP(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost)); return v; }
};
float decode(uint8_t x) {
  int e=(x>>3)&15,m=x&7;
  float v=e?std::ldexp(float(8+m),e-10):std::ldexp(float(m),-9);
  return x&128?-v:v;
}
using Launch=void(*)(const H4*,const uint32_t*,const U2*,float*,uint64_t*,int,int);
struct Case { int schedule,prelude,unroll; Launch launch; };
template<int S,int P,int U> void launch(
    const H4* a,const uint32_t* q,const U2* h,float* y,uint64_t* st,int steps,int blocks) {
  split_kernel<S,P,U><<<blocks,256>>>(a,q,h,y,st,steps);
  HIP(hipGetLastError());
}
std::vector<Case> cases() {
  std::vector<Case> v;
#define ADD(S,P,U) v.push_back({S,P,U,launch<S,P,U>})
#define US(S,P) ADD(S,P,4); ADD(S,P,8); ADD(S,P,32)
#define ALLP(S) US(S,0); US(S,1); US(S,2); US(S,3); US(S,4); US(S,5); US(S,6); US(S,7)
#define SPLITS(S) US(S,0); US(S,1); US(S,4); US(S,7)
  ALLP(0); ALLP(1); ALLP(6);
  SPLITS(2); SPLITS(3); SPLITS(4); SPLITS(5);
  SPLITS(7); SPLITS(8); SPLITS(9); SPLITS(10);
#undef SPLITS
#undef ALLP
#undef US
#undef ADD
  return v;
}
struct Events {
  hipEvent_t begin{},end{};
  Events() { HIP(hipEventCreate(&begin)); HIP(hipEventCreate(&end)); }
  ~Events() { (void)hipEventDestroy(begin); (void)hipEventDestroy(end); }
};

int main(int argc,char** argv) try {
  int steps=argc>1?std::stoi(argv[1]):4096;
  int rounds=argc>2?std::stoi(argv[2]):9;
  unsigned seed=argc>3?unsigned(std::stoul(argv[3])):1729;
  if(steps<=0 || steps%32 || rounds<=0) throw std::runtime_error("steps must be positive multiple of 32; rounds positive");
  HIP(hipSetDevice(0)); hipDeviceProp_t prop{}; HIP(hipGetDeviceProperties(&prop,0));
  int wall_rate=0; HIP(hipDeviceGetAttribute(&wall_rate,hipDeviceAttributeWallClockRate,0));
  int blocks=prop.multiProcessorCount,threads=blocks*256,waves=threads/64;
  std::fprintf(stderr,"device=%s arch=%s CUs=%d steps=%d rounds=%d seed=%u cases=%zu; no model/context; bind CPU+memory NUMA0 externally\n",prop.name,prop.gcnArchName,blocks,steps,rounds,seed,cases().size());
  std::vector<H4> a(64); std::vector<uint32_t> q(64); std::vector<U2> h(64);
  double av[32][8],bv[8][32],ref[32][32]={};
  for(int lane=0;lane<64;++lane) {
    H4 b; uint32_t word=0;
    for(int t=0;t<4;++t) {
      int k=4*(lane/32)+t,j=lane%32;
      a[lane][t]=_Float16(((j*3+k*5)%7-3)*0.125f); av[j][k]=float(a[lane][t]);
      int x=(lane*19+t*7)%57;
      uint8_t byte=uint8_t(x|((lane+t)%3==0?128:0));
      word|=uint32_t(byte)<<(8*t); b[t]=_Float16(decode(byte)); bv[k][j]=float(b[t]);
    }
    q[lane]=word; h[lane]=std::bit_cast<U2>(b);
  }
  for(int i=0;i<32;++i) for(int j=0;j<32;++j) for(int k=0;k<8;++k) ref[i][j]+=av[i][k]*bv[k][j]*steps;
  Device<H4> da(64); Device<uint32_t> dq(64); Device<U2> dh(64);
  Device<float> out(size_t(threads)*16); Device<uint64_t> stamps(waves*4);
  da.put(a); dq.put(q); dh.put(h);
  auto all=cases();
  auto run=[&](const Case& c) { c.launch(da.p,dq.p,dh.p,out.p,stamps.p,steps,blocks); };
  // Resolve every kernel; condition raw, 8/8 and 10/6 variants for >=2 s.
  for(auto& c:all) run(c);
  HIP(hipDeviceSynchronize());
  auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(std::chrono::steady_clock::now()<until) {
    for(auto& c:all) if(c.unroll==8 && c.prelude==4 && (c.schedule==0 || c.schedule==1 || c.schedule==3 || c.schedule==8)) run(c);
    HIP(hipDeviceSynchronize());
  }
  Events events;
  std::mt19937 rng(seed);
  std::printf("round,schedule,prelude,unroll,ms,cycles_per_mfma,cycles_per_loop,wave_p10,wave_p50,wave_p90,core_MHz,TFLOPs,max_abs_error,valid\n");
  int invalid_samples=0;
  std::vector<int> order(all.size()); for(size_t i=0;i<all.size();++i) order[i]=int(i);
  for(int round=0;round<rounds;++round) {
    std::shuffle(order.begin(),order.end(),rng);
    for(int index:order) {
      auto& c=all[index];
      HIP(hipEventRecord(events.begin));
      for(int repeat=0;repeat<8;++repeat) run(c);
      HIP(hipEventRecord(events.end)); HIP(hipEventSynchronize(events.end));
      float ms; HIP(hipEventElapsedTime(&ms,events.begin,events.end)); ms/=8;
      auto y=out.get(size_t(threads)*16); auto st=stamps.get(waves*4);
      double total=0,ticks=0,maxerr=0; std::vector<double> intervals;
      for(int wave=0;wave<waves;++wave) {
        double cycles=double(st[4*wave+1]-st[4*wave]);
        total+=cycles; ticks+=double(st[4*wave+3]-st[4*wave+2]);
        intervals.push_back(cycles/steps);
      }
      for(int tid=0;tid<threads;++tid) for(int r=0;r<16;++r) {
        int lane=tid%64,i=8*(r/4)+4*(lane/32)+r%4,j=lane%32;
        double error=std::isfinite(y[tid*16+r])?std::abs(y[tid*16+r]-ref[i][j]):INFINITY;
        maxerr=std::max(maxerr,error);
      }
      if(maxerr!=0) {
        ++invalid_samples;
        std::fprintf(stderr,"INVALID round=%d schedule=%d prelude=%d U=%d max_error=%.9g\n",round,c.schedule,c.prelude,c.unroll,maxerr);
      }
      std::sort(intervals.begin(),intervals.end());
      double cycles=total/(waves*double(steps));
      std::printf("%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3f,%.6f,%.9g,%d\n",round,c.schedule,c.prelude,c.unroll,ms,cycles,cycles*c.unroll,intervals[waves/10],intervals[waves/2],intervals[waves*9/10],total/ticks*wall_rate/1000,double(waves)*steps*16384/(ms*1e9),maxerr,maxerr==0);
    }
    std::fflush(stdout);
  }
  std::fprintf(stderr,"COMPLETE %zu cases x %d rounds: exact all-output FP64 checks, invalid_samples=%d\n",all.size(),rounds,invalid_samples);
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
