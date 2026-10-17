// Operand-ring depth and staged VALU/source-dependency controls. Originals unchanged.
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

// Kind: 0 natural raw MFMA, 1 barriers only, 2 v_nop, 3 scratch XOR,
// 4 read B, 5 identity-write B, 6 read A, 7 identity-write A,
// 8 paired real sign flips of A/B (product unchanged), 9 FP8 conversion,
// 10 paired flips with conservative s_nop 3 separation (hazard control).
// Identity writes test dependency scheduling, not safe timing for changed data.
template<int Kind,int Ops>
__device__ __forceinline__ void work(H4& a,H4& b,uint32_t& scratch) {
  U2 av=__builtin_bit_cast(U2,a),bv=__builtin_bit_cast(U2,b);
#pragma unroll
  for(int k=0;k<Ops;++k) {
    if constexpr(Kind==2) asm volatile("v_nop");
    if constexpr(Kind==3) asm volatile("v_xor_b32 %0, 1, %0" : "+v"(scratch));
    if constexpr(Kind==4) asm volatile("v_xor_b32 %0, 0, %1" : "=&v"(scratch) : "v"(bv[k%2]));
    if constexpr(Kind==5) asm volatile("v_or_b32 %0, 0, %0" : "+v"(bv[k%2]));
    if constexpr(Kind==6) asm volatile("v_xor_b32 %0, 0, %1" : "=&v"(scratch) : "v"(av[k%2]));
    if constexpr(Kind==7) asm volatile("v_or_b32 %0, 0, %0" : "+v"(av[k%2]));
    if constexpr(Kind==8 || Kind==10) {
      // Consecutive A/B XORs flip matching K positions, preserving each
      // product while genuinely changing source bits between MFMAs.
      if(k%2==0) asm volatile("v_xor_b32 %0, %1, %0" : "+v"(av[(k/2)%2]) : "s"(0x80008000u));
      else asm volatile("v_xor_b32 %0, %1, %0" : "+v"(bv[(k/2)%2]) : "s"(0x80008000u));
    }
  }
  if constexpr(Kind==5 || Kind==8 || Kind==10) b=__builtin_bit_cast(H4,bv);
  if constexpr(Kind==7 || Kind==8 || Kind==10) a=__builtin_bit_cast(H4,av);
  if constexpr(Kind==10) asm volatile("s_nop 3" : "+v"(a), "+v"(b));
}

template<int Kind,int Order,int Banks,int Ops,int Unroll>
__global__ __launch_bounds__(256) void dependency_kernel(
    const H4* acts,const uint32_t* packed,const U2* expanded,
    float* output,uint64_t* stamps,int steps) {
  static_assert(Unroll%Banks==0);
  int tid=blockIdx.x*blockDim.x+threadIdx.x, lane=threadIdx.x%64;
  H4 a=acts[lane], b[Banks];
  uint32_t w=packed[lane],scratch=uint32_t(lane);
  if constexpr(Kind==9) {
#pragma unroll
    for(int r=0;r<Banks;++r) b[r]=convert(w,r?b[r-1]:a);
    __builtin_amdgcn_sched_barrier(0);
  } else b[0]=__builtin_bit_cast(H4,expanded[lane]);
  F16 c={};
  uint64_t begin=clock64(), wall_begin=wall_clock64();
#pragma clang loop unroll(disable)
  for(int i=0;i<steps;i+=Unroll) {
#pragma unroll
    for(int j=0;j<Unroll;++j) {
      if constexpr(Kind==9) {
        if constexpr(Order==0) {
          b[j%Banks]=convert(w,b[(j+1)%Banks]);
          __builtin_amdgcn_sched_barrier(0);
          c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,b[(j+1)%Banks],c,0,0,0);
          __builtin_amdgcn_sched_barrier(0);
        } else {
          c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,b[j%Banks],c,0,0,0);
          __builtin_amdgcn_sched_barrier(0);
          b[j%Banks]=convert(w,b[(j+1)%Banks]);
          __builtin_amdgcn_sched_barrier(0);
        }
      } else if constexpr(Kind==0) {
        c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,b[0],c,0,0,0);
      } else {
        if constexpr(Order==0) {
          work<Kind,Ops>(a,b[0],scratch);
          __builtin_amdgcn_sched_barrier(0);
          c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,b[0],c,0,0,0);
          __builtin_amdgcn_sched_barrier(0);
        } else {
          c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,b[0],c,0,0,0);
          __builtin_amdgcn_sched_barrier(0);
          work<Kind,Ops>(a,b[0],scratch);
          __builtin_amdgcn_sched_barrier(0);
        }
      }
    }
  }
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
struct Case { int kind,order,banks,ops,unroll; Launch launch; };
template<int K,int O,int B,int N,int U> void launch(
    const H4* a,const uint32_t* q,const U2* h,float* y,uint64_t* st,int steps,int blocks) {
  dependency_kernel<K,O,B,N,U><<<blocks,256>>>(a,q,h,y,st,steps);
  HIP(hipGetLastError());
}
std::vector<Case> cases() {
  std::vector<Case> v;
#define ADD(K,O,B,N,U) v.push_back({K,O,B,N,U,launch<K,O,B,N,U>})
#define US(K,O,B,N) ADD(K,O,B,N,4); ADD(K,O,B,N,8); ADD(K,O,B,N,32)
#define BOTH(K,B,N) US(K,0,B,N); US(K,1,B,N)
#define COUNTS(K) BOTH(K,1,1); BOTH(K,1,2); BOTH(K,1,4); BOTH(K,1,5); BOTH(K,1,6); BOTH(K,1,7); BOTH(K,1,8)
  US(0,0,1,0); BOTH(1,1,0);
  COUNTS(2); COUNTS(3); COUNTS(4); COUNTS(5); COUNTS(6); COUNTS(7);
  BOTH(8,1,2); BOTH(8,1,4); BOTH(8,1,8);
  BOTH(10,1,2); BOTH(10,1,4); BOTH(10,1,8);
  BOTH(9,2,8); BOTH(9,4,8);
#undef COUNTS
#undef BOTH
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
  // Resolve every kernel, then condition with raw and four-bank FP8 U=8
  // for at least two seconds. No clock/power overrides.
  for(auto& c:all) run(c);
  HIP(hipDeviceSynchronize());
  auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(std::chrono::steady_clock::now()<until) {
    run(all[1]); run(all[all.size()-5]); run(all[all.size()-2]); HIP(hipDeviceSynchronize());
  }
  Events events;
  std::mt19937 rng(seed);
  std::printf("round,kind,order,banks,ops,unroll,ms,cycles_per_mfma,cycles_per_loop,wave_p10,wave_p50,wave_p90,core_MHz,TFLOPs,max_abs_error,valid,bad_outputs\n");
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
      double total=0,ticks=0,maxerr=0; int bad_outputs=0; std::vector<double> intervals;
      for(int wave=0;wave<waves;++wave) {
        double cycles=double(st[4*wave+1]-st[4*wave]);
        total+=cycles; ticks+=double(st[4*wave+3]-st[4*wave+2]);
        intervals.push_back(cycles/steps);
      }
      for(int tid=0;tid<threads;++tid) for(int r=0;r<16;++r) {
        int lane=tid%64,i=8*(r/4)+4*(lane/32)+r%4,j=lane%32;
        double error=std::isfinite(y[tid*16+r])?std::abs(y[tid*16+r]-ref[i][j]):INFINITY;
        maxerr=std::max(maxerr,error);
        // These small dyadic operands have an exact FP32 accumulation
        // reference at the tested 4096/8192 lengths. Even a small transient
        // sign-flip hazard must not be hidden by a length-dependent tolerance.
        if(error!=0) {
          ++bad_outputs;
        }
      }
      if(bad_outputs) {
        ++invalid_samples;
        std::fprintf(stderr,"INVALID round=%d kind=%d order=%d banks=%d ops=%d U=%d bad_outputs=%d max_error=%.9g\n",round,c.kind,c.order,c.banks,c.ops,c.unroll,bad_outputs,maxerr);
      }
      std::sort(intervals.begin(),intervals.end());
      double cycles=total/(waves*double(steps));
      std::printf("%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3f,%.6f,%.9g,%d,%d\n",round,c.kind,c.order,c.banks,c.ops,c.unroll,ms,cycles,cycles*c.unroll,intervals[waves/10],intervals[waves/2],intervals[waves*9/10],total/ticks*wall_rate/1000,double(waves)*steps*16384/(ms*1e9),maxerr,bad_outputs==0,bad_outputs);
    }
    std::fflush(stdout);
  }
  std::fprintf(stderr,"COMPLETE %zu cases x %d rounds: all outputs checked against FP64; invalid_samples=%d (exclude their timing from valid comparisons)\n",all.size(),rounds,invalid_samples);
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
