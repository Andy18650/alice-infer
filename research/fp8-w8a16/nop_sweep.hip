// Standalone delay-window test. Keep the original measured probe unchanged.
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

// s_nop IMM means IMM+1 wait-state units, NOT IMM+1 core cycles. The
// independent nop_clock.hip probe measures ~4 core cycles per unit here.
// The template name Cycles below denotes these encoded wait units.
// Tie each delay block to its operand so it stays between producer/consumer.
template<int Cycles> __device__ __forceinline__ void delay(H4& value) {
  if constexpr(Cycles >= 16) {
    asm volatile("s_nop 15" : "+v"(value));
    delay<Cycles-16>(value);
  } else if constexpr(Cycles > 0) {
    asm volatile("s_nop %1" : "+v"(value) : "n"(Cycles-1));
  }
}

template<int Cycles> __device__ __forceinline__ H4 convert(uint32_t w, H4 previous) {
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
  H4 next=__builtin_bit_cast(H4,(U2{lo,hi}));
  delay<Cycles>(next);
  return next;
}

// Cycles=-1 is the pure-FP16 control (no conversion or inserted delay).
// Unroll is MFMAs per backedge, not independent accumulator chains.
template<int Cycles,int Unroll>
__global__ __launch_bounds__(256) void sweep_kernel(
    const H4* acts,const uint32_t* packed,const U2* expanded,
    float* output,uint64_t* stamps,int steps) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x, lane=threadIdx.x%64;
  H4 a=acts[lane], current=__builtin_bit_cast(H4,expanded[lane]);
  uint32_t w=packed[lane];
  F16 c={};
  uint64_t begin=clock64(), wall_begin=wall_clock64();
#pragma clang loop unroll(disable)
  for(int i=0;i<steps;i+=Unroll) {
#pragma unroll
    for(int j=0;j<Unroll;++j) {
      if constexpr(Cycles>=0) {
        H4 next=convert<Cycles>(w,current);
        __builtin_amdgcn_sched_barrier(0);
        c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,current,c,0,0,0);
        __builtin_amdgcn_sched_barrier(0);
        current=next;
      } else {
        c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,current,c,0,0,0);
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
struct Case { int cycles,unroll; Launch launch; };
template<int Cycles,int Unroll> void launch(
    const H4* a,const uint32_t* q,const U2* h,float* y,uint64_t* st,int steps,int blocks) {
  sweep_kernel<Cycles,Unroll><<<blocks,256>>>(a,q,h,y,st,steps);
  HIP(hipGetLastError());
}
std::vector<Case> cases() {
  std::vector<Case> v;
#define ADD(D,U) v.push_back({D,U,launch<D,U>})
#define DELAY(D) ADD(D,2); ADD(D,8); ADD(D,32)
  DELAY(-1); DELAY(0); DELAY(1); DELAY(2); DELAY(3); DELAY(4); DELAY(5); DELAY(6);
  DELAY(7); DELAY(9); DELAY(11); DELAY(15); DELAY(17);
  DELAY(8); DELAY(10); DELAY(12); DELAY(14); DELAY(16); DELAY(18);
  DELAY(20); DELAY(22); DELAY(24); DELAY(26); DELAY(28); DELAY(30);
  DELAY(32); DELAY(34); DELAY(36); DELAY(38); DELAY(40); DELAY(42);
  DELAY(44); DELAY(46); DELAY(48); DELAY(50); DELAY(52); DELAY(54);
  DELAY(56); DELAY(58); DELAY(60); DELAY(62); DELAY(64);
  DELAY(72); DELAY(80); DELAY(96);
#undef DELAY
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
  // Resolve every kernel before timing, then condition with alternating FP16
  // and zero-delay FP8 for at least two seconds. No clock/power overrides.
  for(auto& c:all) run(c);
  HIP(hipDeviceSynchronize());
  auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(std::chrono::steady_clock::now()<until) {
    run(all[1]); run(all[4]); HIP(hipDeviceSynchronize());
  }
  Events events;
  std::mt19937 rng(seed);
  std::printf("round,delay,unroll,ms,cycles_per_mfma,cycles_per_loop,wave_p10,wave_p50,wave_p90,core_MHz,TFLOPs,max_abs_error\n");
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
        double error=std::abs(y[tid*16+r]-ref[i][j]); maxerr=std::max(maxerr,error);
        if(!std::isfinite(y[tid*16+r]) || error>0.001*(1+std::abs(ref[i][j]))) {
          std::fprintf(stderr,"FAIL delay=%d unroll=%d tid=%d r=%d got=%.9g ref=%.12g\n",c.cycles,c.unroll,tid,r,y[tid*16+r],ref[i][j]);
          throw std::runtime_error("all-output FP64 oracle mismatch");
        }
      }
      std::sort(intervals.begin(),intervals.end());
      double cycles=total/(waves*double(steps));
      std::printf("%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3f,%.6f,%.9g\n",round,c.cycles,c.unroll,ms,cycles,cycles*c.unroll,intervals[waves/10],intervals[waves/2],intervals[waves*9/10],total/ticks*wall_rate/1000,double(waves)*steps*16384/(ms*1e9),maxerr);
    }
    std::fflush(stdout);
  }
  std::fprintf(stderr,"PASS %zu cases x %d rounds: all outputs checked against FP64\n",all.size(),rounds);
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
