// Matched conversion-before vs conversion-after MFMA test; originals unchanged.
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

// Mode 0: pure FP16. Mode 1: original convert-next / MFMA-current order.
// Mode 2: MFMA-current / convert-future order, with two primed operands.
// Unroll is MFMAs per backedge, not independent accumulator chains.
template<int Mode,int Unroll>
__global__ __launch_bounds__(256) void order_kernel(
    const H4* acts,const uint32_t* packed,const U2* expanded,
    float* output,uint64_t* stamps,int steps) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x, lane=threadIdx.x%64;
  H4 a=acts[lane], current=__builtin_bit_cast(H4,expanded[lane]);
  uint32_t w=packed[lane];
  H4 ahead;
  if constexpr(Mode==2) {
    // Both initial operands are converted in the prologue. Keep one future
    // operand ready, so post-MFMA conversion is NOT an immediate consumer.
    current=convert(w,current);
    ahead=convert(w,current);
    __builtin_amdgcn_sched_barrier(0);
  }
  F16 c={};
  uint64_t begin=clock64(), wall_begin=wall_clock64();
#pragma clang loop unroll(disable)
  for(int i=0;i<steps;i+=Unroll) {
#pragma unroll
    for(int j=0;j<Unroll;++j) {
      if constexpr(Mode==1) {
        H4 next=convert(w,current);
        __builtin_amdgcn_sched_barrier(0);
        c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,current,c,0,0,0);
        __builtin_amdgcn_sched_barrier(0);
        current=next;
      } else if constexpr(Mode==2) {
        c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,current,c,0,0,0);
        __builtin_amdgcn_sched_barrier(0);
        current=ahead;
        ahead=convert(w,current);
        __builtin_amdgcn_sched_barrier(0);
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
struct Case { int mode,unroll; Launch launch; };
template<int Mode,int Unroll> void launch(
    const H4* a,const uint32_t* q,const U2* h,float* y,uint64_t* st,int steps,int blocks) {
  order_kernel<Mode,Unroll><<<blocks,256>>>(a,q,h,y,st,steps);
  HIP(hipGetLastError());
}
std::vector<Case> cases() {
  std::vector<Case> v;
#define ADD(M,U) v.push_back({M,U,launch<M,U>})
#define MODE(M) ADD(M,2); ADD(M,8); ADD(M,32)
  MODE(0); MODE(1); MODE(2);
#undef MODE
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
  // Resolve every kernel, then condition with all three U=8 variants
  // for at least two seconds. No clock/power overrides.
  for(auto& c:all) run(c);
  HIP(hipDeviceSynchronize());
  auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(std::chrono::steady_clock::now()<until) {
    run(all[1]); run(all[4]); run(all[7]); HIP(hipDeviceSynchronize());
  }
  Events events;
  std::mt19937 rng(seed);
  std::printf("round,mode,unroll,ms,cycles_per_mfma,cycles_per_loop,wave_p10,wave_p50,wave_p90,core_MHz,TFLOPs,max_abs_error\n");
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
          std::fprintf(stderr,"FAIL mode=%d unroll=%d tid=%d r=%d got=%.9g ref=%.12g\n",c.mode,c.unroll,tid,r,y[tid*16+r],ref[i][j]);
          throw std::runtime_error("all-output FP64 oracle mismatch");
        }
      }
      std::sort(intervals.begin(),intervals.end());
      double cycles=total/(waves*double(steps));
      std::printf("%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3f,%.6f,%.9g\n",round,c.mode,c.unroll,ms,cycles,cycles*c.unroll,intervals[waves/10],intervals[waves/2],intervals[waves*9/10],total/ticks*wall_rate/1000,double(waves)*steps*16384/(ms*1e9),maxerr);
    }
    std::fflush(stdout);
  }
  std::fprintf(stderr,"PASS %zu cases x %d rounds: all outputs checked against FP64\n",all.size(),rounds);
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
