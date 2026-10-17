// First real-weight-load baseline. Intentionally immediate waits, no prefetch
// tuning and no boundary redistribution. Register-only checkpoint stays intact.
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
using H4=_Float16 __attribute__((ext_vector_type(4)));
using U2=uint32_t __attribute__((ext_vector_type(2)));
using F16=float __attribute__((ext_vector_type(16)));
constexpr int kPatterns=17;

__device__ __forceinline__ uint32_t load8(const uint32_t* p) {
  uint32_t x;
  asm volatile("global_load_dword %0, %1, off\n\ts_waitcnt vmcnt(0)"
               : "=&v"(x) : "v"(p) : "memory");
  return x;
}
__device__ __forceinline__ H4 load16(const U2* p) {
  U2 x;
  asm volatile("global_load_dwordx2 %0, %1, off\n\ts_waitcnt vmcnt(0)"
               : "=&v"(x) : "v"(p) : "memory");
  return __builtin_bit_cast(H4,x);
}
__device__ __forceinline__ H4 convert(uint32_t w,H4 keep) {
  uint32_t lo,hi;
  asm volatile(
      "v_perm_b32 %0, 0, %2, %3\n\tv_perm_b32 %1, 0, %2, %4\n\t"
      "v_pk_ashrrev_i16 %0, 1, %0 op_sel_hi:[0,1]\n\t"
      "v_pk_ashrrev_i16 %1, 1, %1 op_sel_hi:[0,1]\n\t"
      "v_and_b32 %0, %5, %0\n\tv_and_b32 %1, %5, %1"
      : "=&v"(lo),"=&v"(hi)
      : "v"(w),"s"(0x01040004u),"s"(0x03040204u),"s"(0xbfffbfffu),"v"(keep));
  asm volatile("v_pk_mul_f16 %0, %0, %2\n\tv_pk_mul_f16 %1, %1, %2"
               : "+v"(lo),"+v"(hi) : "s"(0x5c005c00u));
  return __builtin_bit_cast(H4,(U2{lo,hi}));
}

// Mode 0 resident FP16, 1 resident FP8 with ordinary 8/8 conversion,
// 2 streamed FP16, 3 streamed FP8. No attempt to hide steady-state load waits.
template<int Mode,int Unroll>
__global__ __launch_bounds__(256) void load_kernel(
    const H4* activation,const uint32_t* stream8,const U2* stream16,
    float* output,uint64_t* stamps,int steps) {
  int tid=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x%64,wave=tid/64;
  size_t base=size_t(wave)*(steps+2)*64+lane;
  H4 a=activation[lane],b[2]; uint32_t seed=0;
  if constexpr(Mode==0) { b[0]=load16(stream16+base); b[1]=b[0]; }
  if constexpr(Mode==1) {
    seed=load8(stream8+base); b[0]=convert(seed,a); b[1]=convert(seed,b[0]);
  }
  if constexpr(Mode==2) { b[0]=load16(stream16+base); b[1]=load16(stream16+base+64); }
  if constexpr(Mode==3) {
    b[0]=convert(load8(stream8+base),a);
    b[1]=convert(load8(stream8+base+64),b[0]);
  }
  F16 c={};
  // Establish first-entry readiness outside the core-clock body, exactly as
  // in the checkpoint. No guessed fixed delay replaces memory completion.
  asm volatile("s_waitcnt vmcnt(0)" : "+a"(c) : "v"(a),"v"(b[0]),"v"(b[1]));
  __builtin_amdgcn_sched_barrier(0);
  uint64_t begin=clock64(),wall_begin=wall_clock64();
#pragma clang loop unroll(disable)
  for(int i=0;i<steps;i+=Unroll) {
#pragma unroll
    for(int j=0;j<Unroll;++j) {
      c=__builtin_amdgcn_mfma_f32_32x32x8f16(a,b[Mode==0?0:j%2],c,0,0,0);
      if constexpr(Mode!=0) {
        __builtin_amdgcn_sched_barrier(0);
        if constexpr(Mode==1) b[j%2]=convert(seed,b[(j+1)%2]);
        if constexpr(Mode==2) b[j%2]=load16(stream16+base+size_t(i+j+2)*64);
        if constexpr(Mode==3) {
          uint32_t future=load8(stream8+base+size_t(i+j+2)*64);
          b[j%2]=convert(future,b[(j+1)%2]);
        }
        __builtin_amdgcn_sched_barrier(0);
      }
    }
  }
  uint64_t end=clock64(),wall_end=wall_clock64();
  if(lane==0) {
    stamps[4*wave]=begin; stamps[4*wave+1]=end;
    stamps[4*wave+2]=wall_begin; stamps[4*wave+3]=wall_end;
  }
  for(int r=0;r<16;++r) output[tid*16+r]=c[r];
}

// Device initialization only, outside measurement. All 416 wave streams use
// disjoint addresses. A 289-step phase pattern catches simple stride/offset
// mistakes, rather than every record containing the same weight fragment.
__global__ void fill(const uint32_t* q,const U2* h,uint32_t* qout,U2* hout,
                     size_t n,int records_per_wave,int epoch) {
  size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(t<n) {
    int lane=int(t%64),wave=int(t/(size_t(records_per_wave)*64));
    int step=int(t/64%records_per_wave);
    int phase=(step+step/kPatterns+wave*7+epoch*3)%kPatterns;
    qout[t]=q[phase*64+lane]; hout[t]=h[phase*64+lane];
  }
}
float decode(uint8_t x) {
  int e=(x>>3)&15,m=x&7;
  float v=e?std::ldexp(float(8+m),e-10):std::ldexp(float(m),-9);
  return x&128?-v:v;
}
template<class T> struct Device {
  T* p=nullptr;
  explicit Device(size_t n) { HIP(hipMalloc(&p,n*sizeof(T))); }
  ~Device() { if(p) (void)hipFree(p); }
  Device(const Device&)=delete;
  void put(const std::vector<T>& v) { HIP(hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice)); }
  std::vector<T> get(size_t n) const { std::vector<T> v(n); HIP(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost)); return v; }
};
using Launch=void(*)(const H4*,const uint32_t*,const U2*,float*,uint64_t*,int,int);
struct Case { int mode,unroll; Launch launch; };
template<int M,int U> void launch(const H4* a,const uint32_t* q,const U2* h,float* y,uint64_t* st,int steps,int blocks) {
  load_kernel<M,U><<<blocks,256>>>(a,q,h,y,st,steps); HIP(hipGetLastError());
}
std::vector<Case> cases() {
  std::vector<Case> v;
#define ADD(M,U) v.push_back({M,U,launch<M,U>})
#define MODE(M) ADD(M,4); ADD(M,8); ADD(M,32)
  MODE(0); MODE(1); MODE(2); MODE(3);
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
  int steps=argc>1?std::stoi(argv[1]):4096,rounds=argc>2?std::stoi(argv[2]):9;
  unsigned seed=argc>3?unsigned(std::stoul(argv[3])):1729;
  if(steps<=0 || steps>8192 || steps%32 || rounds<=0) throw std::runtime_error("steps: positive multiple of 32, <=8192; rounds positive");
  HIP(hipSetDevice(0)); hipDeviceProp_t prop{}; HIP(hipGetDeviceProperties(&prop,0));
  int wall_rate=0; HIP(hipDeviceGetAttribute(&wall_rate,hipDeviceAttributeWallClockRate,0));
  int blocks=prop.multiProcessorCount,threads=blocks*256,waves=threads/64;
  size_t outputs=size_t(threads)*16,records=size_t(waves)*(steps+2)*64;
  std::fprintf(stderr,"device=%s arch=%s CUs=%d steps=%d rounds=%d seed=%u waves=%d records_per_wave=%d FP8_bytes=%zu FP16_bytes=%zu; native O3, no model/context, bind CPU+memory NUMA0 externally\n",prop.name,prop.gcnArchName,blocks,steps,rounds,seed,waves,steps+2,records*4,records*8);
  std::vector<H4> a(64); std::vector<uint32_t> q(kPatterns*64); std::vector<U2> h(kPatterns*64);
  double av[32][8],bv[kPatterns][8][32],dot[kPatterns][32][32]={};
  for(int lane=0;lane<64;++lane) for(int t=0;t<4;++t) {
    int k=4*(lane/32)+t,row=lane%32;
    a[lane][t]=_Float16(((row*3+k*5)%7-3)*0.125f); av[row][k]=float(a[lane][t]);
  }
  for(int m=0;m<kPatterns;++m) for(int lane=0;lane<64;++lane) {
    H4 v; uint32_t word=0;
    for(int t=0;t<4;++t) {
      int x=(lane*19+t*7+m*11)%32;
      uint8_t byte=uint8_t(x|((lane+t+m)%3==0?128:0));
      word|=uint32_t(byte)<<(8*t); v[t]=_Float16(decode(byte));
      bv[m][4*(lane/32)+t][lane%32]=float(v[t]);
    }
    q[m*64+lane]=word; h[m*64+lane]=std::bit_cast<U2>(v);
  }
  for(int m=0;m<kPatterns;++m) for(int i=0;i<32;++i) for(int j=0;j<32;++j)
    for(int k=0;k<8;++k) dot[m][i][j]+=av[i][k]*bv[m][k][j];
  Device<H4> da(64); Device<uint32_t> dq(q.size()),stream8(records); Device<U2> dh(h.size()),stream16(records);
  Device<float> out(outputs+32); Device<uint64_t> stamps(waves*4);
  da.put(a); dq.put(q); dh.put(h);
  std::vector<double> ref_resident(outputs),ref_stream(outputs);
  auto initialize=[&](int epoch) {
    fill<<<(records+255)/256,256>>>(dq.p,dh.p,stream8.p,stream16.p,records,steps+2,epoch);
    HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
    for(int wave=0;wave<waves;++wave) {
      int counts[kPatterns]={};
      for(int s=0;s<steps;++s) ++counts[(s+s/kPatterns+wave*7+epoch*3)%kPatterns];
      int first=(wave*7+epoch*3)%kPatterns;
      for(int lane=0;lane<64;++lane) for(int r=0;r<16;++r) {
        int row=8*(r/4)+4*(lane/32)+r%4,col=lane%32;
        size_t t=(size_t(wave)*64+lane)*16+r;
        ref_resident[t]=steps*dot[first][row][col];
        double value=0; for(int m=0;m<kPatterns;++m) value+=counts[m]*dot[m][row][col];
        ref_stream[t]=value;
      }
    }
  };
  auto all=cases();
  auto run=[&](const Case& c) { c.launch(da.p,stream8.p,stream16.p,out.p+16,stamps.p,steps,blocks); };
  auto verify=[&](int mode) {
    auto y=out.get(outputs+32); double maxerr=0;
    for(size_t t=0;t<16;++t) if(std::bit_cast<uint32_t>(y[t])!=0xffffffffu || std::bit_cast<uint32_t>(y[outputs+16+t])!=0xffffffffu)
      throw std::runtime_error("output guard overwritten");
    auto& ref=mode<2?ref_resident:ref_stream;
    for(size_t t=0;t<outputs;++t) {
      double error=std::isfinite(y[t+16])?std::abs(y[t+16]-ref[t]):INFINITY;
      maxerr=std::max(maxerr,error);
    }
    return maxerr;
  };
  // Poison every output and change the logical weight data without recompiling.
  // Consumed seed records are covered by the oracle. The final two lookahead
  // records are allocated but unused; their contents are not an oracle gate.
  for(int epoch: {1,0}) {
    initialize(epoch);
    for(auto& c:all) {
      HIP(hipMemset(out.p,0xff,(outputs+32)*sizeof(float))); run(c);
      double error=verify(c.mode);
      if(error!=0) { std::fprintf(stderr,"FAIL epoch=%d mode=%d U=%d error=%.9g\n",epoch,c.mode,c.unroll,error); throw std::runtime_error("changed-input/poisoned-output oracle mismatch"); }
    }
  }
  std::fprintf(stderr,"PASS 24 changed-input/poisoned-output cases and output guards\n");
  auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(std::chrono::steady_clock::now()<until) { for(auto& c:all) if(c.unroll==8) run(c); HIP(hipDeviceSynchronize()); }
  Events events; std::mt19937 rng(seed);
  std::vector<int> order(all.size()); for(size_t i=0;i<all.size();++i) order[i]=int(i);
  std::puts("round,mode,unroll,steps,ms,cycles_per_mfma,cycles_per_loop,wave_p10,wave_p50,wave_p90,core_MHz,TFLOPs,useful_weight_GBs,requested_weight_GBs,max_abs_error,valid");
  int invalid_samples=0;
  for(int round=0;round<rounds;++round) {
    std::shuffle(order.begin(),order.end(),rng);
    for(int index:order) {
      auto& c=all[index];
      HIP(hipEventRecord(events.begin)); for(int repeat=0;repeat<8;++repeat) run(c);
      HIP(hipEventRecord(events.end)); HIP(hipEventSynchronize(events.end));
      float ms; HIP(hipEventElapsedTime(&ms,events.begin,events.end)); ms/=8;
      double maxerr=verify(c.mode);
      if(maxerr!=0) { ++invalid_samples; std::fprintf(stderr,"INVALID round=%d mode=%d U=%d error=%.9g\n",round,c.mode,c.unroll,maxerr); }
      auto st=stamps.get(waves*4); double total=0,ticks=0; std::vector<double> intervals;
      for(int wave=0;wave<waves;++wave) {
        double cycles=double(st[4*wave+1]-st[4*wave]); total+=cycles;
        ticks+=double(st[4*wave+3]-st[4*wave+2]); intervals.push_back(cycles/steps);
      }
      std::sort(intervals.begin(),intervals.end());
      double cycles=total/(waves*double(steps)),tf=double(waves)*steps*16384/(ms*1e9);
      int width=c.mode==2?8:4;
      double useful=c.mode>=2?double(waves)*steps*64*width/(ms*1e6):0;
      double requested=c.mode>=2?double(records)*width/(ms*1e6):0;
      std::printf("%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.3f,%.6f,%.6f,%.6f,%.9g,%d\n",round,c.mode,c.unroll,steps,ms,cycles,cycles*c.unroll,intervals[waves/10],intervals[waves/2],intervals[waves*9/10],total/ticks*wall_rate/1000,tf,useful,requested,maxerr,maxerr==0);
    }
    std::fflush(stdout);
  }
  std::fprintf(stderr,"COMPLETE %zu cases x %d rounds; exact FP64 checks, invalid_samples=%d\n",all.size(),rounds,invalid_samples);
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
