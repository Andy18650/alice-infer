// Equal-work 104x1 vs 13x8 CTA grids. Four waves stay along N.
// Real packed A is indexed by M tile, B by N tile. Earlier probes unchanged.
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
constexpr int kA=13,kB=17,kPadding=10;
struct Ready { H4 a,b; };

__device__ __forceinline__ uint32_t load8(const uint32_t* p) {
  uint32_t x; asm volatile("global_load_dword %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=&v"(x) : "v"(p) : "memory"); return x;
}
__device__ __forceinline__ H4 load16(const U2* p) {
  U2 x; asm volatile("global_load_dwordx2 %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=&v"(x) : "v"(p) : "memory"); return __builtin_bit_cast(H4,x);
}
__device__ __forceinline__ void issue_pair8(const U2* ap,const uint32_t* bp,U2& a,uint32_t& b) {
  asm volatile("global_load_dwordx2 %0, %2, off\n\tglobal_load_dword %1, %3, off"
               : "=&v"(a),"=&v"(b) : "v"(ap),"v"(bp) : "memory");
}
__device__ __forceinline__ void issue_pair16(const U2* ap,const U2* bp,U2& a,U2& b) {
  asm volatile("global_load_dwordx2 %0, %2, off\n\tglobal_load_dwordx2 %1, %3, off"
               : "=&v"(a),"=&v"(b) : "v"(ap),"v"(bp) : "memory");
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
// Wait and FIRST reads stay in the same asm, avoiding premature compiler
// copies of asynchronous load results (the previous pipeline's safety rule).
template<int Depth> __device__ __forceinline__ Ready take_pair8(U2 a,uint32_t b,H4 keepa,H4 keepb) {
  uint32_t al,ah,bl,bh;
  asm volatile("s_waitcnt vmcnt(%11)\n\tv_mov_b32 %0, %4\n\tv_mov_b32 %1, %5\n\t"
      "v_perm_b32 %2, 0, %6, %7\n\tv_perm_b32 %3, 0, %6, %8\n\t"
      "v_pk_ashrrev_i16 %2, 1, %2 op_sel_hi:[0,1]\n\t"
      "v_pk_ashrrev_i16 %3, 1, %3 op_sel_hi:[0,1]\n\t"
      "v_and_b32 %2, %9, %2\n\tv_and_b32 %3, %9, %3\n\t"
      "v_pk_mul_f16 %2, %2, %10\n\tv_pk_mul_f16 %3, %3, %10"
      : "=&v"(al),"=&v"(ah),"=&v"(bl),"=&v"(bh)
      : "v"(a[0]),"v"(a[1]),"v"(b),"s"(0x01040004u),"s"(0x03040204u),"s"(0xbfffbfffu),"s"(0x5c005c00u),"n"(2*Depth-2),"v"(keepa),"v"(keepb) : "memory");
  return {__builtin_bit_cast(H4,(U2{al,ah})),__builtin_bit_cast(H4,(U2{bl,bh}))};
}
template<int Depth> __device__ __forceinline__ Ready take_pair16(U2 a,U2 b,H4 keepa,H4 keepb) {
  uint32_t al,ah,bl,bh;
  asm volatile("s_waitcnt vmcnt(%8)\n\tv_mov_b32 %0, %4\n\tv_mov_b32 %1, %5\n\t"
               "v_mov_b32 %2, %6\n\tv_mov_b32 %3, %7"
               : "=&v"(al),"=&v"(ah),"=&v"(bl),"=&v"(bh)
               : "v"(a[0]),"v"(a[1]),"v"(b[0]),"v"(b[1]),"n"(2*Depth-2),"v"(keepa),"v"(keepb) : "memory");
  return {__builtin_bit_cast(H4,(U2{al,ah})),__builtin_bit_cast(H4,(U2{bl,bh}))};
}
template<int D,class T> __device__ __forceinline__ void drain_pair(const U2* a,const T* b) {
#define P(I) "v"(a[I]),"v"(b[I])
  if constexpr(D==1) asm volatile("s_waitcnt vmcnt(0)" : : P(0) : "memory");
  if constexpr(D==2) asm volatile("s_waitcnt vmcnt(0)" : : P(0),P(1) : "memory");
  if constexpr(D==4) asm volatile("s_waitcnt vmcnt(0)" : : P(0),P(1),P(2),P(3) : "memory");
  if constexpr(D==8) asm volatile("s_waitcnt vmcnt(0)" : : P(0),P(1),P(2),P(3),P(4),P(5),P(6),P(7) : "memory");
#undef P
}

// Format 0=FP16 B, 1=FP8 B. Both launch shapes use this SAME kernel entry.
// Both cases load real A/B pairs; no changes to wave tiling or pipeline policy.
template<int Format,int Depth,int Unroll>
__global__ __launch_bounds__(256) void grid_kernel(
    const U2* activation,const uint32_t* stream8,const U2* stream16,
    float* output,uint64_t* stamps,int steps) {
  static_assert(Depth>0 && Unroll%Depth==0);
  int tid=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
  int lane=threadIdx.x%64,wave=tid/64;
  int nt=blockIdx.x*4+threadIdx.x/64;
  size_t base=size_t(nt)*(steps+kPadding)*64+lane;
  size_t abase=size_t(blockIdx.y)*(steps+kPadding)*64+lane;
  H4 a[2],b[2]; U2 qa[Depth],q16[Depth]; uint32_t q8[Depth];
  a[0]=load16(activation+abase); a[1]=load16(activation+abase+64);
  if constexpr(Format==0) { b[0]=load16(stream16+base); b[1]=load16(stream16+base+64); }
  else { b[0]=convert(load8(stream8+base),a[0]); b[1]=convert(load8(stream8+base+64),b[0]); }
  F16 c={};
  asm volatile("s_waitcnt vmcnt(0)" : "+a"(c) : "v"(a[0]),"v"(a[1]),"v"(b[0]),"v"(b[1]));
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for(int r=0;r<Depth-1;++r) {
    size_t ai=abase+size_t(r+2)*64,bi=base+size_t(r+2)*64;
    if constexpr(Format==0) issue_pair16(activation+ai,stream16+bi,qa[r],q16[r]);
    else issue_pair8(activation+ai,stream8+bi,qa[r],q8[r]);
  }
  __builtin_amdgcn_sched_barrier(0);
  uint64_t begin=clock64(),wall_begin=wall_clock64();
#pragma clang loop unroll(disable)
  for(int i=0;i<steps;i+=Unroll) {
#pragma unroll
    for(int j=0;j<Unroll;++j) {
      c=__builtin_amdgcn_mfma_f32_32x32x8f16(a[j%2],b[j%2],c,0,0,0);
      __builtin_amdgcn_sched_barrier(0);
      int tail=(j+Depth-1)%Depth,head=j%Depth;
      size_t s=size_t(i+j+Depth+1),ai=abase+s*64,bi=base+s*64;
      if constexpr(Format==0) issue_pair16(activation+ai,stream16+bi,qa[tail],q16[tail]);
      else issue_pair8(activation+ai,stream8+bi,qa[tail],q8[tail]);
      __builtin_amdgcn_sched_barrier(0);
      Ready next;
      if constexpr(Format==0) next=take_pair16<Depth>(qa[head],q16[head],a[(j+1)%2],b[(j+1)%2]);
      else next=take_pair8<Depth>(qa[head],q8[head],a[(j+1)%2],b[(j+1)%2]);
      a[j%2]=next.a; b[j%2]=next.b;
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  if constexpr(Format==0) drain_pair<Depth>(qa,q16);
  else drain_pair<Depth>(qa,q8);
  __builtin_amdgcn_sched_barrier(0);
  uint64_t end=clock64(),wall_end=wall_clock64();
  if(lane==0) { stamps[4*wave]=begin; stamps[4*wave+1]=end; stamps[4*wave+2]=wall_begin; stamps[4*wave+3]=wall_end; }
  for(int r=0;r<16;++r) output[tid*16+r]=c[r];
}

// A holds eight distinct M streams; B holds the 416 N streams needed by
// the 104x1 baseline. The 13x8 grid uses the same addresses for shared tiles.
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
using Launch=void(*)(const U2*,const uint32_t*,const U2*,float*,uint64_t*,int,int);
struct Case { int format,grid_m,depth,unroll; Launch launch; };
template<int F,int M,int D,int U> void launch(const U2* a,const uint32_t* q,const U2* h,float* y,uint64_t* st,int steps,int blocks) {
  grid_kernel<F,D,U><<<dim3(blocks/M,M),256>>>(a,q,h,y,st,steps); HIP(hipGetLastError());
}
std::vector<Case> cases() {
  std::vector<Case> v;
#define ADD(F,A,D,U) v.push_back({F,A,D,U,launch<F,A,D,U>})
#define US(F,A,D) ADD(F,A,D,8); ADD(F,A,D,32)
#define DEPTHS(F,A) US(F,A,1); US(F,A,2); US(F,A,4); US(F,A,8)
  DEPTHS(0,1); DEPTHS(1,1); DEPTHS(0,8); DEPTHS(1,8);
#undef DEPTHS
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
  int steps=argc>1?std::stoi(argv[1]):4096,rounds=argc>2?std::stoi(argv[2]):9;
  unsigned seed=argc>3?unsigned(std::stoul(argv[3])):1729;
  if(steps<=0 || steps>8192 || steps%32 || rounds<=0) throw std::runtime_error("steps: positive multiple of 32, <=8192; rounds positive");
  HIP(hipSetDevice(0)); hipDeviceProp_t prop{}; HIP(hipGetDeviceProperties(&prop,0));
  int wall_rate=0; HIP(hipDeviceGetAttribute(&wall_rate,hipDeviceAttributeWallClockRate,0));
  int blocks=prop.multiProcessorCount,threads=blocks*256,waves=threads/64;
  if(blocks!=104) throw std::runtime_error("13x8 experiment requires this 104-CU GCD");
  size_t outputs=size_t(threads)*16,tile_records=size_t(steps+kPadding)*64;
  size_t na=tile_records*8,nb=tile_records*waves;
  std::fprintf(stderr,"device=%s arch=%s CUs=%d steps=%d rounds=%d seed=%u waves=%d allocated_A_bytes=%zu allocated_B8_bytes=%zu allocated_B16_bytes=%zu; native O3, CPU+memory NUMA0 externally, no model/context\n",prop.name,prop.gcnArchName,blocks,steps,rounds,seed,waves,na*8,nb*4,nb*8);
  for(int gm: {1,8}) std::fprintf(stderr,"grid=%dx%d logical_M=%d N=%d K=%d active_A_bytes=%zu active_B8_bytes=%zu active_B16_bytes=%zu\n",blocks/gm,gm,32*gm,32*waves/gm,8*steps,tile_records*gm*8,tile_records*(waves/gm)*4,tile_records*(waves/gm)*8);
  std::vector<U2> as(kA*64),bs16(kB*64); std::vector<uint32_t> bs8(kB*64);
  double av[kA][32][8],bv[kB][8][32]; std::vector<double> dot(size_t(kA)*kB*1024);
  for(int m=0;m<kA;++m) for(int lane=0;lane<64;++lane) {
    H4 v;
    for(int t=0;t<4;++t) {
      int row=lane%32,k=4*(lane/32)+t;
      v[t]=_Float16(((row*3+k*5+m*(2+row%3)+(m*k)%5)%7-3)*0.125f);
      av[m][row][k]=float(v[t]);
    }
    as[m*64+lane]=std::bit_cast<U2>(v);
  }
  for(int m=0;m<kB;++m) for(int lane=0;lane<64;++lane) {
    H4 v; uint32_t word=0;
    for(int t=0;t<4;++t) {
      int x=(lane*19+t*7+m*11)%32;
      uint8_t byte=uint8_t(x|((lane+t+m)%3==0?128:0));
      word|=uint32_t(byte)<<(8*t); v[t]=_Float16(decode(byte)); bv[m][4*(lane/32)+t][lane%32]=float(v[t]);
    }
    bs8[m*64+lane]=word; bs16[m*64+lane]=std::bit_cast<U2>(v);
  }
  auto di=[](int a,int b,int row,int col) { return (size_t(a)*kB+b)*1024+row*32+col; };
  for(int a=0;a<kA;++a) for(int b=0;b<kB;++b) for(int row=0;row<32;++row) for(int col=0;col<32;++col)
    for(int k=0;k<8;++k) dot[di(a,b,row,col)]+=av[a][row][k]*bv[b][k][col];
  Device<U2> das(as.size()),dbs16(bs16.size()),activation(na),b16(nb);
  Device<uint32_t> dbs8(bs8.size()),b8(nb); Device<float> out(outputs+32); Device<uint64_t> stamps(waves*4);
  das.put(as); dbs8.put(bs8); dbs16.put(bs16);
  std::vector<double> reference[2]={std::vector<double>(outputs),std::vector<double>(outputs)};
  auto initialize=[&](int ae,int be) {
    fill_data<<<(nb+255)/256,256>>>(das.p,dbs8.p,dbs16.p,activation.p,b8.p,b16.p,nb,na,steps+kPadding,ae,be);
    HIP(hipGetLastError()); HIP(hipDeviceSynchronize());
    for(int gm: {1,8}) for(int wave=0;wave<waves;++wave) {
      int nw=waves/gm,mt=wave/nw,nt=wave%nw;
      int pair_counts[kA][kB]={};
      for(int s=0;s<steps;++s) {
        int a=(5*s+s/kA+ae*7+3*mt)%kA,b=(s+s/kB+nt*7+be*3)%kB;
        ++pair_counts[a][b];
      }
      for(int lane=0;lane<64;++lane) for(int r=0;r<16;++r) {
        int row=8*(r/4)+4*(lane/32)+r%4,col=lane%32; size_t t=(size_t(wave)*64+lane)*16+r;
        double value=0;
        for(int a=0;a<kA;++a) for(int b=0;b<kB;++b) value+=pair_counts[a][b]*dot[di(a,b,row,col)];
        reference[gm==8][t]=value;
      }
    }
  };
  auto all=cases();
  auto run=[&](const Case& c) { c.launch(activation.p,b8.p,b16.p,out.p+16,stamps.p,steps,blocks); };
  auto verify=[&](int gm) {
    auto y=out.get(outputs+32); double maxerr=0;
    for(size_t t=0;t<16;++t) if(std::bit_cast<uint32_t>(y[t])!=0xffffffffu || std::bit_cast<uint32_t>(y[outputs+16+t])!=0xffffffffu) throw std::runtime_error("output guard overwritten");
    for(size_t t=0;t<outputs;++t) {
      double error=std::isfinite(y[t+16])?std::abs(y[t+16]-reference[gm==8][t]):INFINITY;
      maxerr=std::max(maxerr,error);
    }
    return maxerr;
  };
  // Independent A-only, B-only, both-changed and restored-baseline fixtures.
  for(auto epochs: {std::pair{1,0},std::pair{0,1},std::pair{1,1},std::pair{0,0}}) {
    initialize(epochs.first,epochs.second);
    for(auto& c:all) {
      HIP(hipMemset(out.p,0xff,(outputs+32)*sizeof(float))); run(c); double error=verify(c.grid_m);
      if(error!=0) { std::fprintf(stderr,"FAIL ae=%d be=%d format=%d gridM=%d D=%d U=%d err=%.9g\n",epochs.first,epochs.second,c.format,c.grid_m,c.depth,c.unroll,error); throw std::runtime_error("changed-input oracle mismatch"); }
    }
  }
  std::fprintf(stderr,"PASS %zu independent A/B changed-input/poisoned-output cases and guards\n",all.size()*4);
  auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
  while(std::chrono::steady_clock::now()<until) { for(auto& c:all) if(c.unroll==8) run(c); HIP(hipDeviceSynchronize()); }
  Events events; std::mt19937 rng(seed); std::vector<int> order(all.size()); for(size_t i=0;i<all.size();++i) order[i]=int(i);
  std::puts("round,format,grid_m,grid_n,depth,unroll,steps,ms,cycles_per_mfma,wave_p10,wave_p50,wave_p90,core_MHz,TFLOPs,useful_B_GBs,useful_A_request_GBs,unique_B_equiv_GBs,unique_A_equiv_GBs,requested_B_GBs,requested_A_GBs,max_abs_error,valid");
  int invalid=0;
  for(int round=0;round<rounds;++round) {
    std::shuffle(order.begin(),order.end(),rng);
    for(int index:order) {
      auto& c=all[index]; HIP(hipEventRecord(events.begin)); for(int n=0;n<8;++n) run(c);
      HIP(hipEventRecord(events.end)); HIP(hipEventSynchronize(events.end)); float ms; HIP(hipEventElapsedTime(&ms,events.begin,events.end)); ms/=8;
      double maxerr=verify(c.grid_m); if(maxerr!=0) { ++invalid; std::fprintf(stderr,"INVALID round=%d format=%d gridM=%d D=%d U=%d err=%.9g\n",round,c.format,c.grid_m,c.depth,c.unroll,maxerr); }
      auto st=stamps.get(waves*4); double cycles=0,ticks=0; std::vector<double> intervals;
      for(int w=0;w<waves;++w) { double n=double(st[4*w+1]-st[4*w]); cycles+=n; ticks+=double(st[4*w+3]-st[4*w+2]); intervals.push_back(n/steps); }
      std::sort(intervals.begin(),intervals.end());
      int width=c.format?4:8; double tf=double(waves)*steps*16384/(ms*1e9);
      double bb=double(waves)*steps*64*width/(ms*1e6),ab=double(waves)*steps*512/(ms*1e6);
      double br=double(waves)*(steps+c.depth+1)*64*width/(ms*1e6),ar=double(waves)*(steps+c.depth+1)*512/(ms*1e6);
      double ub=bb/c.grid_m,ua=ab*c.grid_m/waves;
      std::printf("%d,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.9g,%d\n",round,c.format,c.grid_m,blocks/c.grid_m,c.depth,c.unroll,steps,ms,cycles/(waves*double(steps)),intervals[waves/10],intervals[waves/2],intervals[waves*9/10],cycles/ticks*wall_rate/1000,tf,bb,ab,ub,ua,br,ar,maxerr,maxerr==0);
    }
    std::fflush(stdout);
  }
  std::fprintf(stderr,"COMPLETE %zu cases x %d rounds; exact FP64 checks, invalid_samples=%d\n",all.size(),rounds,invalid);
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
