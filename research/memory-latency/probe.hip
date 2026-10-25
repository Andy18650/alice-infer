// gfx90a dependent VMEM completion latency. Standalone research only.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while(0)
using U4=uint32_t __attribute__((ext_vector_type(4))); // inline-asm VGPR tuple
struct Result { uint64_t cycles,ticks,single_cycles,empty_cycles; uint32_t index,hw0,hw1,last[4]; };
__device__ __forceinline__ uint64_t empty_interval() {
  uint64_t a,b;
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_memtime %0\n\ts_waitcnt lgkmcnt(0)\n\ts_waitcnt vmcnt(0)\n\ts_memtime %1\n\ts_waitcnt lgkmcnt(0)" : "=&s"(a),"=&s"(b) :: "memory");
  return b-a;
}
template<int Width,bool Glc>
__device__ __forceinline__ uint64_t timed_load(const uint32_t* p,U4& value) {
  uint64_t a,b;
#define TIMED_LOAD(OP) "s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_memtime %0\n\ts_waitcnt lgkmcnt(0)\n\t" OP "\n\ts_waitcnt vmcnt(0)\n\ts_memtime %1\n\ts_waitcnt lgkmcnt(0)"
  if constexpr(Width==4) {
    if constexpr(Glc) asm volatile(TIMED_LOAD("global_load_dwordx4 %2, %3, off glc") : "=&s"(a),"=&s"(b),"=v"(value) : "v"(p) : "memory");
    else asm volatile(TIMED_LOAD("global_load_dwordx4 %2, %3, off") : "=&s"(a),"=&s"(b),"=v"(value) : "v"(p) : "memory");
  } else {
    uint32_t x;
    if constexpr(Glc) asm volatile(TIMED_LOAD("global_load_dword %2, %3, off glc") : "=&s"(a),"=&s"(b),"=v"(x) : "v"(p) : "memory");
    else asm volatile(TIMED_LOAD("global_load_dword %2, %3, off") : "=&s"(a),"=&s"(b),"=v"(x) : "v"(p) : "memory");
    value=U4{x,0,0,0};
  }
#undef TIMED_LOAD
  return b-a;
}
__device__ __forceinline__ uint64_t core_time() {
  uint64_t t; asm volatile("s_memtime %0\n\ts_waitcnt lgkmcnt(0)" : "=s"(t) :: "memory"); return t;
}
__device__ __forceinline__ uint64_t steady_time() {
  uint64_t t; asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)" : "=s"(t) :: "memory"); return t;
}
__device__ __forceinline__ uint32_t hwid() {return __builtin_amdgcn_s_getreg((31u<<11)|4u);}
__global__ void initialize(uint32_t* p,const uint32_t* next,uint32_t nodes,uint32_t words,int width) {
  const size_t count=size_t(nodes)*words;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
    const uint32_t j=i%words, n=next[i/words];
    p[i]=(width==4 && j%4)? n^(0x9e3779b9u*(j%4)+j/4):n;
  }
}
template<int Width,bool Glc>
__device__ __forceinline__ U4 load(const uint32_t* p) {
  U4 v{};
  if constexpr(Width==4) {
    if constexpr(Glc) asm volatile("global_load_dwordx4 %0, %1, off glc\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
    else asm volatile("global_load_dwordx4 %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  } else {
    uint32_t x;
    if constexpr(Glc) asm volatile("global_load_dword %0, %1, off glc\n\ts_waitcnt vmcnt(0)" : "=v"(x) : "v"(p) : "memory");
    else asm volatile("global_load_dword %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=v"(x) : "v"(p) : "memory");
    v[0]=x;
  }
  return v;
}
template<int Width,bool Single,bool Glc,bool Control=false>
__global__ __launch_bounds__(64) void chase(const uint32_t* p,Result* out,uint32_t nodes,uint32_t words,uint32_t iterations,uint32_t origin) {
  if constexpr(Single) {if(threadIdx.x) return;}
  const uint32_t lane=threadIdx.x;
  uint32_t index=origin;
  U4 v{};
  // Warm the complete random cycle on THIS wave/CU, not in another dispatch.
  if constexpr(!Control) {
#pragma clang loop unroll(disable)
    for(uint32_t j=0;j<nodes;++j) {v=load<Width,Glc>(p+size_t(index)*words+lane*Width);index=v[0];}
  }
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
  const uint32_t h0=hwid();
  const uint64_t empty=empty_interval();
  uint64_t single=0;
  if constexpr(!Control) {single=timed_load<Width,Glc>(p+size_t(index)*words+lane*Width,v);index=v[0];}
  const auto s0=steady_time(),c0=core_time();
#pragma clang loop unroll(disable)
  for(uint32_t j=0;j<iterations;++j) {
    if constexpr(Control) {
      // A dependent VALU-loop floor, not an exact address/timer subtraction.
      asm volatile("v_add_u32 %0, 1, %0\n\ts_waitcnt vmcnt(0)" : "+v"(index) :: "memory");
    } else {v=load<Width,Glc>(p+size_t(index)*words+lane*Width);index=v[0];}
  }
  const auto c1=core_time(),s1=steady_time();
  const uint32_t h1=hwid();
  Result r{c1-c0,s1-s0,single,empty,index,h0,h1,{v[0],v[1],v[2],v[3]}};
  out[lane]=r;
}
void launch(int mode,bool glc,bool control,const uint32_t* p,Result* out,uint32_t nodes,uint32_t words,uint32_t iters,uint32_t origin,hipStream_t s) {
#define L(W,S,G,C) chase<W,S,G,C><<<1,64,0,s>>>(p,out,nodes,words,iters,origin)
  if(control) {L(1,true,false,true);return;}
  if(mode==0) {if(glc){L(1,true,true,false);}else{L(1,true,false,false);}}
  else if(mode==1) {if(glc){L(1,false,true,false);}else{L(1,false,false,false);}}
  else {if(glc){L(4,false,true,false);}else{L(4,false,false,false);}}
#undef L
}
int main(int argc,char** argv) try {
  int rounds=3;uint32_t iterations=65536;uint32_t seed=12345;
  for(int i=1;i<argc;++i) {
    const std::string key=argv[i];if(i+1>=argc)throw std::runtime_error("missing value");
    const uint32_t value=std::stoul(argv[++i]);
    if(key=="--rounds")rounds=value;else if(key=="--iterations")iterations=value;else if(key=="--seed")seed=value;else throw std::runtime_error("unknown argument "+key);
  }
  if(rounds<1 || !iterations)throw std::runtime_error("positive rounds/iterations required");
  HIP(hipSetDevice(0));hipDeviceProp_t props{};HIP(hipGetDeviceProperties(&props,0));
  if(std::string(props.gcnArchName).find("gfx90a")==std::string::npos || props.warpSize!=64)throw std::runtime_error("requires gfx90a wave64");
  int khz=0;HIP(hipDeviceGetAttribute(&khz,hipDeviceAttributeWallClockRate,0));
  char pci[64];HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
  std::fprintf(stderr,"device=0 pci=%s arch=%s CUs=%d steady_khz=%d seed=%u model=none context=none automatic_clocks\n",pci,props.gcnArchName,props.multiProcessorCount,khz,seed);
  hipStream_t s;HIP(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));Result* out;HIP(hipMalloc(&out,64*sizeof(Result)));
  std::puts("mode,policy,working_bytes,node_bytes,nodes,round,iterations,cycles_per_step,ns_per_step,effective_MHz,single_cycles,empty_cycles,hwid,final_index");
  std::vector<size_t> sizes={4096,16384,65536,262144,1048576,4194304,16777216,67108864,268435456};
  std::mt19937 rng(seed);std::shuffle(sizes.begin(),sizes.end(),rng);
  for(size_t bytes:sizes) for(int mode:{0,1,2}) {
    const uint32_t words=mode==0?32:mode==1?64:256; // 128 / 256 / 1024 B nodes
    const uint32_t nodes=bytes/(words*4),width=mode==2?4:1;
    std::vector<uint32_t> order(nodes),next(nodes);std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);
    for(uint32_t i=0;i<nodes;++i)next[order[i]]=order[(i+1)%nodes];
    uint32_t *p,*mapping;HIP(hipMalloc(&p,bytes));HIP(hipMalloc(&mapping,nodes*4));
    HIP(hipMemcpyAsync(mapping,next.data(),nodes*4,hipMemcpyHostToDevice,s));initialize<<<104,256,0,s>>>(p,mapping,nodes,words,width);HIP(hipGetLastError());HIP(hipStreamSynchronize(s));
    for(int round=0;round<rounds;++round)for(int kind:{0,1,2}) {
      const bool control=kind==2;if(control && mode!=0)continue;
      const uint32_t origin=order[(size_t(round)*7919)%nodes];
      HIP(hipMemsetAsync(out,0xff,64*sizeof(Result),s));launch(mode,kind==1,control,p,out,nodes,words,iterations,origin,s);HIP(hipGetLastError());HIP(hipStreamSynchronize(s));
      Result results[64];HIP(hipMemcpy(results,out,sizeof(results),hipMemcpyDeviceToHost));
      uint32_t expected=origin;
      if(control)expected+=iterations;else for(uint32_t i=0;i<iterations+1;++i)expected=next[expected];
      const int lanes=mode==0?1:64;
      for(int lane=0;lane<lanes;++lane) {
        const auto&r=results[lane];if(r.index!=expected || !r.cycles || !r.ticks || r.hw0!=r.hw1)throw std::runtime_error("chain/timestamp/HW_ID validation failed");
        if(!control && r.last[0]!=expected)throw std::runtime_error("load result validation failed");
        if(mode==2)for(uint32_t j=1;j<4;++j)if(r.last[j]!=(expected^(0x9e3779b9u*j+lane)))throw std::runtime_error("vector payload mismatch");
      }
      const auto&r=results[0];const double ns=double(r.ticks)*1e6/khz;
      std::printf("%s,%s,%zu,%u,%u,%d,%u,%.6f,%.6f,%.3f,%llu,%llu,%u,%u\n",control?"dependent-add":mode==0?"lane-dword":mode==1?"wave-dword":"wave-dwordx4",control?"control":kind==1?"glc":"default",bytes,words*4,nodes,round,iterations,double(r.cycles)/iterations,ns/iterations,double(r.cycles)/ns*1000,static_cast<unsigned long long>(r.single_cycles),static_cast<unsigned long long>(r.empty_cycles),r.hw0,r.index);std::fflush(stdout);
    }
    HIP(hipFree(mapping));HIP(hipFree(p));
  }
  HIP(hipFree(out));HIP(hipStreamDestroy(s));return 0;
} catch(const std::exception&e) {std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
