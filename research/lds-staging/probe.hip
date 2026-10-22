// Standalone gfx90a global->LDS issue-backpressure experiment, not GEMM.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do { auto status_=(x); if(status_!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(status_)); } while(0)
constexpr size_t kSourceBytes=1024*1024,kLdsWords=64*1024/4;
constexpr uint32_t kSentinel=0xdeadbeef,kGuard=0x76543210;
// All writes to the same LDS word have identical bits, even across waves,
// source wraps and unordered returns. Different lanes retain varying bits.
__host__ __device__ uint32_t payload(uint32_t i) { return 0xa5a50000u^((i%64)*0x9e3779b9u); }
#include "loads.hpp"
__global__ __launch_bounds__(256) void staging(const uint32_t* source,uint32_t* out,uint64_t* stamps,uint32_t groups,int private_lds) {
  extern __shared__ __attribute__((aligned(16))) uint32_t smem[];
  const uint32_t lane=threadIdx.x%64,wave=threadIdx.x/64;
  for(uint32_t i=threadIdx.x;i<kLdsWords;i+=256)smem[i]=kSentinel;
  __syncthreads(); // Initialization only, before the timed asm.
  const uintptr_t src=reinterpret_cast<uintptr_t>(source)+wave*(kSourceBytes/4)+lane*4;
  const uint32_t dst=__builtin_amdgcn_readfirstlane(uint32_t(reinterpret_cast<uintptr_t>(smem))+(private_lds?wave*16384:0));
  uint64_t b,i,c,wb,wi,wc;
  load_loop(uint32_t(src),uint32_t(src>>32),dst,groups,b,i,c,wb,wi,wc);
  // Terminal VM0 completed this wave's copies; join other waves before check.
  __syncthreads();
  for(uint32_t j=threadIdx.x;j<kLdsWords;j+=256)out[j]=smem[j];
  if(lane==0) { stamps[wave*6]=b;stamps[wave*6+1]=i;stamps[wave*6+2]=c;
    stamps[wave*6+3]=wb;stamps[wave*6+4]=wi;stamps[wave*6+5]=wc; }
}
template<class T> struct Buffer {
  T* p=nullptr;explicit Buffer(size_t n){HIP(hipMalloc(&p,n*sizeof(T)));}
  ~Buffer(){if(p)(void)hipFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
};
struct Events { hipEvent_t b,e;Events(){HIP(hipEventCreate(&b));HIP(hipEventCreate(&e));}~Events(){(void)hipEventDestroy(b);(void)hipEventDestroy(e);} };
int main(int argc,char** argv) try {
  std::string mode="private";uint32_t groups=16384;int rounds=9,warm_ms=200;
  for(int a=1;a<argc;++a) {
    std::string k=argv[a];if(k=="--help"){std::puts("--mode shared|private --groups N --rounds N --warmup-ms N; fixed1MiB source,64 loads/group/wave,64KiB allocated LDS");return 0;}
    if(++a==argc)throw std::runtime_error("missing argument");std::string v=argv[a];
    if(k=="--mode")mode=v;else if(k=="--groups"){const auto n=std::stoull(v);if(n==0||n>UINT32_MAX)throw std::runtime_error("groups out of range");groups=uint32_t(n);}
    else if(k=="--rounds")rounds=std::stoi(v);else if(k=="--warmup-ms")warm_ms=std::stoi(v);else throw std::runtime_error("unknown option "+k);
  }
  if((mode!="shared"&&mode!="private")||rounds<1||warm_ms<0)throw std::runtime_error("invalid arguments");
  HIP(hipSetDevice(0));int wallrate;HIP(hipDeviceGetAttribute(&wallrate,hipDeviceAttributeWallClockRate,0));
  char bdf[64];HIP(hipDeviceGetPCIBusId(bdf,sizeof(bdf),0));
  std::fprintf(stderr,"HIP0 BDF=%s, native O3 gfx90a, grid1/block256, source1MiB, source256KiB/wave, allocatedLDS64KiB, destination=%s, activeLDS=%dKiB, no model/context/clock changes; external NUMA placement\n",bdf,mode.c_str(),mode=="private"?64:16);
  Buffer<uint32_t> input(kSourceBytes/4),output(kLdsWords+32);Buffer<uint64_t> stamps(24);
  std::vector<uint32_t> src(kSourceBytes/4),got(kLdsWords+32,kGuard);for(size_t n=0;n<src.size();++n)src[n]=payload(n);
  HIP(hipMemcpy(input.p,src.data(),kSourceBytes,hipMemcpyHostToDevice));
  auto launch=[&](uint32_t n){staging<<<1,256,kLdsWords*4>>>(input.p,output.p+16,stamps.p,n,mode=="private");HIP(hipGetLastError());};
  launch(16);HIP(hipDeviceSynchronize()); // One full1MiB traversal before warming.
  auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(warm_ms);
  while(std::chrono::steady_clock::now()<until){launch(groups);HIP(hipDeviceSynchronize());}
  Events events;const uint64_t loads=uint64_t(groups)*64,bytes=loads*4*64*4;
  std::puts("round,mode,source_bytes,allocated_lds_bytes,active_lds_bytes,groups,loads_per_wave,requested_bytes,event_us,issue_cycles,completion_cycles,issue_us,completion_us,cycles_per_four_wave_load,requested_B_per_issue_cycle,requested_GBps,core_MHz,mismatches,guards,valid,begin0,issue0,done0,wall_begin0,wall_issue0,wall_done0,begin1,issue1,done1,wall_begin1,wall_issue1,wall_done1,begin2,issue2,done2,wall_begin2,wall_issue2,wall_done2,begin3,issue3,done3,wall_begin3,wall_issue3,wall_done3");
  for(int r=0;r<rounds;++r) {
    std::fill(got.begin(),got.end(),kGuard);HIP(hipMemcpy(output.p,got.data(),got.size()*4,hipMemcpyHostToDevice));
    HIP(hipEventRecord(events.b));launch(groups);HIP(hipEventRecord(events.e));HIP(hipEventSynchronize(events.e));
    float ms;HIP(hipEventElapsedTime(&ms,events.b,events.e));uint64_t ts[24];HIP(hipMemcpy(ts,stamps.p,sizeof(ts),hipMemcpyDeviceToHost));
    HIP(hipMemcpy(got.data(),output.p,got.size()*4,hipMemcpyDeviceToHost));int bad=0,guards=0;
    for(size_t j=0;j<kLdsWords;++j)bad+=got[j+16]!=(mode=="private"||j<4096?payload(j):kSentinel);
    for(int j=0;j<16;++j)guards+=(got[j]!=kGuard)+(got[kLdsWords+16+j]!=kGuard);
    uint64_t first=UINT64_MAX,issued=0,done=0,wfirst=UINT64_MAX,wissued=0,wdone=0;
    for(int w=0;w<4;++w){auto t=ts+w*6;first=std::min(first,t[0]);issued=std::max(issued,t[1]);done=std::max(done,t[2]);wfirst=std::min(wfirst,t[3]);wissued=std::max(wissued,t[4]);wdone=std::max(wdone,t[5]);}
    const double cycles=double(issued-first),completed=double(done-first),us=double(wissued-wfirst)*1000/wallrate,doneus=double(wdone-wfirst)*1000/wallrate;
    std::printf("%d,%s,%zu,%zu,%d,%u,%llu,%llu,%.6f,%.0f,%.0f,%.6f,%.6f,%.9f,%.9f,%.6f,%.3f,%d,%d,%d",r,mode.c_str(),kSourceBytes,kLdsWords*4,mode=="private"?65536:16384,groups,(unsigned long long)loads,(unsigned long long)bytes,ms*1000,cycles,completed,us,doneus,cycles/loads,bytes/cycles,bytes/(us*1000),cycles/us,bad,guards,!bad&&!guards);
    for(auto t:ts)std::printf(",%llu",(unsigned long long)t);std::puts("");
    if(bad||guards)throw std::runtime_error("LDS/guard verification failed");
  }
  std::fprintf(stderr,"COMPLETE all LDS words/guards exact; no periodic waits/barriers/consumers\n");return 0;
} catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
