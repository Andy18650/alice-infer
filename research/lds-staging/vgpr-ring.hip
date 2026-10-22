// One CTA,1/2/4 waves, direct global DWORDx4 ->32-slot VGPR ring.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do{auto status_=(x);if(status_!=hipSuccess)throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(status_));}while(0)
using U4=uint32_t __attribute__((ext_vector_type(4)));
constexpr size_t kSourceBytes=1048576;
constexpr uint32_t kGuard=0x76543210;
// A1024-byte lane-varying pattern: every load to a ring slot has identical
// bits, so unordered WAW returns cannot make the final oracle ambiguous.
__host__ __device__ uint32_t payload(uint32_t i){return 0xa5a50000u^((i%256)*0x9e3779b9u);}
#include "vgpr-loads.hpp"
__global__ __launch_bounds__(256) void vgpr_ring(const uint32_t* source,U4* out,uint64_t* stamps,uint32_t groups){
  const unsigned lane=threadIdx.x%64,wave=threadIdx.x/64,waves=blockDim.x/64;
  const uint32_t window=kSourceBytes/waves;
  const uintptr_t src=reinterpret_cast<uintptr_t>(source)+wave*window+lane*16;
  U4 ring[kRing];uint64_t b,i,c,wb,wi,wc;
  const uint32_t mask=__builtin_amdgcn_readfirstlane(window-1);
  load_vgpr(uint32_t(src),uint32_t(src>>32),mask,groups,ring,b,i,c,wb,wi,wc);
#pragma unroll
  for(unsigned j=0;j<kRing;++j)out[j*blockDim.x+threadIdx.x]=ring[j];
  if(lane==0){auto s=stamps+wave*6;s[0]=b;s[1]=i;s[2]=c;s[3]=wb;s[4]=wi;s[5]=wc;}
}
template<class T>struct Buffer{T* p=nullptr;explicit Buffer(size_t n){HIP(hipMalloc(&p,n*sizeof(T)));}~Buffer(){if(p)(void)hipFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;};
struct Events{hipEvent_t b,e;Events(){HIP(hipEventCreate(&b));HIP(hipEventCreate(&e));}~Events(){(void)hipEventDestroy(b);(void)hipEventDestroy(e);}};
int main(int argc,char** argv)try{
  unsigned waves=4,groups=0;int rounds=9,warm_ms=200;
  for(int a=1;a<argc;++a){std::string k=argv[a];if(k=="--help"){std::puts("--waves 1|2|4 --groups N --rounds N --warmup-ms N; default groups requests1GiB/CTA; source1MiB; ring32 x4VGPR/lane");return 0;}if(++a==argc)throw std::runtime_error("missing value");std::string v=argv[a];
    if(k=="--waves"){auto n=std::stoull(v);if(n!=1&&n!=2&&n!=4)throw std::runtime_error("waves must be1,2,4");waves=unsigned(n);}
    else if(k=="--groups"){auto n=std::stoull(v);if(!n||n>UINT32_MAX)throw std::runtime_error("groups out of range");groups=unsigned(n);}
    else if(k=="--rounds")rounds=std::stoi(v);else if(k=="--warmup-ms")warm_ms=std::stoi(v);else throw std::runtime_error("unknown option "+k);}
  if(rounds<1||warm_ms<0)throw std::runtime_error("invalid arguments");if(!groups)groups=32768/waves;
  HIP(hipSetDevice(0));int wallrate;HIP(hipDeviceGetAttribute(&wallrate,hipDeviceAttributeWallClockRate,0));hipDeviceProp_t props;HIP(hipGetDeviceProperties(&props,0));char bdf[64];HIP(hipDeviceGetPCIBusId(bdf,sizeof(bdf),0));
  std::fprintf(stderr,"HIP0 BDF=%s O3 gfx90a, grid1 block%u waves%u, CUs=%d L2=%d,source1MiB,ring32 x4VGPR/lane,no LDS; no model/context/clock override, external NUMA placement\n",bdf,waves*64,waves,props.multiProcessorCount,props.l2CacheSize);
  const unsigned threads=waves*64;const size_t words=size_t(threads)*kRing*4;
  Buffer<uint32_t> input(kSourceBytes/4),output(words+32);Buffer<uint64_t> stamps(waves*6);
  std::vector<uint32_t> src(kSourceBytes/4),got(words+32,kGuard);for(size_t j=0;j<src.size();++j)src[j]=payload(j);
  HIP(hipMemcpy(input.p,src.data(),kSourceBytes,hipMemcpyHostToDevice));
  auto launch=[&](unsigned n){vgpr_ring<<<1,threads>>>(input.p,reinterpret_cast<U4*>(output.p+16),stamps.p,n);HIP(hipGetLastError());};
  launch(32/waves);HIP(hipDeviceSynchronize()); // Exactly one source traversal.
  auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(warm_ms);
  while(std::chrono::steady_clock::now()<until){launch(groups);HIP(hipDeviceSynchronize());}
  Events events;const uint64_t loads=uint64_t(groups)*kRing,bytes=loads*waves*1024;
  std::puts("round,waves,threads,ring_slots,source_bytes,groups,loads_per_wave,requested_bytes,event_us,issue_cycles,completion_cycles,issue_us,completion_us,cycles_per_wave_group_load,requested_B_per_issue_cycle,requested_GBps,core_MHz,mismatches,guards,valid,begin0,issue0,done0,wall_begin0,wall_issue0,wall_done0,begin1,issue1,done1,wall_begin1,wall_issue1,wall_done1,begin2,issue2,done2,wall_begin2,wall_issue2,wall_done2,begin3,issue3,done3,wall_begin3,wall_issue3,wall_done3");
  for(int r=0;r<rounds;++r){std::fill(got.begin(),got.end(),kGuard);HIP(hipMemcpy(output.p,got.data(),got.size()*4,hipMemcpyHostToDevice));
    HIP(hipEventRecord(events.b));launch(groups);HIP(hipEventRecord(events.e));HIP(hipEventSynchronize(events.e));float ms;HIP(hipEventElapsedTime(&ms,events.b,events.e));
    std::vector<uint64_t> ts(waves*6);HIP(hipMemcpy(ts.data(),stamps.p,ts.size()*8,hipMemcpyDeviceToHost));HIP(hipMemcpy(got.data(),output.p,got.size()*4,hipMemcpyDeviceToHost));
    int bad=0,guards=0;for(size_t j=0;j<words;++j)bad+=got[j+16]!=payload((j/4%threads%64)*4+j%4);
    for(int j=0;j<16;++j)guards+=(got[j]!=kGuard)+(got[words+16+j]!=kGuard);
    uint64_t first=UINT64_MAX,end=0,done=0,wfirst=UINT64_MAX,wend=0,wdone=0;
    for(unsigned w=0;w<waves;++w){first=std::min(first,ts[w*6]);end=std::max(end,ts[w*6+1]);done=std::max(done,ts[w*6+2]);wfirst=std::min(wfirst,ts[w*6+3]);wend=std::max(wend,ts[w*6+4]);wdone=std::max(wdone,ts[w*6+5]);}
    double cycles=double(end-first),us=double(wend-wfirst)*1000/wallrate,doneus=double(wdone-wfirst)*1000/wallrate;
    std::printf("%d,%u,%u,%d,%zu,%u,%llu,%llu,%.6f,%.0f,%.0f,%.6f,%.6f,%.9f,%.9f,%.6f,%.3f,%d,%d,%d",r,waves,threads,kRing,kSourceBytes,groups,(unsigned long long)loads,(unsigned long long)bytes,ms*1000,cycles,double(done-first),us,doneus,cycles/loads,bytes/cycles,bytes/(us*1000),cycles/us,bad,guards,!bad&&!guards);
    for(unsigned w=0;w<4;++w)for(unsigned j=0;j<6;++j)std::printf(",%llu",(unsigned long long)(w<waves?ts[w*6+j]:0));std::puts("");
    if(bad||guards)throw std::runtime_error("VGPR ring/guard check failed");}
  std::fprintf(stderr,"COMPLETE every final ring DWORD/lane/slot and output guard exact; no periodic waits or consumers\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
