// Matched64-CTA x4-wave disjoint versus8x8 A-row/B-column source sharing.
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
__global__ __launch_bounds__(256) void vgpr_ring(const uint32_t* source,U4* out,uint64_t* stamps,uint64_t* addresses,uint32_t groups,int outer){
  const unsigned lane=threadIdx.x%64,wave=threadIdx.x/64,waves=blockDim.x/64;
  const uint32_t window=kSourceBytes/waves;
  const size_t panel=wave<2 ? size_t(blockIdx.x/8)*524288+(wave%2)*262144
                            : 4194304+size_t(blockIdx.x%8)*524288+(wave%2)*262144;
  const size_t offset=outer?panel:size_t(blockIdx.x)*kSourceBytes+wave*window;
  const uintptr_t src=reinterpret_cast<uintptr_t>(source)+offset+lane*16;
  U4 ring[kRing];uint64_t b,i,c,wb,wi,wc;
  const uint32_t mask=__builtin_amdgcn_readfirstlane(window-1);
  load_vgpr(uint32_t(src),uint32_t(src>>32),mask,groups,ring,b,i,c,wb,wi,wc);
#pragma unroll
  for(unsigned j=0;j<kRing;++j)out[(size_t(blockIdx.x)*kRing+j)*blockDim.x+threadIdx.x]=ring[j];
  if(lane==0){addresses[size_t(blockIdx.x)*waves+wave]=src;auto s=stamps+(size_t(blockIdx.x)*waves+wave)*6;s[0]=b;s[1]=i;s[2]=c;s[3]=wb;s[4]=wi;s[5]=wc;}
}
template<class T>struct Buffer{T* p=nullptr;explicit Buffer(size_t n){HIP(hipMalloc(&p,n*sizeof(T)));}~Buffer(){if(p)(void)hipFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;};
struct Events{hipEvent_t b,e;Events(){HIP(hipEventCreate(&b));HIP(hipEventCreate(&e));}~Events(){(void)hipEventDestroy(b);(void)hipEventDestroy(e);}};
int main(int argc,char** argv)try{
  constexpr unsigned waves=4,ctas=64;unsigned groups=0;int rounds=9,warm_ms=200;std::string layout="outer";
  for(int a=1;a<argc;++a){std::string k=argv[a];if(k=="--help"){std::puts("--layout disjoint|outer --groups N --rounds N --warmup-ms N; fixed64 CTAs x4 waves,256MiB requested/CTA");return 0;}if(++a==argc)throw std::runtime_error("missing value");std::string v=argv[a];
    if(k=="--layout")layout=v;
    else if(k=="--groups"){auto n=std::stoull(v);if(!n||n>UINT32_MAX)throw std::runtime_error("groups out of range");groups=unsigned(n);}
    else if(k=="--rounds")rounds=std::stoi(v);else if(k=="--warmup-ms")warm_ms=std::stoi(v);else throw std::runtime_error("unknown option "+k);}
  if(rounds<1||warm_ms<0||(layout!="disjoint"&&layout!="outer"))throw std::runtime_error("invalid arguments");if(!groups)groups=8192/waves;
  const size_t source_bytes=layout=="outer"?8388608:size_t(ctas)*kSourceBytes;
  constexpr size_t allocated_source_bytes=size_t(ctas)*kSourceBytes;
  HIP(hipSetDevice(0));int wallrate;HIP(hipDeviceGetAttribute(&wallrate,hipDeviceAttributeWallClockRate,0));hipDeviceProp_t props;HIP(hipGetDeviceProperties(&props,0));char bdf[64];HIP(hipDeviceGetPCIBusId(bdf,sizeof(bdf),0));
  const unsigned threads=waves*64;int resident=0;HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,vgpr_ring,threads,0));
  std::fprintf(stderr,"HIP0 BDF=%s O3 gfx90a, CTAs=%u block%u waves%u, CUs=%d L2=%d,source1MiB/CTA,total_source=%zu,ring32 x4VGPR/lane,no LDS,resident_CTA_limit_per_CU=%d; no model/context/clock override, external NUMA placement\n",bdf,ctas,threads,waves,props.multiProcessorCount,props.l2CacheSize,source_bytes,resident);
  std::fprintf(stderr,"layout=%s; allocated_source_bytes=%zu in both cases; reachable outer A[8] x512KiB then B[8] x512KiB, waves0/1 A-row,waves2/3 B-col; source bases recorded outside timing\n",layout.c_str(),allocated_source_bytes);
  const size_t words=size_t(ctas)*threads*kRing*4;
  Buffer<uint32_t> input(allocated_source_bytes/4),output(words+32);Buffer<uint64_t> stamps(size_t(ctas)*waves*6),addresses(ctas*waves);
  std::vector<uint32_t> src(allocated_source_bytes/4),got(words+32,kGuard);for(size_t j=0;j<src.size();++j)src[j]=payload(j);
  HIP(hipMemcpy(input.p,src.data(),src.size()*4,hipMemcpyHostToDevice));
  auto launch=[&](unsigned n){vgpr_ring<<<ctas,threads>>>(input.p,reinterpret_cast<U4*>(output.p+16),stamps.p,addresses.p,n,layout=="outer");HIP(hipGetLastError());};
  launch(32/waves);HIP(hipDeviceSynchronize()); // Exactly one source traversal.
  auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(warm_ms);
  while(std::chrono::steady_clock::now()<until){launch(groups);HIP(hipDeviceSynchronize());}
  Events events;const uint64_t loads=uint64_t(groups)*kRing,bytes=loads*waves*1024;
  std::puts("layout,round,cta,ctas,waves,threads,ring_slots,source_bytes_per_cta,groups,loads_per_wave,requested_bytes,event_us,issue_cycles,completion_cycles,issue_us,completion_us,cycles_per_wave_group_load,requested_B_per_issue_cycle,requested_GBps,core_MHz,mismatches,guards,valid,begin0,issue0,done0,wall_begin0,wall_issue0,wall_done0,begin1,issue1,done1,wall_begin1,wall_issue1,wall_done1,begin2,issue2,done2,wall_begin2,wall_issue2,wall_done2,begin3,issue3,done3,wall_begin3,wall_issue3,wall_done3,source_offset0,source_offset1,source_offset2,source_offset3");
  for(int r=0;r<rounds;++r){std::fill(got.begin(),got.end(),kGuard);HIP(hipMemcpy(output.p,got.data(),got.size()*4,hipMemcpyHostToDevice));
    HIP(hipEventRecord(events.b));launch(groups);HIP(hipEventRecord(events.e));HIP(hipEventSynchronize(events.e));float ms;HIP(hipEventElapsedTime(&ms,events.b,events.e));
    std::vector<uint64_t> ts(size_t(ctas)*waves*6);HIP(hipMemcpy(ts.data(),stamps.p,ts.size()*8,hipMemcpyDeviceToHost));HIP(hipMemcpy(got.data(),output.p,got.size()*4,hipMemcpyDeviceToHost));
    std::vector<uint64_t> actual(ctas*waves);HIP(hipMemcpy(actual.data(),addresses.p,actual.size()*8,hipMemcpyDeviceToHost));
    for(unsigned cta=0;cta<ctas;++cta)for(unsigned w=0;w<waves;++w){
      const size_t panel=w<2?size_t(cta/8)*524288+(w%2)*262144:4194304+size_t(cta%8)*524288+(w%2)*262144;
      const size_t expected=layout=="outer"?panel:size_t(cta)*kSourceBytes+w*262144;
      if(expected+262144>source_bytes||actual[cta*waves+w]!=reinterpret_cast<uintptr_t>(input.p)+expected)throw std::runtime_error("GPU source-base map mismatch");}
    int bad=0,guards=0;for(size_t j=0;j<words;++j)bad+=got[j+16]!=payload((j/4%threads%64)*4+j%4);
    for(int j=0;j<16;++j)guards+=(got[j]!=kGuard)+(got[words+16+j]!=kGuard);
    for(unsigned cta=0;cta<ctas;++cta){auto t=ts.data()+size_t(cta)*waves*6;
      uint64_t first=UINT64_MAX,end=0,done=0,wfirst=UINT64_MAX,wend=0,wdone=0;
      for(unsigned w=0;w<waves;++w){first=std::min(first,t[w*6]);end=std::max(end,t[w*6+1]);done=std::max(done,t[w*6+2]);wfirst=std::min(wfirst,t[w*6+3]);wend=std::max(wend,t[w*6+4]);wdone=std::max(wdone,t[w*6+5]);}
      double cycles=double(end-first),us=double(wend-wfirst)*1000/wallrate,doneus=double(wdone-wfirst)*1000/wallrate;
      std::printf("%s,%d,%u,%u,%u,%u,%d,%zu,%u,%llu,%llu,%.6f,%.0f,%.0f,%.6f,%.6f,%.9f,%.9f,%.6f,%.3f,%d,%d,%d",layout.c_str(),r,cta,ctas,waves,threads,kRing,kSourceBytes,groups,(unsigned long long)loads,(unsigned long long)bytes,ms*1000,cycles,double(done-first),us,doneus,cycles/loads,bytes/cycles,bytes/(us*1000),cycles/us,bad,guards,!bad&&!guards);
      for(unsigned w=0;w<4;++w)for(unsigned j=0;j<6;++j)std::printf(",%llu",(unsigned long long)t[w*6+j]);
      for(unsigned w=0;w<waves;++w)std::printf(",%llu",(unsigned long long)(actual[cta*waves+w]-reinterpret_cast<uintptr_t>(input.p)));std::puts("");
    }
    if(bad||guards)throw std::runtime_error("VGPR ring/guard check failed");}
  std::fprintf(stderr,"COMPLETE every final ring DWORD/lane/slot and output guard exact; GPU source-base map exact; no periodic waits or consumers\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
