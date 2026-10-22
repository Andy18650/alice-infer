// Watermark sweep: identical streaming source/LDS geometry,1 or8 CTAs.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do{auto status_=(x);if(status_!=hipSuccess)throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(status_));}while(0)
constexpr size_t kSourceBytes=1048576,kLdsWords=16384;
constexpr uint32_t kGuard=0x76543210;
__host__ __device__ uint32_t payload(uint32_t i){return 0xa5a50000u^((i%64)*0x9e3779b9u);}
#include "loads.hpp"
__global__ __launch_bounds__(256) void staging(const uint32_t* source,uint32_t* out,uint64_t* stamps,uint32_t groups){
  extern __shared__ __attribute__((aligned(16))) uint32_t smem[];
  const unsigned lane=threadIdx.x%64,wave=threadIdx.x/64;
  for(unsigned j=threadIdx.x;j<kLdsWords;j+=256)smem[j]=0xdeadbeef;
  __syncthreads();
  const uintptr_t src=reinterpret_cast<uintptr_t>(source)+size_t(blockIdx.x)*kSourceBytes+wave*(kSourceBytes/4)+lane*4;
  const uint32_t dst=__builtin_amdgcn_readfirstlane(uint32_t(reinterpret_cast<uintptr_t>(smem))+wave*16384);
  uint64_t b,i,c,wb,wi,wc;load_loop(uint32_t(src),uint32_t(src>>32),dst,groups,b,i,c,wb,wi,wc);
  __syncthreads();
  for(unsigned j=threadIdx.x;j<kLdsWords;j+=256)out[size_t(blockIdx.x)*kLdsWords+j]=smem[j];
  if(lane==0){auto s=stamps+(size_t(blockIdx.x)*4+wave)*6;s[0]=b;s[1]=i;s[2]=c;s[3]=wb;s[4]=wi;s[5]=wc;}
}
template<class T>struct Buffer{T* p=nullptr;explicit Buffer(size_t n){HIP(hipMalloc(&p,n*sizeof(T)));}~Buffer(){if(p)(void)hipFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;};
struct Events{hipEvent_t b,e;Events(){HIP(hipEventCreate(&b));HIP(hipEventCreate(&e));}~Events(){(void)hipEventDestroy(b);(void)hipEventDestroy(e);}};
int main(int argc,char** argv)try{
  unsigned ctas=1,groups=16384;int rounds=9,warm_ms=200;
  for(int a=1;a<argc;++a){std::string k=argv[a];if(k=="--help"){std::puts("--ctas 1|8 --groups N --rounds N --warmup-ms N; watermark encoded in native variant, private LDS16KiB/wave");return 0;}if(++a==argc)throw std::runtime_error("missing value");std::string v=argv[a];
    if(k=="--ctas"){auto n=std::stoull(v);if(n!=1&&n!=8)throw std::runtime_error("ctas must be1 or8");ctas=unsigned(n);}
    else if(k=="--groups"){auto n=std::stoull(v);if(!n||n>UINT32_MAX)throw std::runtime_error("groups out of range");groups=unsigned(n);}
    else if(k=="--rounds")rounds=std::stoi(v);else if(k=="--warmup-ms")warm_ms=std::stoi(v);else throw std::runtime_error("unknown option "+k);}
  if(rounds<1||warm_ms<0)throw std::runtime_error("invalid arguments");
  HIP(hipSetDevice(0));int wallrate;HIP(hipDeviceGetAttribute(&wallrate,hipDeviceAttributeWallClockRate,0));hipDeviceProp_t props;HIP(hipGetDeviceProperties(&props,0));char bdf[64];HIP(hipDeviceGetPCIBusId(bdf,sizeof(bdf),0));
  std::fprintf(stderr,"HIP0 BDF=%s O3 gfx90a, CTAs=%u block256 CUs=%d L2=%d,source1MiB/CTA,private LDS16KiB/wave64KiB/CTA; watermark from verified native manifest; no model/context/clock override, external NUMA placement\n",bdf,ctas,props.multiProcessorCount,props.l2CacheSize);
  Buffer<uint32_t> input(size_t(ctas)*kSourceBytes/4),output(size_t(ctas)*kLdsWords+32);Buffer<uint64_t> stamps(ctas*24);
  std::vector<uint32_t> src(size_t(ctas)*kSourceBytes/4),got(size_t(ctas)*kLdsWords+32,kGuard);for(size_t j=0;j<src.size();++j)src[j]=payload(j);
  HIP(hipMemcpy(input.p,src.data(),src.size()*4,hipMemcpyHostToDevice));
  auto launch=[&](unsigned n){staging<<<ctas,256,65536>>>(input.p,output.p+16,stamps.p,n);HIP(hipGetLastError());};
  launch(16);HIP(hipDeviceSynchronize());auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(warm_ms);
  while(std::chrono::steady_clock::now()<until){launch(groups);HIP(hipDeviceSynchronize());}
  Events events;const uint64_t loads=uint64_t(groups)*64,bytes=loads*1024;
  std::puts("round,cta,ctas,source_bytes_per_cta,groups,loads_per_wave,requested_bytes_per_cta,event_us,issue_cycles,completion_cycles,issue_us,completion_us,cycles_per_four_wave_load,requested_B_per_issue_cycle,core_MHz,mismatches,guards,valid,begin0,issue0,done0,wall_begin0,wall_issue0,wall_done0,begin1,issue1,done1,wall_begin1,wall_issue1,wall_done1,begin2,issue2,done2,wall_begin2,wall_issue2,wall_done2,begin3,issue3,done3,wall_begin3,wall_issue3,wall_done3");
  for(int r=0;r<rounds;++r){std::fill(got.begin(),got.end(),kGuard);HIP(hipMemcpy(output.p,got.data(),got.size()*4,hipMemcpyHostToDevice));
    HIP(hipEventRecord(events.b));launch(groups);HIP(hipEventRecord(events.e));HIP(hipEventSynchronize(events.e));float ms;HIP(hipEventElapsedTime(&ms,events.b,events.e));
    std::vector<uint64_t> ts(ctas*24);HIP(hipMemcpy(ts.data(),stamps.p,ts.size()*8,hipMemcpyDeviceToHost));HIP(hipMemcpy(got.data(),output.p,got.size()*4,hipMemcpyDeviceToHost));
    int bad=0,guards=0;for(size_t j=0;j<size_t(ctas)*kLdsWords;++j)bad+=got[j+16]!=payload(j);
    for(int j=0;j<16;++j)guards+=(got[j]!=kGuard)+(got[size_t(ctas)*kLdsWords+16+j]!=kGuard);
    for(unsigned cta=0;cta<ctas;++cta){auto t=ts.data()+cta*24;uint64_t first=UINT64_MAX,end=0,done=0,wfirst=UINT64_MAX,wend=0,wdone=0;
      for(unsigned w=0;w<4;++w){first=std::min(first,t[w*6]);end=std::max(end,t[w*6+1]);done=std::max(done,t[w*6+2]);wfirst=std::min(wfirst,t[w*6+3]);wend=std::max(wend,t[w*6+4]);wdone=std::max(wdone,t[w*6+5]);}
      double cycles=double(end-first),us=double(wend-wfirst)*1000/wallrate,doneus=double(wdone-wfirst)*1000/wallrate;
      std::printf("%d,%u,%u,%zu,%u,%llu,%llu,%.6f,%.0f,%.0f,%.6f,%.6f,%.9f,%.9f,%.3f,%d,%d,%d",r,cta,ctas,kSourceBytes,groups,(unsigned long long)loads,(unsigned long long)bytes,ms*1000,cycles,double(done-first),us,doneus,cycles/loads,bytes/cycles,cycles/us,bad,guards,!bad&&!guards);
      for(unsigned j=0;j<24;++j)std::printf(",%llu",(unsigned long long)t[j]);std::puts("");}
    if(bad||guards)throw std::runtime_error("LDS/guard check failed");}
  std::fprintf(stderr,"COMPLETE all per-CTA LDS words and output guards exact\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
