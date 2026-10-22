// One coalesced global->LDS load + VM0,64 clock-inclusive intervals/wave.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do {auto status_=(x);if(status_!=hipSuccess)throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(status_));}while(0)
constexpr size_t kSourceBytes=1024*1024,kLdsWords=16384;
constexpr uint32_t kGuard=0x76543210,kSentinel=0xdeadbeef;
__host__ __device__ uint32_t payload(uint32_t i){return 0xa5a50000u^((i%64)*0x9e3779b9u);}
#include "latency-records.hpp"
__global__ void warm_source(const volatile uint32_t* src,uint32_t* sink){
  uint32_t acc=0;for(unsigned i=threadIdx.x;i<kSourceBytes/4;i+=blockDim.x)acc^=src[i];sink[threadIdx.x]=acc;
}
__global__ __launch_bounds__(256) void latency64(const uint32_t* source,uint32_t* records,uint32_t* check,unsigned offset,int private_lds){
  extern __shared__ __attribute__((aligned(16))) uint32_t smem[];
  const unsigned wave=threadIdx.x/64,lane=threadIdx.x%64;
  for(unsigned j=threadIdx.x;j<kLdsWords;j+=256)smem[j]=kSentinel;
  __syncthreads();
  const uintptr_t p=reinterpret_cast<uintptr_t>(source)+wave*(kSourceBytes/4)+offset+lane*4;
  const uint32_t dst=__builtin_amdgcn_readfirstlane(uint32_t(reinterpret_cast<uintptr_t>(smem))+(private_lds?wave*16384:0));
  uint32_t samples[kSamples];
  measure(uint32_t(p),uint32_t(p>>32),dst,samples);
  // No wave may issue record/check stores while another is still timing.
  __syncthreads();
  if(lane==0){
#pragma unroll
    for(unsigned j=0;j<kSamples;++j)records[wave*kSamples+j]=samples[j];
  }
  for(unsigned j=threadIdx.x;j<kLdsWords;j+=256)check[j]=smem[j];
}
template<class T> struct Buffer{T* p=nullptr;explicit Buffer(size_t n){HIP(hipMalloc(&p,n*sizeof(T)));}~Buffer(){if(p)(void)hipFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;};
int main(int argc,char** argv) try {
  std::string mode="private";unsigned batches=64;int warm_ms=200;
  for(int a=1;a<argc;++a){std::string k=argv[a];if(k=="--help"){std::puts("--mode shared|private --batches N --warmup-ms N;64 intervals/wave/batch, one CTA,1MiB source");return 0;}if(++a==argc)throw std::runtime_error("missing value");std::string v=argv[a];
    if(k=="--mode")mode=v;else if(k=="--batches"){auto n=std::stoull(v);if(!n||n>65536)throw std::runtime_error("batches out of range");batches=unsigned(n);}else if(k=="--warmup-ms")warm_ms=std::stoi(v);else throw std::runtime_error("unknown option "+k);}
  if((mode!="shared"&&mode!="private")||warm_ms<0)throw std::runtime_error("invalid arguments");
  HIP(hipSetDevice(0));char bdf[64];HIP(hipDeviceGetPCIBusId(bdf,sizeof(bdf),0));
  std::fprintf(stderr,"HIP0 BDF=%s native O3 gfx90a, one CTA/256 threads,1MiB source,64KiB allocated LDS, mode=%s, %u batches x64 samples/wave; timing overhead included, no model/context/clock overrides\n",bdf,mode.c_str(),batches);
  const size_t count=size_t(batches)*4*kSamples;
  Buffer<uint32_t> input(kSourceBytes/4),records(count+32),check(kLdsWords+32),warm(256);
  std::vector<uint32_t> host(kSourceBytes/4);for(size_t i=0;i<host.size();++i)host[i]=payload(i);
  HIP(hipMemcpy(input.p,host.data(),kSourceBytes,hipMemcpyHostToDevice));
  std::vector<uint32_t> check_host(kLdsWords+32,kGuard),raw(count+32,kGuard);
  HIP(hipMemcpy(check.p,check_host.data(),check_host.size()*4,hipMemcpyHostToDevice));
  warm_source<<<1,256>>>(input.p,warm.p);HIP(hipGetLastError());HIP(hipDeviceSynchronize());
  auto launch=[&](unsigned b,uint32_t* r){latency64<<<1,256,65536>>>(input.p,r,check.p+16,(b%16)*16384,mode=="private");HIP(hipGetLastError());};
  auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(warm_ms);unsigned w=0;
  while(std::chrono::steady_clock::now()<until){launch(w++,records.p+16);HIP(hipDeviceSynchronize());}
  HIP(hipMemcpy(records.p,raw.data(),raw.size()*4,hipMemcpyHostToDevice));
  for(unsigned b=0;b<batches;++b)launch(b,records.p+16+size_t(b)*4*kSamples);
  HIP(hipDeviceSynchronize());
  HIP(hipMemcpy(raw.data(),records.p,raw.size()*4,hipMemcpyDeviceToHost));
  HIP(hipMemcpy(check_host.data(),check.p,check_host.size()*4,hipMemcpyDeviceToHost));
  int bad=0,guards=0;
  for(size_t i=0;i<kLdsWords;++i)bad+=check_host[i+16]!=(mode=="private"||i<4096?payload(i):kSentinel);
  for(int i=0;i<16;++i){guards+=(raw[i]!=kGuard)+(raw[count+16+i]!=kGuard);guards+=(check_host[i]!=kGuard)+(check_host[kLdsWords+16+i]!=kGuard);}
  for(size_t i=16;i<count+16;++i)if(!raw[i]||raw[i]==kGuard||raw[i]>=0x80000000u)throw std::runtime_error("invalid cycle sample");
  if(bad||guards)throw std::runtime_error("LDS/sample/check guard mismatch");
  std::puts("mode,batch,source_offset,wave,sample,cycles");
  for(unsigned b=0;b<batches;++b)for(unsigned wave=0;wave<4;++wave)for(unsigned j=0;j<kSamples;++j)
    std::printf("%s,%u,%u,%u,%u,%u\n",mode.c_str(),b,(b%16)*16384,wave,j,raw[16+(size_t(b)*4+wave)*kSamples+j]);
  std::fprintf(stderr,"COMPLETE %zu raw intervals retained; final full LDS/check guards and all sample guards pass; no per-sample global stores\n",count);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
