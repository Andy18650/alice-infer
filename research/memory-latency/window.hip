// Independent burst-window sweep; no per-load timestamps in measured kernels.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#define HIP(x) do {auto e=(x);if(e!=hipSuccess)throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e));}while(0)
using U4=uint32_t __attribute__((ext_vector_type(4))); // asm VGPR tuple
#include "window-bursts.hpp"
constexpr uint32_t Threads=256,A=1664525u,C=1013904223u;
__global__ void initialize_window(uint32_t* data,size_t words) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x)data[i]=uint32_t(i)*A+C;
}
template<int W,int D>
__device__ __forceinline__ void window_body(const uint32_t* input,uint32_t* output,uint32_t iterations) {
  uint32_t address=(blockIdx.x*blockDim.x+threadIdx.x)*W*4;
  const uint32_t stride=gridDim.x*blockDim.x*W*4;
  using V=typename Batch<W,D>::V;V sum{};
#pragma clang loop unroll(disable)
  for(uint32_t i=0;i<iterations;++i) {
    auto batch=issue_batch<W,D>(input,address,stride);
#pragma unroll
    for(int j=0;j<D;++j)sum+=batch.value[j];
  }
  uint32_t checksum;
  if constexpr(W==4)checksum=sum[0]+sum[1]+sum[2]+sum[3];else checksum=sum;
  output[blockIdx.x*blockDim.x+threadIdx.x]=checksum;
}
template<int W,int D>
__global__ __launch_bounds__(Threads,1) void window_entry(const uint32_t* input,uint32_t* output,uint32_t iterations) {window_body<W,D>(input,output,iterations);}
__global__ __launch_bounds__(Threads,1) void warm_reader(const uint32_t* input,uint32_t* output,uint32_t iterations) {window_body<4,4>(input,output,iterations);}
struct Candidate {int width,depth;const void* fn;hipFuncAttributes attributes;int capacity;};
template<int W,int D>void add(std::vector<Candidate>& cs) {
  Candidate c{W,D,reinterpret_cast<const void*>(window_entry<W,D>),{},0};
  HIP(hipFuncGetAttributes(&c.attributes,c.fn));HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(&c.capacity,c.fn,Threads,0));
  if(c.attributes.localSizeBytes||c.attributes.sharedSizeBytes)throw std::runtime_error("unexpected private/LDS storage");
  cs.push_back(c);
}
uint32_t expected(uint32_t global_thread,uint32_t width,uint32_t grid,size_t bytes) {
  const uint32_t n=bytes/(uint64_t(grid)*Threads*width*4);
  const uint32_t first=global_thread*width,step=grid*Threads*width;
  const uint32_t triangular=uint64_t(n)*(n-1)/2;
  return n*(width*C+A*(width*first+width*(width-1)/2))+A*width*step*triangular; // modulo 2^32
}
int main(int argc,char**argv)try {
  uint32_t mib=1040,rounds=3,repeats=4,seed=12345;
  for(int i=1;i<argc;++i){std::string key=argv[i];if(i+1>=argc)throw std::runtime_error("missing argument");uint32_t v=std::stoul(argv[++i]);if(key=="--mib")mib=v;else if(key=="--rounds")rounds=v;else if(key=="--repeats")repeats=v;else if(key=="--seed")seed=v;else throw std::runtime_error("unknown argument "+key);}
  const size_t bytes=size_t(mib)<<20;
  if(!mib||mib%52||bytes+416*Threads*16>=UINT32_MAX||!rounds||!repeats)throw std::runtime_error("positive counts; MiB must be a multiple of 52 and fit 32-bit byte offsets");
  HIP(hipSetDevice(0));hipDeviceProp_t prop{};HIP(hipGetDeviceProperties(&prop,0));
  if(std::string(prop.gcnArchName).find("gfx90a")==std::string::npos||prop.warpSize!=64||prop.multiProcessorCount!=104)throw std::runtime_error("requires 104-CU gfx90a wave64");
  char pci[64];HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
  std::fprintf(stderr,"device=0 pci=%s arch=%s CUs=104 NUMA=external model=none context=none automatic_clocks bytes=%zu seed=%u protocol=eager_%u_operations warmup=2s_reference+3_per_case\n",pci,prop.gcnArchName,bytes,seed,repeats);
  uint32_t *input,*output;hipStream_t stream;hipEvent_t begin,end;
  HIP(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));HIP(hipEventCreate(&begin));HIP(hipEventCreate(&end));
  HIP(hipMalloc(&input,bytes));HIP(hipMalloc(&output,416*Threads*4));
  initialize_window<<<832,256,0,stream>>>(input,bytes/4);HIP(hipGetLastError());HIP(hipStreamSynchronize(stream));
  std::vector<Candidate> cs;
#define ADD_W(W) add<W,1>(cs);add<W,2>(cs);add<W,4>(cs);add<W,8>(cs);add<W,16>(cs)
  ADD_W(1);ADD_W(4);
#undef ADD_W
  const auto warm_start=std::chrono::steady_clock::now();
  do {warm_reader<<<104,Threads,0,stream>>>(input,output,bytes/(104*Threads*4*4*4));HIP(hipStreamSynchronize(stream));}while(std::chrono::steady_clock::now()-warm_start<std::chrono::seconds(2));
  struct Job {size_t candidate;uint32_t grid;};std::vector<Job> jobs;
  for(size_t i=0;i<cs.size();++i)for(uint32_t grid:{4,8,16,32,64,104,208,416})jobs.push_back({i,grid});
  std::mt19937 rng(seed);std::shuffle(jobs.begin(),jobs.end(),rng);
  std::puts("width_dwords,depth,grid,threads,waves,payload_per_wave,nominal_window_bytes,resident_capacity_ctas,resident_window_upper_bytes,registers,private_bytes,lds_bytes,bytes,round,repeats,gpu_us,requested_GBps");
  std::vector<uint32_t> host(416*Threads);
  for(const auto&j:jobs){const auto&c=cs[j.candidate];
    uint32_t iterations=bytes/(uint64_t(j.grid)*Threads*c.width*c.depth*4);
    if(uint64_t(iterations)*j.grid*Threads*c.width*c.depth*4!=bytes)throw std::runtime_error("nonuniform tail");
    auto launch=[&]{void*args[]={&input,&output,&iterations};HIP(hipLaunchKernel(c.fn,dim3(j.grid),dim3(Threads),args,0,stream));};
    HIP(hipMemsetAsync(output,0xa5,j.grid*Threads*4,stream));for(int i=0;i<3;++i)launch();HIP(hipStreamSynchronize(stream));
    for(uint32_t round=0;round<rounds;++round){
      HIP(hipEventRecord(begin,stream));for(uint32_t r=0;r<repeats;++r)launch();HIP(hipEventRecord(end,stream));HIP(hipEventSynchronize(end));
      float ms;HIP(hipEventElapsedTime(&ms,begin,end));const double us=double(ms)*1000/repeats;
      const size_t window=size_t(j.grid)*Threads*c.width*c.depth*4;
      const int resident=std::min<int>(j.grid,104*c.capacity);
      std::printf("%d,%d,%u,%u,%u,%u,%zu,%d,%zu,%d,%zu,%zu,%zu,%u,%u,%.6f,%.6f\n",c.width,c.depth,j.grid,Threads,j.grid*4,64*c.width*4,window,resident,size_t(resident)*Threads*c.width*c.depth*4,c.attributes.numRegs,c.attributes.localSizeBytes,c.attributes.sharedSizeBytes,bytes,round,repeats,us,bytes/us/1000);std::fflush(stdout);
    }
    HIP(hipMemcpy(host.data(),output,j.grid*Threads*4,hipMemcpyDeviceToHost));
    for(uint32_t t=0;t<j.grid*Threads;++t)if(host[t]!=expected(t,c.width,j.grid,bytes))throw std::runtime_error("checksum mismatch thread="+std::to_string(t));
  }
  HIP(hipFree(input));HIP(hipFree(output));HIP(hipEventDestroy(begin));HIP(hipEventDestroy(end));HIP(hipStreamDestroy(stream));return 0;
}catch(const std::exception&e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
