// Query the preserved executable's exact code object, not a recompiled twin.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while(0)
int main(int argc,char** argv) try {
  if(argc!=2) throw std::runtime_error("usage: occupancy original-gfx90a-code-object");
  HIP(hipSetDevice(0));
  hipDeviceProp_t p{}; HIP(hipGetDeviceProperties(&p,0));
  char pci[64]; HIP(hipDeviceGetPCIBusId(pci,sizeof(pci),0));
  hipModule_t module; HIP(hipModuleLoad(&module,argv[1]));
  hipFunction_t function;
  HIP(hipModuleGetFunction(&function,module,"_Z11read_streamILi2ELi16ELb0EEvPKN6VectorIXT_EE4TypeEmPj"));
  int regs,shared,local,max_block;
  HIP(hipFuncGetAttribute(&regs,HIP_FUNC_ATTRIBUTE_NUM_REGS,function));
  HIP(hipFuncGetAttribute(&shared,HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,function));
  HIP(hipFuncGetAttribute(&local,HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,function));
  HIP(hipFuncGetAttribute(&max_block,HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,function));
  std::printf("PCI=%s CUs=%d wave=%d max_threads_per_CU=%d regs_per_CU=%d\n",pci,p.multiProcessorCount,p.warpSize,p.maxThreadsPerMultiProcessor,p.regsPerMultiprocessor);
  std::printf("exact_read_stream_2_16_global regs=%d shared_bytes=%d local_bytes=%d max_block=%d\n",regs,shared,local,max_block);
  std::puts("threads_per_CTA,active_CTAs_per_CU,device_CTA_capacity,waves_per_CU,resident_threads_per_CU");
  for(int threads:{64,128,256,512}) {
    int active; HIP(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,threads,0));
    std::printf("%d,%d,%d,%d,%d\n",threads,active,active*p.multiProcessorCount,active*threads/p.warpSize,active*threads);
  }
  HIP(hipModuleUnload(module));
  return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
