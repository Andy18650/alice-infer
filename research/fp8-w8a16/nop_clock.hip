// Independent s_nop wait-state/core-clock calibration; no MFMA or VALU body.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess) throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e)); } while(0)
template<int Wait> __device__ __forceinline__ void nop() {
  if constexpr(Wait>=16) { asm volatile("s_nop 15"); nop<Wait-16>(); }
  else if constexpr(Wait>0) asm volatile("s_nop %0" : : "n"(Wait-1));
}
template<int Wait> __global__ void nop_clock(uint64_t* result,int steps) {
  uint64_t begin=clock64();
#pragma clang loop unroll(disable)
  for(int i=0;i<steps;i+=32) {
#pragma unroll
    for(int j=0;j<32;++j) nop<Wait>();
  }
  uint64_t end=clock64();
  if(threadIdx.x%64==0) result[(blockIdx.x*blockDim.x+threadIdx.x)/64]=end-begin;
}
int main() try {
  HIP(hipSetDevice(0)); hipDeviceProp_t prop{}; HIP(hipGetDeviceProperties(&prop,0));
  int steps=16384;
  uint64_t* p; HIP(hipMalloc(&p,prop.multiProcessorCount*4*sizeof(uint64_t)));
  std::puts("geometry,wait_units,median_core_cycles_per_delay,min,max");
  for(int geometry=0;geometry<2;++geometry) {
    int blocks=geometry?prop.multiProcessorCount:1,threads=geometry?256:64,waves=blocks*threads/64;
    for(int wait: {1,2,4,6,7,8,10,12,16,24,32,64}) {
      std::vector<double> samples;
      for(int round=0;round<9;++round) {
#define RUN(N) case N: nop_clock<N><<<blocks,threads>>>(p,steps); break
        switch(wait) { RUN(1);RUN(2);RUN(4);RUN(6);RUN(7);RUN(8);RUN(10);RUN(12);RUN(16);RUN(24);RUN(32);RUN(64); }
#undef RUN
        HIP(hipGetLastError());
        std::vector<uint64_t> clocks(waves);
        HIP(hipMemcpy(clocks.data(),p,waves*sizeof(uint64_t),hipMemcpyDeviceToHost));
        double sum=0; for(auto n:clocks) sum+=double(n);
        samples.push_back(sum/(waves*double(steps)));
      }
      std::sort(samples.begin(),samples.end());
      std::printf("%s,%d,%.9f,%.9f,%.9f\n",geometry?"104x256":"1x64",wait,samples[4],samples[0],samples[8]);
    }
  }
  HIP(hipFree(p));
} catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what());return 1; }
