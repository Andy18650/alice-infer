#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#define HIP(x) do{auto status_=(x);if(status_!=hipSuccess)throw std::runtime_error(hipGetErrorString(status_));}while(0)
using F16=float __attribute__((ext_vector_type(16)));
template<bool Same>__global__ __launch_bounds__(256) void banks(float* out,uint64_t* ts,int groups){
  F16 c0={},c1={},c2={},c3={};uint64_t b,e;
  asm volatile("v_mov_b32 v0, 0x2c002c00\n\tv_mov_b32 v1, v0\n\tv_mov_b32 v2, v0\n\tv_mov_b32 v3, v0\n\tv_mov_b32 v4, v0\n\tv_mov_b32 v5, v0\n\ts_memtime %0\n\ts_waitcnt lgkmcnt(0)" : "=&s"(b),"+a"(c0),"+a"(c1),"+a"(c2),"+a"(c3)::"v0","v1","v2","v3","v4","v5","memory");
  if constexpr(Same)asm volatile(
    ".Lbank_same_%=:\n\tv_mfma_f32_32x32x8f16 %0, v[0:1], v[4:5], %0\n\tv_mfma_f32_32x32x8f16 %1, v[0:1], v[4:5], %1\n\tv_mfma_f32_32x32x8f16 %2, v[0:1], v[4:5], %2\n\tv_mfma_f32_32x32x8f16 %3, v[0:1], v[4:5], %3\n\ts_sub_u32 %4, %4, 1\n\ts_cmp_lg_u32 %4, 0\n\ts_cbranch_scc1 .Lbank_same_%="
    :"+a"(c0),"+a"(c1),"+a"(c2),"+a"(c3),"+&s"(groups)::"v0","v1","v2","v3","v4","v5","scc","memory");
  else asm volatile(
    ".Lbank_diff_%=:\n\tv_mfma_f32_32x32x8f16 %0, v[0:1], v[2:3], %0\n\tv_mfma_f32_32x32x8f16 %1, v[0:1], v[2:3], %1\n\tv_mfma_f32_32x32x8f16 %2, v[0:1], v[2:3], %2\n\tv_mfma_f32_32x32x8f16 %3, v[0:1], v[2:3], %3\n\ts_sub_u32 %4, %4, 1\n\ts_cmp_lg_u32 %4, 0\n\ts_cbranch_scc1 .Lbank_diff_%="
    :"+a"(c0),"+a"(c1),"+a"(c2),"+a"(c3),"+&s"(groups)::"v0","v1","v2","v3","v4","v5","scc","memory");
  asm volatile("s_nop 15\n\ts_nop 15\n\ts_nop 15\n\ts_nop 15\n\ts_memtime %0\n\ts_waitcnt lgkmcnt(0)":"=&s"(e),"+a"(c0),"+a"(c1),"+a"(c2),"+a"(c3)::"memory");
  if(threadIdx.x%64==0){ts[threadIdx.x/64*2]=b;ts[threadIdx.x/64*2+1]=e;}
#pragma unroll
  for(int j=0;j<16;++j){out[threadIdx.x*64+j]=c0[j];out[threadIdx.x*64+16+j]=c1[j];out[threadIdx.x*64+32+j]=c2[j];out[threadIdx.x*64+48+j]=c3[j];}
}
int main()try{HIP(hipSetDevice(0));float* out;uint64_t* ts;HIP(hipMalloc(&out,256*64*4));HIP(hipMalloc(&ts,64));
  std::puts("same_pair_position,round,cycles_per_four_wave_mfma,valid");
  for(int same:{0,1})for(int r=0;r<4;++r){if(same)banks<true><<<1,256>>>(out,ts,65536);else banks<false><<<1,256>>>(out,ts,65536);HIP(hipGetLastError());HIP(hipDeviceSynchronize());uint64_t t[8];HIP(hipMemcpy(t,ts,sizeof(t),hipMemcpyDeviceToHost));std::vector<float> x(256*64);HIP(hipMemcpy(x.data(),out,x.size()*4,hipMemcpyDeviceToHost));for(float v:x)if(v!=2048)throw std::runtime_error("bank control mismatch");uint64_t b=UINT64_MAX,e=0;for(int w=0;w<4;++w){b=std::min(b,t[w*2]);e=std::max(e,t[w*2+1]);}std::printf("%d,%d,%.9f,1\n",same,r,double(e-b)/262144);}
  (void)hipFree(out);(void)hipFree(ts);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
