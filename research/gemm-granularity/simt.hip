// Diagnostic BF16-input SIMT GEMV consuming exactly the same fixed tile blobs
// as CK. This does not change the earlier full-FP32-activation GEMV study.
#include "api.hpp"
#include <stdexcept>
#include <cmath>
namespace granularity {
namespace {
template<bool Packed>
__global__ void gemv(Args a,size_t a_step,size_t b_step,size_t c_step) {
  const int slice=blockIdx.z,row=blockIdx.x*32+threadIdx.x/8,lane=threadIdx.x%8;
  const auto* x=a.a+slice*a_step;
  const auto* b=a.b+(Packed?0:slice*b_step);
  float acc[4]={0,0,0,0};
  if(row<a.n) for(int base=0;base<a.k;base+=128) {
    const int k=base+lane*16;if(k>=a.k) continue;
    size_t offset;
    if constexpr(Packed) offset=packed_offset(a.packed.n_origin+row,a.packed.k_origin+slice*a_step+k,
        a.packed.n_extent,a.packed.k_extent,a.packed.format);
    else offset=size_t(row)*a.ldb+k;
    const uint4 v0=*reinterpret_cast<const uint4*>(b+offset);
    const uint4 v1=*reinterpret_cast<const uint4*>(b+offset+8);
    const uint32_t words[8]={v0.x,v0.y,v0.z,v0.w,v1.x,v1.y,v1.z,v1.w};
    #pragma unroll
    for(int j=0;j<8;++j) {
      const uint32_t xp=*reinterpret_cast<const uint32_t*>(x+k+2*j);
      acc[j&3]=fmaf(__uint_as_float((words[j]&65535u)<<16),__uint_as_float((xp&65535u)<<16),acc[j&3]);
      acc[j&3]=fmaf(__uint_as_float(words[j]&0xffff0000u),__uint_as_float(xp&0xffff0000u),acc[j&3]);
    }
  }
  float sum=(acc[0]+acc[1])+(acc[2]+acc[3]);
  sum+=__shfl_down(sum,4,8);sum+=__shfl_down(sum,2,8);sum+=__shfl_down(sum,1,8);
  if(lane==0 && row<a.n) a.c[slice*c_step+row]=sum;
}
bool supported(const Args& a) {
  return a.m==1 && a.k%16==0 && a.lda%16==0 && a.ldb%16==0 &&
      a.packed.k_origin%16==0;
}
void launch(const Args& a,hipStream_t s) {gemv<false><<<(a.n+31)/32,256,0,s>>>(a,0,0,0);}
void split(const Args& a,int q,size_t as,size_t bs,size_t cs,hipStream_t s) {
  gemv<false><<<dim3((a.n+31)/32,1,q),256,0,s>>>(a,as,bs,cs);
}
void packed(const Args& a,int q,size_t as,size_t cs,hipStream_t s) {
  gemv<true><<<dim3((a.n+31)/32,1,q),256,0,s>>>(a,as,0,cs);
}
void check(hipError_t e) {if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e));}
}
void register_simt(std::vector<Kernel>& ks) {
  Kernel k{"simt_gemv",1,32,128,256,{},0,supported,launch,{},0,split,{},0,packed};
  check(hipFuncGetAttributes(&k.attributes,reinterpret_cast<const void*>(gemv<false>)));
  k.split_attributes=k.attributes;
  check(hipFuncGetAttributes(&k.packed_attributes,reinterpret_cast<const void*>(gemv<true>)));
  check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&k.blocks_per_cu,reinterpret_cast<const void*>(gemv<false>),256,0));
  k.split_blocks_per_cu=k.blocks_per_cu;
  check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&k.packed_blocks_per_cu,reinterpret_cast<const void*>(gemv<true>),256,0));
  ks.push_back(k);
}
}
