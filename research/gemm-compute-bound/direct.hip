// Real GEMM, not repeated/register-resident synthetic operands. Four independent
// 32x32 output groups per wave, full K streamed through a register prefetch ring.
#include "api.hpp"
#include <stdexcept>
namespace compute_bound {
namespace {
using U4=uint16_t __attribute__((ext_vector_type(4)));
using F16=float __attribute__((ext_vector_type(16)));
struct Frag {U4 a0,a1,b0,b1;};
template<bool Panel>
__device__ U4 load(const uint16_t* p,int rows,int k,int row,int kk) {
  const size_t index=Panel?(size_t(kk/8)*rows+row)*8+kk%8:size_t(row)*k+kk;
  return __builtin_bit_cast(U4,*reinterpret_cast<const uint64_t*>(p+index));
}
template<Storage Format>
__device__ Frag fetch(Args a,int mr,int nr,int kk) {
  const int lane=threadIdx.x%64;
  const int pos=kk+(lane/32)*4;
  return {load<Format==Storage::ABPanel>(a.a,a.gm*128,a.k,mr+lane%32,pos),
          load<Format==Storage::ABPanel>(a.a,a.gm*128,a.k,mr+32+lane%32,pos),
          load<Format!=Storage::Affine>(a.b,a.gn*128,a.k,nr+lane%32,pos),
          load<Format!=Storage::Affine>(a.b,a.gn*128,a.k,nr+32+lane%32,pos)};
}
template<Storage Format>
__global__ __launch_bounds__(256,1) void entry(Args a) {
  int mi,ni;if(a.order==0){mi=blockIdx.x/a.gn;ni=blockIdx.x%a.gn;}else{mi=blockIdx.x%a.gm;ni=blockIdx.x/a.gm;}
  const int wave=threadIdx.x/64,lane=threadIdx.x%64;
  const int mr=mi*128+(wave/2)*64,nr=ni*128+(wave%2)*64;
  F16 c00{},c01{},c10{},c11{};
  Frag ring[STAGES];
  #pragma unroll
  for(int j=0;j<STAGES;++j)ring[j]=fetch<Format>(a,mr,nr,j*8);
  for(int base=0;base<a.k;base+=STAGES*8) {
    #pragma unroll
    for(int j=0;j<STAGES;++j) {
      const Frag current=ring[j];Frag next{};
      if(base+(j+STAGES)*8<a.k)next=fetch<Format>(a,mr,nr,base+(j+STAGES)*8);
      __builtin_amdgcn_sched_barrier(0);
      c00=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(current.a0,current.b0,c00,0,0,0);
      c01=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(current.a0,current.b1,c01,0,0,0);
      c10=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(current.a1,current.b0,c10,0,0,0);
      c11=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(current.a1,current.b1,c11,0,0,0);
      __builtin_amdgcn_sched_barrier(0);ring[j]=next;
    }
  }
  #pragma unroll
  for(int i=0;i<16;++i) {
    const int row=(i/4)*8+(lane/32)*4+i%4,col=lane%32;
    a.c[size_t(mr+row)*(a.gn*128)+nr+col]=c00[i];
    a.c[size_t(mr+row)*(a.gn*128)+nr+32+col]=c01[i];
    a.c[size_t(mr+32+row)*(a.gn*128)+nr+col]=c10[i];
    a.c[size_t(mr+32+row)*(a.gn*128)+nr+32+col]=c11[i];
  }
}
void launch(Args a,Storage f,hipStream_t s) {
  if(a.k%(STAGES*8))throw std::runtime_error("direct ring K must divide stage group");
  if(f==Storage::Affine)entry<Storage::Affine><<<a.gm*a.gn,256,0,s>>>(a);
  else if(f==Storage::BPanel)entry<Storage::BPanel><<<a.gm*a.gn,256,0,s>>>(a);
  else entry<Storage::ABPanel><<<a.gm*a.gn,256,0,s>>>(a);
}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
}
void REGISTER(std::vector<Candidate>& cs) {
  Candidate c{KERNEL_NAME,128,128,8,32,256,21,STAGES,0,{}, {},launch};
  const void* fn[]={reinterpret_cast<const void*>(entry<Storage::Affine>),reinterpret_cast<const void*>(entry<Storage::BPanel>),reinterpret_cast<const void*>(entry<Storage::ABPanel>)};
  for(int i=0;i<3;++i){check(hipFuncGetAttributes(&c.attributes[i],fn[i]));check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&c.capacity[i],fn[i],256,0));}
  cs.push_back(c);
}
}
