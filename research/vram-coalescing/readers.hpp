#pragma once
// Private gfx90a HIP definitions shared by bandwidth and placement probes.
#include <hip/hip_runtime.h>
#include <cstdint>

using Float4 = float __attribute__((ext_vector_type(4)));
using Words4 = uint32_t __attribute__((ext_vector_type(4)));
constexpr uint32_t mantissa_mask=(1u<<23)-1, float_base=0x3f000000u;

[[maybe_unused]] static __global__ void initialize(float* p,size_t count) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
    const uint32_t bits=float_base|((uint32_t(i)*1664525u+1013904223u)&mantissa_mask);
    p[i]=__uint_as_float(bits);
  }
}

__device__ __forceinline__ uint32_t tail_word(const float* p,size_t index,size_t count) {
  return index<count?__float_as_uint(p[index]):0;
}

struct A {
  static constexpr const char* name="A_vector_contiguous";
  static __device__ __forceinline__ Words4 load(const float* p,size_t base,unsigned lane) {
    const auto* vectors=reinterpret_cast<const Float4*>(p+base);
    const Float4 value=vectors[lane];
    __builtin_amdgcn_sched_barrier(0);
    return __builtin_bit_cast(Words4,value);
  }
  static __device__ __forceinline__ Words4 tail(const float* p,size_t base,unsigned lane,size_t count) {
    const size_t first=base+4*lane;
    return {tail_word(p,first,count),tail_word(p,first+1,count),
            tail_word(p,first+2,count),tail_word(p,first+3,count)};
  }
};

struct B {
  static constexpr const char* name="B_scalar_striped";
  static __device__ __forceinline__ Words4 load(const float* p,size_t base,unsigned lane) {
    const uint32_t x=__float_as_uint(p[base+lane]);
    __builtin_amdgcn_sched_barrier(0);
    const uint32_t y=__float_as_uint(p[base+lane+64]);
    __builtin_amdgcn_sched_barrier(0);
    const uint32_t z=__float_as_uint(p[base+lane+128]);
    __builtin_amdgcn_sched_barrier(0);
    const uint32_t w=__float_as_uint(p[base+lane+192]);
    __builtin_amdgcn_sched_barrier(0);
    return {x,y,z,w};
  }
  static __device__ __forceinline__ Words4 tail(const float* p,size_t base,unsigned lane,size_t count) {
    return {tail_word(p,base+lane,count),tail_word(p,base+lane+64,count),
            tail_word(p,base+lane+128,count),tail_word(p,base+lane+192,count)};
  }
};

struct C {
  static constexpr const char* name="C_scalar_contiguous";
  static __device__ __forceinline__ Words4 load(const float* p,size_t base,unsigned lane) {
    const size_t first=base+4*lane;
    const uint32_t x=__float_as_uint(p[first]);
    __builtin_amdgcn_sched_barrier(0);
    const uint32_t y=__float_as_uint(p[first+1]);
    __builtin_amdgcn_sched_barrier(0);
    const uint32_t z=__float_as_uint(p[first+2]);
    __builtin_amdgcn_sched_barrier(0);
    const uint32_t w=__float_as_uint(p[first+3]);
    __builtin_amdgcn_sched_barrier(0);
    return {x,y,z,w};
  }
  static __device__ __forceinline__ Words4 tail(const float* p,size_t base,unsigned lane,size_t count) {
    return A::tail(p,base,lane,count);
  }
};

template<class Reader,int Chunks>
__device__ __forceinline__
void read_body(const float* __restrict__ input,size_t count,uint32_t* __restrict__ sink) {
  const size_t wave=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/64;
  const size_t waves=size_t(gridDim.x)*blockDim.x/64;
  const unsigned lane=threadIdx.x&63;
  const size_t full_tiles=count/256;
  Words4 sums[Chunks]={};
  size_t tile=wave;
  for(;tile+(Chunks-1)*waves<full_tiles;tile+=Chunks*waves) {
    Words4 values[Chunks];
#pragma unroll
    for(int chunk=0;chunk<Chunks;++chunk)
      values[chunk]=Reader::load(input,(tile+chunk*waves)*256,lane);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for(int chunk=0;chunk<Chunks;++chunk) sums[chunk]+=values[chunk];
    __builtin_amdgcn_sched_barrier(0);
  }
  for(;tile*256<count;tile+=waves) sums[0]+=Reader::tail(input,tile*256,lane,count);
  Words4 total={};
#pragma unroll
  for(int chunk=0;chunk<Chunks;++chunk) total+=sums[chunk];
  uint32_t sum=total[0]+total[1]+total[2]+total[3];
  for(int d=32;d;d/=2) sum+=__shfl_down(sum,d,64);
  if(lane==0) sink[wave]=sum;
}

template<class Reader,int Chunks>
__global__ __launch_bounds__(256,1)
void read_tiles(const float* __restrict__ input,size_t count,uint32_t* __restrict__ sink) {
  read_body<Reader,Chunks>(input,count,sink);
}

inline uint32_t expected_sum(size_t words) {
  constexpr uint64_t period=uint64_t(1)<<23;
  constexpr uint64_t period_sum=period*float_base+period*(period-1)/2;
  uint32_t sum=uint32_t(words/period)*uint32_t(period_sum);
  for(uint32_t i=0;i<words%period;++i)
    sum+=float_base|((i*1664525u+1013904223u)&mantissa_mask);
  return sum;
}
