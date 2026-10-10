#pragma once
// Private gfx90a HIP definitions shared by bandwidth and placement probes.
#include <hip/hip_runtime.h>
#include <cstdint>

// Default: no compiler scheduling barriers. Retain an explicit matched
// control build for the barrier-removal experiment; no runtime branching.
#ifndef RESEARCH_SCHED_BARRIERS
#define RESEARCH_SCHED_BARRIERS 0
#endif

__device__ __forceinline__ void scheduling_boundary() {
#if RESEARCH_SCHED_BARRIERS
  __builtin_amdgcn_sched_barrier(0);
#endif
}

constexpr uint32_t multiplier=1664525u, increment=1013904223u;

[[maybe_unused]] static __global__ void initialize(uint32_t* p,size_t count) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)
    p[i]=uint32_t(i)*multiplier+increment;
}

__device__ __forceinline__ uint32_t tail_word(const uint32_t* p,size_t index,size_t count) {
  return index<count?p[index]:0;
}

struct A {
  static constexpr const char* name="A_vector_contiguous";
  static __device__ __forceinline__ uint4 load(const uint32_t* p,size_t base,unsigned lane) {
    const uint4 value=reinterpret_cast<const uint4*>(p+base)[lane];
    scheduling_boundary();
    return value;
  }
  static __device__ __forceinline__ uint4 tail(const uint32_t* p,size_t base,unsigned lane,size_t count) {
    const size_t first=base+4*lane;
    return {tail_word(p,first,count),tail_word(p,first+1,count),
            tail_word(p,first+2,count),tail_word(p,first+3,count)};
  }
};

struct B {
  static constexpr const char* name="B_scalar_striped";
  static __device__ __forceinline__ uint4 load(const uint32_t* p,size_t base,unsigned lane) {
    const uint32_t x=p[base+lane];
    scheduling_boundary();
    const uint32_t y=p[base+lane+64];
    scheduling_boundary();
    const uint32_t z=p[base+lane+128];
    scheduling_boundary();
    const uint32_t w=p[base+lane+192];
    scheduling_boundary();
    return {x,y,z,w};
  }
  static __device__ __forceinline__ uint4 tail(const uint32_t* p,size_t base,unsigned lane,size_t count) {
    return {tail_word(p,base+lane,count),tail_word(p,base+lane+64,count),
            tail_word(p,base+lane+128,count),tail_word(p,base+lane+192,count)};
  }
};

struct C {
  static constexpr const char* name="C_scalar_contiguous";
  static __device__ __forceinline__ uint4 load(const uint32_t* p,size_t base,unsigned lane) {
    const size_t first=base+4*lane;
    const uint32_t x=p[first];
    scheduling_boundary();
    const uint32_t y=p[first+1];
    scheduling_boundary();
    const uint32_t z=p[first+2];
    scheduling_boundary();
    const uint32_t w=p[first+3];
    scheduling_boundary();
    return {x,y,z,w};
  }
  static __device__ __forceinline__ uint4 tail(const uint32_t* p,size_t base,unsigned lane,size_t count) {
    return A::tail(p,base,lane,count);
  }
};

template<class Reader,int Chunks>
__device__ __forceinline__
void read_body(const uint32_t* __restrict__ input,size_t count,uint32_t* __restrict__ sink) {
  const size_t wave=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/64;
  const size_t waves=size_t(gridDim.x)*blockDim.x/64;
  const unsigned lane=threadIdx.x&63;
  const size_t full_tiles=count/256;
  uint4 sums[Chunks]={};
  size_t tile=wave;
  for(;tile+(Chunks-1)*waves<full_tiles;tile+=Chunks*waves) {
    uint4 values[Chunks];
#pragma unroll
    for(int chunk=0;chunk<Chunks;++chunk)
      values[chunk]=Reader::load(input,(tile+chunk*waves)*256,lane);
    scheduling_boundary();
#pragma unroll
    for(int chunk=0;chunk<Chunks;++chunk) sums[chunk]+=values[chunk];
    scheduling_boundary();
  }
  for(;tile*256<count;tile+=waves) sums[0]+=Reader::tail(input,tile*256,lane,count);
  uint4 total={};
#pragma unroll
  for(int chunk=0;chunk<Chunks;++chunk) total+=sums[chunk];
  uint32_t sum=total.x+total.y+total.z+total.w;
  for(int d=32;d;d/=2) sum+=__shfl_down(sum,d,64);
  if(lane==0) sink[wave]=sum;
}

template<class Reader,int Chunks>
__global__ __launch_bounds__(256,1)
void read_tiles(const uint32_t* __restrict__ input,size_t count,uint32_t* __restrict__ sink) {
  read_body<Reader,Chunks>(input,count,sink);
}

inline uint32_t expected_sum(size_t words) {
  // Sum the affine sequence analytically, with defined modulo-2^32 arithmetic.
  // The input allocation is at most 4 GiB, so the 64-bit triangular sum fits.
  const uint64_t n=words;
  return uint32_t(n*(n-1)/2)*multiplier+uint32_t(n)*increment;
}
