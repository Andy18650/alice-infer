#pragma once
// Historical wave-indexed control, retained only for matched comparisons.
#include "readers.hpp"

template<class Reader,int Chunks>
__device__ __forceinline__
void read_wave_body(const uint32_t* __restrict__ input,size_t count,uint32_t* __restrict__ sink) {
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
      values[chunk]=Reader::load(input,(tile+chunk*waves)*256,lane,64);
    scheduling_boundary();
#pragma unroll
    for(int chunk=0;chunk<Chunks;++chunk) sums[chunk]+=values[chunk];
    scheduling_boundary();
  }
  for(;tile*256<count;tile+=waves) sums[0]+=Reader::tail(input,tile*256,lane,count,64);
  uint4 total={};
#pragma unroll
  for(int chunk=0;chunk<Chunks;++chunk) total+=sums[chunk];
  uint32_t sum=total.x+total.y+total.z+total.w;
  for(int d=32;d;d/=2) sum+=__shfl_down(sum,d,64);
  if(lane==0) sink[wave]=sum;
}

template<class Reader,int Chunks>
__global__ __launch_bounds__(256,1)
void read_wave_tiles(const uint32_t* __restrict__ input,size_t count,uint32_t* __restrict__ sink) {
  read_wave_body<Reader,Chunks>(input,count,sink);
}
