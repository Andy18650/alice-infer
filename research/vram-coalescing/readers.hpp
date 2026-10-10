#pragma once
// Private gfx90a HIP definitions shared by bandwidth and placement probes.
#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef RESEARCH_SCHED_BARRIERS
#define RESEARCH_SCHED_BARRIERS 0
#endif
#ifndef RESEARCH_LOAD_BARRIERS
#define RESEARCH_LOAD_BARRIERS RESEARCH_SCHED_BARRIERS
#endif

__device__ __forceinline__ void scheduling_boundary() {
#if RESEARCH_SCHED_BARRIERS
  __builtin_amdgcn_sched_barrier(0);
#endif
}

// Optional fine-grained constraints inside each reader. Phase-only builds
// retain the load/consume boundaries but leave address/load scheduling free.
__device__ __forceinline__ void load_boundary() {
#if RESEARCH_LOAD_BARRIERS
  scheduling_boundary();
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

// Every reader returns four dwords. A/C use per-thread contiguous records;
// B uses four stripes of component_stride dwords (normally blockDim.x).
struct A {
  static constexpr const char* name="A_vector_contiguous";
  static __device__ __forceinline__ uint4 load(const uint32_t* p,size_t base,unsigned thread,unsigned /*component_stride*/) {
    const uint4 value=reinterpret_cast<const uint4*>(p+base)[thread];
    load_boundary();
    return value;
  }
  static __device__ __forceinline__ uint4 tail(const uint32_t* p,size_t base,unsigned thread,size_t count,unsigned /*component_stride*/) {
    const size_t first=base+4*thread;
    return {tail_word(p,first,count),tail_word(p,first+1,count),
            tail_word(p,first+2,count),tail_word(p,first+3,count)};
  }
};

struct B {
  static constexpr const char* name="B_scalar_striped";
  static __device__ __forceinline__ uint4 load(const uint32_t* p,size_t base,unsigned thread,unsigned component_stride) {
    const uint32_t x=p[base+thread];
    load_boundary();
    const uint32_t y=p[base+thread+component_stride];
    load_boundary();
    const uint32_t z=p[base+thread+2*component_stride];
    load_boundary();
    const uint32_t w=p[base+thread+3*component_stride];
    load_boundary();
    return {x,y,z,w};
  }
  static __device__ __forceinline__ uint4 tail(const uint32_t* p,size_t base,unsigned thread,size_t count,unsigned component_stride) {
    return {tail_word(p,base+thread,count),tail_word(p,base+thread+component_stride,count),
            tail_word(p,base+thread+2*component_stride,count),tail_word(p,base+thread+3*component_stride,count)};
  }
};

struct C {
  static constexpr const char* name="C_scalar_contiguous";
  static __device__ __forceinline__ uint4 load(const uint32_t* p,size_t base,unsigned thread,unsigned /*component_stride*/) {
    const size_t first=base+4*thread;
    const uint32_t x=p[first];
    load_boundary();
    const uint32_t y=p[first+1];
    load_boundary();
    const uint32_t z=p[first+2];
    load_boundary();
    const uint32_t w=p[first+3];
    load_boundary();
    return {x,y,z,w};
  }
  static __device__ __forceinline__ uint4 tail(const uint32_t* p,size_t base,unsigned thread,size_t count,unsigned component_stride) {
    return A::tail(p,base,thread,count,component_stride);
  }
};

// Load indexing is block/thread based. Native wave arithmetic is confined
// to the final checksum reduction and one-output-per-wave bookkeeping.
// Adjacent flips the tile/chunk axes: each CTA owns Chunks consecutive
// tiles, rather than one tile in each of Chunks grid-sized regions.
template<class Reader,int Chunks,bool Adjacent=false>
__device__ __forceinline__
void read_body(const uint32_t* __restrict__ input,size_t count,uint32_t* __restrict__ sink) {
  const unsigned thread=threadIdx.x, threads=blockDim.x;
  const size_t block_words=4*size_t(threads);
  const size_t grid_words=gridDim.x*block_words;
  const size_t chunk_words=Adjacent?block_words:grid_words;
  size_t base=blockIdx.x*block_words*(Adjacent?Chunks:1);
  uint4 sums[Chunks]={};
  for(;base+(Chunks-1)*chunk_words+block_words<=count;base+=Chunks*grid_words) {
    uint4 values[Chunks];
#pragma unroll
    for(int chunk=0;chunk<Chunks;++chunk)
      values[chunk]=Reader::load(input,base+chunk*chunk_words,thread,threads);
    scheduling_boundary();
#pragma unroll
    for(int chunk=0;chunk<Chunks;++chunk) sums[chunk]+=values[chunk];
    scheduling_boundary();
  }
  if constexpr(Adjacent) {
    for(;base<count;base+=Chunks*grid_words) {
#pragma unroll
      for(int chunk=0;chunk<Chunks;++chunk)
        sums[0]+=Reader::tail(input,base+chunk*block_words,thread,count,threads);
    }
  } else {
    for(;base<count;base+=grid_words) sums[0]+=Reader::tail(input,base,thread,count,threads);
  }
  uint4 total={};
#pragma unroll
  for(int chunk=0;chunk<Chunks;++chunk) total+=sums[chunk];
  uint32_t sum=total.x+total.y+total.z+total.w;
  for(int d=32;d;d/=2) sum+=__shfl_down(sum,d,64);
  if((thread&63)==0) sink[(size_t(blockIdx.x)*threads+thread)/64]=sum;
}

template<class Reader,int Chunks>
__global__ __launch_bounds__(256,1)
void read_tiles(const uint32_t* __restrict__ input,size_t count,uint32_t* __restrict__ sink) {
  read_body<Reader,Chunks>(input,count,sink);
}

template<class Reader,int Chunks>
__global__ __launch_bounds__(256,1)
void read_adjacent_tiles(const uint32_t* __restrict__ input,size_t count,uint32_t* __restrict__ sink) {
  read_body<Reader,Chunks,true>(input,count,sink);
}

// Four body-level traversals, one resident grid. No CTA or grid barrier:
// waves may enter the next window independently. Keep distinct outputs.
template<class Reader,int Chunks,bool Adjacent>
__global__ __launch_bounds__(256,1)
void read_window_bodies(const uint32_t* __restrict__ input,size_t window_words,uint32_t* __restrict__ sink) {
  const size_t outputs=size_t(gridDim.x)*blockDim.x/64;
#pragma unroll 1
  for(unsigned window=0;window<4;++window) {
    read_body<Reader,Chunks,Adjacent>(input+window*window_words,window_words,sink+window*outputs);
    scheduling_boundary();
  }
}

inline uint32_t expected_sum(size_t words) {
  const uint64_t n=words;
  return uint32_t(n*(n-1)/2)*multiplier+uint32_t(n)*increment;
}
