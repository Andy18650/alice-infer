#pragma once
// Standalone gfx90a study: BF16 weights, genuinely FP32 x/accumulation/y.
#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef GEMV_LAST_STRONG_PUBLICATION
#define GEMV_LAST_STRONG_PUBLICATION 0
#endif

namespace gemv_layout {
constexpr uint32_t threads=256, rows_per_tile=2*threads;
enum class Format { Interleaved, Adjacent, Column };
enum class Mode { Read, Expand, Gemv };
struct Geometry {
  uint32_t rows,cols,row_tiles,splits,steps,padded_rows,jobs;
  __host__ __device__ uint32_t padded_cols() const { return steps*4*splits; }
  __host__ __device__ size_t records() const { return size_t(jobs)*steps*threads; }
};

__host__ __device__ inline uint16_t weight_bits(uint32_t row,uint32_t col,uint32_t seed) {
  if(seed==0) return 0x3f80; // Exact ones for FP32-x precision tests.
  uint32_t h=seed^(row*0x9e3779b9u)^(col*0x85ebca6bu);
  h^=h>>16; h*=0x7feb352du; h^=h>>15; h*=0x846ca68bu; h^=h>>16;
  return uint16_t(0x3f00u|(h&127u)|((h>>1)&0x8000u)); // Finite signed BF16.
}
__host__ __device__ inline uint32_t logical_word(Geometry g,uint32_t pair,uint32_t k,uint32_t seed) {
  if(k>=g.cols) return 0;
  const uint32_t n=2*pair;
  return (n<g.rows?uint32_t(weight_bits(n,k,seed)):0u) |
         (n+1<g.rows?uint32_t(weight_bits(n+1,k,seed))<<16:0u);
}

template<Format F>
__global__ void initialize(uint4* packed,Geometry g,uint32_t seed) {
  if constexpr(F==Format::Column) {
    auto* p=reinterpret_cast<uint32_t*>(packed);
    const size_t words=g.records()*4;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x)
      p[i]=logical_word(g,i%(g.padded_rows/2),i/(g.padded_rows/2),seed);
  } else {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<g.records();i+=size_t(gridDim.x)*blockDim.x) {
      const uint32_t t=i%threads;
      const size_t tile=i/threads;
      const uint32_t job=F==Format::Adjacent?tile/g.steps:tile%g.jobs;
      const uint32_t step=F==Format::Adjacent?tile%g.steps:tile/g.jobs;
      const uint32_t pair=(job%g.row_tiles)*threads+t;
      const uint32_t k=((job/g.row_tiles)*g.steps+step)*4;
      packed[i]={logical_word(g,pair,k,seed),logical_word(g,pair,k+1,seed),
                 logical_word(g,pair,k+2,seed),logical_word(g,pair,k+3,seed)};
    }
  }
}

__device__ __forceinline__ void phase_boundary() { __builtin_amdgcn_sched_barrier(0); }
__device__ __forceinline__ float low(uint32_t v) { return __uint_as_float(v<<16); }
__device__ __forceinline__ float high(uint32_t v) { return __uint_as_float(v&0xffff0000u); }

template<Format F>
__device__ __forceinline__ uint4 load(const uint4* packed,Geometry g,uint32_t job,uint32_t step,uint32_t pair,uint32_t k) {
  if constexpr(F==Format::Column) {
    const auto* p=reinterpret_cast<const uint32_t*>(packed);
    const size_t stride=g.padded_rows/2;
    return {p[size_t(k)*stride+pair],p[size_t(k+1)*stride+pair],
            p[size_t(k+2)*stride+pair],p[size_t(k+3)*stride+pair]};
  } else {
    const size_t tile=F==Format::Adjacent?size_t(job)*g.steps+step:size_t(step)*g.jobs+job;
    return packed[tile*threads+threadIdx.x];
  }
}

struct Accumulator {
  float4 even{},odd{};
  __device__ __forceinline__ void expand(uint4 w) {
    even.x+=low(w.x); odd.x+=high(w.x); even.y+=low(w.y); odd.y+=high(w.y);
    even.z+=low(w.z); odd.z+=high(w.z); even.w+=low(w.w); odd.w+=high(w.w);
  }
  __device__ __forceinline__ void multiply(uint4 w,float4 x) {
    even.x=fmaf(low(w.x),x.x,even.x); odd.x=fmaf(high(w.x),x.x,odd.x);
    even.y=fmaf(low(w.y),x.y,even.y); odd.y=fmaf(high(w.y),x.y,odd.y);
    even.z=fmaf(low(w.z),x.z,even.z); odd.z=fmaf(high(w.z),x.z,odd.z);
    even.w=fmaf(low(w.w),x.w,even.w); odd.w=fmaf(high(w.w),x.w,odd.w);
  }
  __device__ __forceinline__ float2 finish() const {
    return {(even.x+even.y)+(even.z+even.w),(odd.x+odd.y)+(odd.z+odd.w)};
  }
};

template<Format F,int Batch,Mode M>
__device__ __forceinline__
void compute_job(const uint4* __restrict__ weights,const float* __restrict__ x,
                 float* __restrict__ output,uint32_t* __restrict__ sink,Geometry g,uint32_t job) {
    const uint32_t split=job/g.row_tiles;
    const uint32_t pair=(job%g.row_tiles)*threads+threadIdx.x;
    const uint32_t kbase=split*g.steps*4;
    uint4 checksum{};
    Accumulator accum;
    uint32_t step=0;
    for(;step+Batch<=g.steps;step+=Batch) {
      uint4 values[Batch];
#pragma unroll
      for(int b=0;b<Batch;++b) values[b]=load<F>(weights,g,job,step+b,pair,kbase+(step+b)*4);
      phase_boundary();
#pragma unroll
      for(int b=0;b<Batch;++b) {
        if constexpr(M==Mode::Read) checksum+=values[b];
        else if constexpr(M==Mode::Expand) accum.expand(values[b]);
        else accum.multiply(values[b],reinterpret_cast<const float4*>(x)[kbase/4+step+b]);
      }
      phase_boundary();
    }
    for(;step<g.steps;++step) {
      const uint4 w=load<F>(weights,g,job,step,pair,kbase+step*4);
      if constexpr(M==Mode::Read) checksum+=w;
      else if constexpr(M==Mode::Expand) accum.expand(w);
      else accum.multiply(w,reinterpret_cast<const float4*>(x)[kbase/4+step]);
    }
    if constexpr(M==Mode::Read) sink[size_t(job)*threads+threadIdx.x]=checksum.x+checksum.y+checksum.z+checksum.w;
    else reinterpret_cast<float2*>(output+size_t(split)*g.padded_rows)[pair]=accum.finish();
}

template<Format F,int Batch,Mode M>
__global__ __launch_bounds__(256,1)
void partial(const uint4* __restrict__ weights,const float* __restrict__ x,
             float* __restrict__ output,uint32_t* __restrict__ sink,Geometry g) {
  for(uint32_t job=blockIdx.x;job<g.jobs;job+=gridDim.x)
    compute_job<F,Batch,M>(weights,x,output,sink,g,job);
}

[[maybe_unused]] static __global__ void reset_counters(uint32_t* counters,uint32_t tiles,uint32_t splits,uint32_t stride) {
  const uint32_t tile=blockIdx.x*blockDim.x+threadIdx.x;
  if(tile<tiles) counters[size_t(tile)*stride]=splits;
}

// Match the scalar combine's arithmetic tree, but own two adjacent rows.
__device__ __forceinline__ float2 reduce_pair(const float* partials,Geometry g,uint32_t pair) {
  float2 accum[8]={}; uint32_t split=0;
  for(;split+8<=g.splits;split+=8) {
    float2 values[8];
#pragma unroll
    for(int b=0;b<8;++b) values[b]=reinterpret_cast<const float2*>(partials+size_t(split+b)*g.padded_rows)[pair];
    phase_boundary();
#pragma unroll
    for(int b=0;b<8;++b) accum[b]+=values[b];
    phase_boundary();
  }
  float2 tail[8]={};
#pragma unroll
  for(int b=0;b<8;++b) if(split+b<g.splits)
    tail[b]=reinterpret_cast<const float2*>(partials+size_t(split+b)*g.padded_rows)[pair];
  phase_boundary();
#pragma unroll
  for(int b=0;b<8;++b) if(split+b<g.splits) accum[0]+=tail[b];
  return ((accum[0]+accum[1])+(accum[2]+accum[3]))+((accum[4]+accum[5])+(accum[6]+accum[7]));
}

template<Format F,int Batch>
__global__ __launch_bounds__(256,1)
void last_arriver(const uint4* __restrict__ weights,const float* __restrict__ x,
                  float* __restrict__ partials,float* __restrict__ y,
                  uint32_t* __restrict__ counters,Geometry g,uint32_t counter_stride) {
  __shared__ uint32_t last;
  for(uint32_t job=blockIdx.x;job<g.jobs;job+=gridDim.x) {
    compute_job<F,Batch,Mode::Gemv>(weights,x,partials,nullptr,g,job);
    // Every writer completes device-scope publication before the elected
    // thread releases the contribution through the counter RMW.
    if constexpr(GEMV_LAST_STRONG_PUBLICATION) __threadfence();
    else __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
    __syncthreads();
    const uint32_t tile=job%g.row_tiles;
    if(threadIdx.x==0) {
      constexpr int order=GEMV_LAST_STRONG_PUBLICATION?__ATOMIC_ACQ_REL:__ATOMIC_RELEASE;
      const uint32_t old=__hip_atomic_fetch_add(counters+size_t(tile)*counter_stride,uint32_t(-1),
                                              order,__HIP_MEMORY_SCOPE_AGENT);
      // Atomic-fence synchronization: the returned old value reads the
      // preceding release sequence. Acquire before broadcasting the win.
      if constexpr(!GEMV_LAST_STRONG_PUBLICATION)
        if(old==1) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
      last=old==1;
    }
    __syncthreads(); // Broadcast winner/acquire to all participating waves.
    if(!last) continue; // Do not abandon subsequent persistent-grid jobs.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
    const uint32_t pair=tile*threads+threadIdx.x;
    reinterpret_cast<float2*>(y)[pair]=reduce_pair(partials,g,pair);
    __syncthreads();
    // No subsequent contribution to this tile exists in this invocation.
    // Stream-ordered kernel completion protects y and state reuse; this
    // scratch/counter set is NOT reentrant across concurrent streams.
    if(threadIdx.x==0) __hip_atomic_store(counters+size_t(tile)*counter_stride,g.splits,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
    __syncthreads(); // Shared 'last' may now be reused by the next job.
  }
}

[[maybe_unused]] static __global__ void combine(const float* __restrict__ partials,float* __restrict__ y,Geometry g) {
  const uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=g.rows) return;
  float accum[8]={}; uint32_t split=0;
  for(;split+8<=g.splits;split+=8) {
    float values[8];
#pragma unroll
    for(int b=0;b<8;++b) values[b]=partials[size_t(split+b)*g.padded_rows+row];
    phase_boundary();
#pragma unroll
    for(int b=0;b<8;++b) accum[b]+=values[b];
    phase_boundary();
  }
  for(;split<g.splits;++split) accum[0]+=partials[size_t(split)*g.padded_rows+row];
  y[row]=((accum[0]+accum[1])+(accum[2]+accum[3]))+((accum[4]+accum[5])+(accum[6]+accum[7]));
}
} // namespace gemv_layout
