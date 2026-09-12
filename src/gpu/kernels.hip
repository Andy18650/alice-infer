// Engine trunk/state SIMT kernels (design §3). Compiled by hipcc (see
// CMakeLists alice_gpu_kernels.o). Plain SIMT: shared-memory reductions
// only, no matrix intrinsics. bf16 conversions truncate (__float_as_uint >>
// 16), bit-identical to host bf16::from_f32.

#include <hip/hip_runtime.h>

#include "kernels.hpp"

namespace {

__device__ __forceinline__ unsigned short f2b(float f) {
  return (unsigned short)(__float_as_uint(f) >> 16);
}
__device__ __forceinline__ float b2f(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}

// Block-wide sum over NT threads via shared memory (no shuffle tricks).
template <unsigned NT>
__device__ float block_reduce_sum(float v, float* shared) {
  shared[threadIdx.x] = v;
  __syncthreads();
  for (unsigned s = NT / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  return shared[0];
}

__global__ void k_f32_to_bf16(const float* __restrict__ src, unsigned short* __restrict__ dst,
                              size_t n) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t k = i; k < n; k += stride) dst[k] = f2b(src[k]);
}

__global__ void k_rmsnorm(const float* __restrict__ in, const float* __restrict__ gamma,
                          float* __restrict__ out, uint32_t d, float eps) {
  __shared__ float shared[256];
  const uint32_t s = blockIdx.x;
  const float* row = in + (size_t)s * d;
  float* orow = out + (size_t)s * d;
  float ss = 0.0f;
  for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) ss += row[i] * row[i];
  ss = block_reduce_sum<256>(ss, shared);
  const float inv = 1.0f / sqrtf(ss / (float)d + eps);
  for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) orow[i] = row[i] * inv * gamma[i];
}

__global__ void k_add_rows(float* __restrict__ dst, const float* __restrict__ src, size_t n) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t k = i; k < n; k += stride) dst[k] += src[k];
}

__global__ void k_embed_gather(const unsigned short* __restrict__ table,
                               const int* __restrict__ ids, float* __restrict__ out,
                               uint32_t d) {
  const uint32_t s = blockIdx.x;
  const unsigned short* row = table + (size_t)ids[s] * d;
  for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) out[(size_t)s * d + i] = b2f(row[i]);
}

__global__ void k_argmax(const float* __restrict__ x, size_t n, int* __restrict__ out) {
  __shared__ float vals[1024];
  __shared__ int idxs[1024];
  float best = -INFINITY;
  int besti = 0;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (x[i] > best) {
      best = x[i];
      besti = (int)i;
    }
  }
  vals[threadIdx.x] = best;
  idxs[threadIdx.x] = besti;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = vals[threadIdx.x + s];
      const int oi = idxs[threadIdx.x + s];
      if (ov > vals[threadIdx.x] || (ov == vals[threadIdx.x] && oi < idxs[threadIdx.x])) {
        vals[threadIdx.x] = ov;
        idxs[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = idxs[0];
}

unsigned grid_for(size_t n, unsigned block) {
  unsigned g = (unsigned)((n + block - 1) / block);
  return g == 0 ? 1 : (g > 65535 ? 65535 : g);
}

}  // namespace

namespace alice::gpu::kernels {

void f32_to_bf16(const float* src, void* dst, size_t n) {
  k_f32_to_bf16<<<grid_for(n, 256), 256>>>(src, (unsigned short*)dst, n);
}

void rmsnorm(const float* in, const float* gamma, float* out, uint32_t S, uint32_t d,
             float eps) {
  k_rmsnorm<<<S, 256>>>(in, gamma, out, d, eps);
}

void add_rows(float* dst, const float* src, size_t n) {
  k_add_rows<<<grid_for(n, 256), 256>>>(dst, src, n);
}

void embed_gather(const void* table, const int32_t* ids, float* out, uint32_t S,
                  uint32_t d) {
  k_embed_gather<<<S, 256>>>((const unsigned short*)table, (const int*)ids, out, d);
}

void argmax(const float* x, size_t n, int32_t* out) {
  k_argmax<<<1, 1024>>>(x, n, (int*)out);
}

}  // namespace alice::gpu::kernels
