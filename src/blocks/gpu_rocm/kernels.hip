// gpu_rocm block-private SIMT kernels (design §3: self-contained impls).
// Compiled by hipcc (see CMakeLists gpu_rocm_kernels.o). Plain SIMT:
// shared-memory reductions only, no matrix intrinsics. bf16 conversions
// truncate (__float_as_uint >> 16), bit-identical to host bf16::from_f32.

#include <hip/hip_runtime.h>

#include "kernels.hpp"

namespace {

__device__ __forceinline__ unsigned short f2b(float f) {
  return (unsigned short)(__float_as_uint(f) >> 16);
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

__global__ void k_silu_mul(const float* __restrict__ gu, unsigned short* __restrict__ act,
                           uint32_t S, uint32_t I) {
  const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t n = (size_t)S * I;
  if (idx >= n) return;
  const uint32_t s = (uint32_t)(idx / I), i = (uint32_t)(idx % I);
  // gu is the fused GEMM output col-major [2I, S] == row-major [S, 2I].
  const float g = gu[(size_t)s * 2 * I + i];
  const float u = gu[(size_t)s * 2 * I + I + i];
  act[idx] = f2b(g / (1.0f + expf(-g)) * u);
}

__global__ void k_rope_append(float* __restrict__ qkv, float* __restrict__ kcache,
                              float* __restrict__ vcache, const float* __restrict__ cos_t,
                              const float* __restrict__ sin_t, uint32_t pos0, uint32_t nh,
                              uint32_t nkv, uint32_t hd) {
  const uint32_t s = blockIdx.x;
  const uint32_t pos = pos0 + s, half = hd / 2, qd = nh * hd, kvd = nkv * hd;
  const uint32_t qkvd = qd + 2 * kvd;
  const float* crow = cos_t + (size_t)pos * half;
  const float* srow = sin_t + (size_t)pos * half;
  float* row = qkv + (size_t)s * qkvd;
  // q heads: rotate in place
  for (uint32_t p = threadIdx.x; p < nh * half; p += blockDim.x) {
    const uint32_t h = p / half, i = p % half;
    float* base = row + h * hd;
    const float x1 = base[i], x2 = base[i + half];
    base[i] = x1 * crow[i] - x2 * srow[i];
    base[i + half] = x2 * crow[i] + x1 * srow[i];
  }
  // k heads: rotate, then write into the cache at pos
  for (uint32_t p = threadIdx.x; p < nkv * half; p += blockDim.x) {
    const uint32_t g = p / half, i = p % half;
    const float* base = row + qd + g * hd;
    const float x1 = base[i], x2 = base[i + half];
    kcache[(size_t)pos * kvd + g * hd + i] = x1 * crow[i] - x2 * srow[i];
    kcache[(size_t)pos * kvd + g * hd + i + half] = x2 * crow[i] + x1 * srow[i];
  }
  // v: plain append
  for (uint32_t e = threadIdx.x; e < kvd; e += blockDim.x)
    vcache[(size_t)pos * kvd + e] = row[qd + kvd + e];
}

// One block per (s, h): online-softmax attention over cache rows [0, T).
// Lane i owns output element i (i < hd <= 128); the score dot is a
// block-wide reduction per cache row.
__global__ void k_attention(const float* __restrict__ qkv, const float* __restrict__ kcache,
                            const float* __restrict__ vcache,
                            unsigned short* __restrict__ attn, uint32_t pos0, uint32_t nh,
                            uint32_t nkv, uint32_t hd, float scale) {
  __shared__ float shared[128];
  const uint32_t h = blockIdx.x % nh, s = blockIdx.x / nh;
  const uint32_t T = pos0 + s + 1, group = nh / nkv, g = h / group;
  const uint32_t kvd = nkv * hd, qkvd = (nh + 2 * nkv) * hd, qd = nh * hd;
  const uint32_t i = threadIdx.x;
  const float q_i = i < hd ? qkv[(size_t)s * qkvd + h * hd + i] : 0.0f;
  float m = -INFINITY, l = 0.0f, acc = 0.0f;
  for (uint32_t t = 0; t < T; ++t) {
    const float part =
        i < hd ? q_i * kcache[(size_t)t * kvd + g * hd + i] : 0.0f;
    const float dot = block_reduce_sum<128>(part, shared);
    const float sc = dot * scale;
    const float nm = fmaxf(m, sc);
    const float f = expf(m - nm);  // first iter: exp(-inf)=0
    const float p = expf(sc - nm);
    l = l * f + p;
    if (i < hd) acc = acc * f + p * vcache[(size_t)t * kvd + g * hd + i];
    m = nm;
  }
  if (i < hd) attn[(size_t)s * qd + h * hd + i] = f2b(acc / l);
}

unsigned grid_for(size_t n, unsigned block) {
  unsigned g = (unsigned)((n + block - 1) / block);
  return g == 0 ? 1 : (g > 65535 ? 65535 : g);
}

}  // namespace

namespace alice::gpu_rocm::kernels {

void f32_to_bf16(const float* src, void* dst, size_t n) {
  k_f32_to_bf16<<<grid_for(n, 256), 256>>>(src, (unsigned short*)dst, n);
}

void silu_mul(const float* gu, void* act, uint32_t S, uint32_t I) {
  k_silu_mul<<<grid_for((size_t)S * I, 256), 256>>>(gu, (unsigned short*)act, S, I);
}

void rope_append(float* qkv, float* kcache, float* vcache, const float* cos_t,
                 const float* sin_t, uint32_t pos0, uint32_t S, uint32_t nh, uint32_t nkv,
                 uint32_t hd) {
  k_rope_append<<<S, 256>>>(qkv, kcache, vcache, cos_t, sin_t, pos0, nh, nkv, hd);
}

void attention(const float* qkv, const float* kcache, const float* vcache, void* attn_bf16,
               uint32_t pos0, uint32_t S, uint32_t nh, uint32_t nkv, uint32_t hd,
               float scale) {
  k_attention<<<S * nh, 128>>>(qkv, kcache, vcache, (unsigned short*)attn_bf16, pos0, nh,
                               nkv, hd, scale);
}

}  // namespace alice::gpu_rocm::kernels
