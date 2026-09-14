// gpu_rocm block-private SIMT kernels (design §3: self-contained impls).
// Compiled by hipcc (see CMakeLists gpu_rocm_kernels.o). Plain SIMT:
// shared-memory reductions only, no matrix intrinsics. bf16 conversions
// truncate (__float_as_uint >> 16), bit-identical to host bf16::from_f32.

#include <cstdlib>

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

// Flash-decoding v2, phase 1: one block per (split, query, head); 4 warps of
// 64 lanes stride the split's cache rows, each keeping a private online
// softmax state (m, l, acc), merged via shared memory at the end. Partials
// are written unnormalized in units of the block max: [acc(hd), m, l] per
// (query*head, split). Empty splits (short query in a multi-query call)
// write l = 0 and are skipped by the combine.
__global__ void k_attention_split(const float* __restrict__ qkv,
                                  const float* __restrict__ kcache,
                                  const float* __restrict__ vcache,
                                  float* __restrict__ partials, uint32_t pos0, uint32_t nh,
                                  uint32_t nkv, uint32_t hd, float scale,
                                  uint32_t rows_per_split) {
  const uint32_t sh = blockIdx.y, h = sh % nh, s = sh / nh;
  const uint32_t T = pos0 + s + 1;
  const uint32_t split = blockIdx.x;
  const uint32_t r0 = split * rows_per_split;
  const uint32_t r1 = min(T, r0 + rows_per_split);
  const uint32_t group = nh / nkv, g = h / group;
  const uint32_t kvd = nkv * hd, qkvd = (nh + 2 * nkv) * hd;
  const uint32_t lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
  const uint32_t epl = (hd + 63) / 64;  // hd <= 128 -> 1 or 2 elements per lane

  float q[2];
  for (uint32_t e = 0; e < epl; ++e) {
    const uint32_t i = lane + e * 64;
    q[e] = i < hd ? qkv[(size_t)s * qkvd + h * hd + i] : 0.0f;
  }
  float m = -INFINITY, l = 0.0f, acc[2] = {0.0f, 0.0f};
  for (uint32_t t = r0 + warp; t < r1; t += 4) {
    const float* krow = kcache + (size_t)t * kvd + g * hd;
    float part = 0.0f;
    for (uint32_t e = 0; e < epl; ++e) {
      const uint32_t i = lane + e * 64;
      if (i < hd) part += q[e] * krow[i];
    }
    for (uint32_t off = 32; off > 0; off >>= 1)
      part += __shfl_down_sync(0xffffffffffffffffULL, part, off);
    const float dot = __shfl_sync(0xffffffffffffffffULL, part, 0);
    const float sc = dot * scale;
    const float nm = fmaxf(m, sc);
    const float f = expf(m - nm);  // first row: exp(-inf) = 0
    const float p = expf(sc - nm);
    l = l * f + p;
    const float* vrow = vcache + (size_t)t * kvd + g * hd;
    for (uint32_t e = 0; e < epl; ++e) {
      const uint32_t i = lane + e * 64;
      if (i < hd) acc[e] = acc[e] * f + p * vrow[i];
    }
    m = nm;
  }

  __shared__ float sm[4], sl[4], sacc[4][128];
  if (lane == 0) {
    sm[warp] = m;
    sl[warp] = l;
  }
  for (uint32_t e = 0; e < epl; ++e) {
    const uint32_t i = lane + e * 64;
    if (i < hd) sacc[warp][i] = acc[e];
  }
  __syncthreads();
  if (warp != 0) return;

  const float M = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
  float* out = partials + ((size_t)sh * gridDim.x + split) * (hd + 2);
  if (M == -INFINITY) {  // empty split
    if (lane == 0) out[hd + 1] = 0.0f;
    return;
  }
  float w[4], L = 0.0f;
  for (uint32_t j = 0; j < 4; ++j) {
    w[j] = expf(sm[j] - M);  // empty warp: exp(-inf - M) = 0
    L += sl[j] * w[j];
  }
  for (uint32_t e = 0; e < epl; ++e) {
    const uint32_t i = lane + e * 64;
    if (i < hd) out[i] = sacc[0][i] * w[0] + sacc[1][i] * w[1] + sacc[2][i] * w[2] +
                         sacc[3][i] * w[3];
  }
  if (lane == 0) {
    out[hd] = M;
    out[hd + 1] = L;
  }
}

// Phase 2: one block per (query, head), lane i owns output element i
// (i < hd <= 128); merge the split partials and write bf16 attn. Output
// index sh*hd + i == s*(nh*hd) + h*hd + i.
__global__ void k_attention_combine(const float* __restrict__ partials,
                                    unsigned short* __restrict__ attn, uint32_t hd,
                                    uint32_t splits) {
  const uint32_t sh = blockIdx.x;
  const float* base = partials + (size_t)sh * splits * (hd + 2);
  float M = -INFINITY, L = 0.0f;
  for (uint32_t sp = 0; sp < splits; ++sp) {
    const float l = base[sp * (hd + 2) + hd + 1];
    if (l > 0.0f) M = fmaxf(M, base[sp * (hd + 2) + hd]);
  }
  for (uint32_t sp = 0; sp < splits; ++sp) {
    const float l = base[sp * (hd + 2) + hd + 1];
    if (l > 0.0f) L += l * expf(base[sp * (hd + 2) + hd] - M);
  }
  const uint32_t i = threadIdx.x;
  if (i < hd) {
    float a = 0.0f;
    for (uint32_t sp = 0; sp < splits; ++sp) {
      const float l = base[sp * (hd + 2) + hd + 1];
      if (l > 0.0f) a += base[sp * (hd + 2) + i] * expf(base[sp * (hd + 2) + hd] - M);
    }
    attn[sh * hd + i] = f2b(a / L);
  }
}

unsigned grid_for(size_t n, unsigned block) {
  unsigned g = (unsigned)((n + block - 1) / block);
  return g == 0 ? 1 : (g > 65535 ? 65535 : g);
}

// Dev-only A/B knob: ALICE_FORCE_ATTN_V1=1 pins attention to the v1 kernel.
bool force_attn_v1() {
  static const bool v1 = [] {
    const char* e = getenv("ALICE_FORCE_ATTN_V1");
    return e != nullptr && e[0] == '1';
  }();
  return v1;
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
               float* split_scratch, uint32_t pos0, uint32_t S, uint32_t nh, uint32_t nkv,
               uint32_t hd, float scale) {
  const uint32_t T = pos0 + S;
  if (!force_attn_v1() && S <= kAttnSplitMaxS && T >= kAttnSplitMinT) {
    // Flash-decoding split-T path (decode at long context): the v1 kernel is
    // sequential in T with a block reduction per cache row, so decode was
    // reduction-latency-bound; here splits*nh*S blocks run in parallel.
    uint32_t splits = (T + kAttnRowsPerSplit - 1) / kAttnRowsPerSplit;
    uint32_t rps = kAttnRowsPerSplit;
    if (splits > kAttnMaxSplits) {
      splits = kAttnMaxSplits;
      rps = (T + splits - 1) / splits;  // keep full coverage of [0, T)
    }
    k_attention_split<<<dim3(splits, S * nh), 256>>>(qkv, kcache, vcache, split_scratch,
                                                     pos0, nh, nkv, hd, scale, rps);
    k_attention_combine<<<S * nh, 128>>>((const float*)split_scratch,
                                         (unsigned short*)attn_bf16, hd, splits);
  } else {
    k_attention<<<S * nh, 128>>>(qkv, kcache, vcache, (unsigned short*)attn_bf16, pos0, nh,
                                 nkv, hd, scale);
  }
}

}  // namespace alice::gpu_rocm::kernels
