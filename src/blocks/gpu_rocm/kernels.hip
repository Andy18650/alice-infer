// gpu_rocm block-private SIMT kernels (design §3: self-contained impls).
// Compiled by hipcc (see CMakeLists gpu_rocm_kernels.o). Plain SIMT:
// shared-memory reductions only, no matrix intrinsics. bf16 conversions
// truncate (__float_as_uint >> 16), bit-identical to host bf16::from_f32.

#include <stdexcept>

#include <hip/hip_runtime.h>

#include "kernels.hpp"

namespace {

constexpr uint32_t kCK = alice::gpu_rocm::kernels::kPrefillChunkKeys;

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

// Split-T prefill, main phase (v4 of research/prefill-attn-ab, the measured
// winner): one block per (query-tile of 64, key-chunk of 64, q-head), one
// thread per query. Raw scores go to the key-major LDS tile (conflict-free
// in both phases), exp is deferred until the single-chunk max is known, and
// the partial (acc, max, sum_exp) goes to the shared scratch. vec[] holds q
// in the dot phase and is reused as acc in the value phase (one array, one
// register budget); see docs/amd-gpu-notes.md for the full occupancy
// accounting. __launch_bounds__'s 2nd arg on HIP is waves-per-EU (per SIMD),
// NOT per CU (a CUDA divergence): (64, 1) targets 1 wavefront/SIMD =
// 4 wavefronts/CU = 4 blocks/CU (a 64-thread block is one wavefront on
// gfx9) — the design target and the achieved occupancy (the resource report
// shows Occupancy 1 wave/SIMD; the 16 KB LDS tile is the binding resource).
// VGPR use is 116 (hd=64) / 183 (hd=128), under the 256 archVGPR ceiling,
// and the kernel has no MFMA (AGPRs: 0) so it gets the full VGPR budget —
// no spill (ScratchSize 0). (64, 4) would request 4 waves/SIMD = 16/CU,
// unachievable here, and only produced a spurious -Wpass-failed.
// Zero __syncthreads (score rows are thread-private); fully-masked
// chunk-blocks return before touching K/V.
template <uint32_t HD>
__global__ void __launch_bounds__(64, 1)
    k_prefill_split(const float* __restrict__ qkv, const float* __restrict__ kcache,
                    const float* __restrict__ vcache, float* __restrict__ partials,
                    uint32_t pos0, uint32_t S, uint32_t nh, uint32_t nkv, float scale,
                    uint32_t max_chunks) {
  __shared__ float scores[kCK * 64];  // scores[64 * key + thread]
  const uint32_t query = blockIdx.x * 64 + threadIdx.x;
  if (query >= S) return;
  const uint32_t keys_for_query = pos0 + query + 1;  // causal bound (exclusive)
  const uint32_t chunk_start = blockIdx.y * kCK;
  if (chunk_start >= keys_for_query) return;  // fully-masked chunk-block
  const uint32_t chunk_len = min(kCK, keys_for_query - chunk_start);
  const uint32_t h = blockIdx.z, g = h / (nh / nkv);  // GQA: kv head of q-head h
  const uint32_t kvd = nkv * HD, qkvd = (nh + 2 * nkv) * HD;
  const float* k_base = kcache + (size_t)chunk_start * kvd + g * HD;
  const float* v_base = vcache + (size_t)chunk_start * kvd + g * HD;

  float vec[HD];  // q during the dot phase, reused as acc in the value phase
  const float* q_row = qkv + (size_t)query * qkvd + h * HD;
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) vec[i] = q_row[i];
  float chunk_max = -INFINITY;
  for (uint32_t delta_t = 0; delta_t < chunk_len; ++delta_t) {
    const float* k_row = k_base + (size_t)delta_t * kvd;  // broadcast
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < HD; i += 4) {
      d0 += vec[i] * k_row[i];
      d1 += vec[i + 1] * k_row[i + 1];
      d2 += vec[i + 2] * k_row[i + 2];
      d3 += vec[i + 3] * k_row[i + 3];
    }
    const float score = ((d0 + d1) + (d2 + d3)) * scale;
    scores[64 * delta_t + threadIdx.x] = score;
    chunk_max = fmaxf(chunk_max, score);
  }

#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) vec[i] = 0.0f;  // q is dead; vec becomes acc
  float sum_exp = 0.0f;
  for (uint32_t delta_t = 0; delta_t < chunk_len; ++delta_t) {
    const float p = expf(scores[64 * delta_t + threadIdx.x] - chunk_max);
    sum_exp += p;
    const float* v_row = v_base + (size_t)delta_t * kvd;  // broadcast
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) vec[i] += p * v_row[i];
  }

  float* out =
      partials + (((size_t)h * S + query) * max_chunks + blockIdx.y) * (HD + 2);
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) out[i] = vec[i];
  out[HD] = chunk_max;
  out[HD + 1] = sum_exp;
}

// Combine: one block per (query, q-head), lane i owns output element i.
// Merge the chunk partials with the max-rescale rule, divide once, bf16 out.
template <uint32_t HD>
__global__ void k_prefill_combine(const float* __restrict__ partials,
                                  unsigned short* __restrict__ attn, uint32_t pos0,
                                  uint32_t S, uint32_t nh, uint32_t max_chunks) {
  const uint32_t s = blockIdx.x, h = blockIdx.y;
  const uint32_t num_chunks = (pos0 + s + 1 + kCK - 1) / kCK;
  const float* base = partials + ((size_t)h * S + s) * max_chunks * (HD + 2);
  float max_score = -INFINITY, total_exp = 0.0f;
  for (uint32_t c = 0; c < num_chunks; ++c)
    max_score = fmaxf(max_score, base[c * (HD + 2) + HD]);
  for (uint32_t c = 0; c < num_chunks; ++c)
    total_exp += base[c * (HD + 2) + HD + 1] * expf(base[c * (HD + 2) + HD] - max_score);
  const uint32_t i = threadIdx.x;
  if (i < HD) {
    float acc = 0.0f;
    for (uint32_t c = 0; c < num_chunks; ++c)
      acc += base[c * (HD + 2) + i] * expf(base[c * (HD + 2) + HD] - max_score);
    attn[(size_t)s * nh * HD + h * HD + i] = f2b(acc / total_exp);
  }
}

// Truncate an fp32 value to bf16 precision, keeping it as fp32 (bit-
// identical to f32_to_bf16 followed by widening: the GEMM operand-
// quantization contract, fused into the GEMV's x loads).
__device__ __forceinline__ float qbf16(float f) {
  return __uint_as_float(__float_as_uint(f) & 0xffff0000u);
}

// S=1 split-K GEMV, phase 1 (design §15 dispatch; measured in
// research/decode-roofline): one warp per output row, 8 warps per block,
// striding the row's K-chunk with uint64 weight loads (4 bf16 per lane per
// step). fp32 accumulation; one partial per (chunk, row) to global memory.
template <bool kXF32>
__global__ void k_gemv_partial(const unsigned short* __restrict__ W,
                               const void* __restrict__ x, float* __restrict__ partials,
                               uint32_t N, uint32_t K, uint32_t CK) {
  const uint32_t n = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (n >= N) return;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t k0 = blockIdx.y * CK, k1 = min(K, k0 + CK);
  const uint64_t* Wr = reinterpret_cast<const uint64_t*>(W + (size_t)n * K);
  float acc = 0.0f;
  if (kXF32) {  // fp32 stream input: quantize in-register (qbf16)
    const float4* xr = reinterpret_cast<const float4*>(x);
    for (uint32_t p = (k0 >> 2) + lane; p < (k1 >> 2); p += 32) {
      const uint64_t wv = Wr[p];
      const float4 xv = xr[p];
      acc += b2f((unsigned short)(wv & 0xffff)) * qbf16(xv.x);
      acc += b2f((unsigned short)((wv >> 16) & 0xffff)) * qbf16(xv.y);
      acc += b2f((unsigned short)((wv >> 32) & 0xffff)) * qbf16(xv.z);
      acc += b2f((unsigned short)(wv >> 48)) * qbf16(xv.w);
    }
  } else {  // bf16 input (attention/activation output): uint64 pairs
    const uint64_t* xr = reinterpret_cast<const uint64_t*>(x);
    for (uint32_t p = (k0 >> 2) + lane; p < (k1 >> 2); p += 32) {
      const uint64_t wv = Wr[p], xv = xr[p];
      acc += b2f((unsigned short)(wv & 0xffff)) * b2f((unsigned short)(xv & 0xffff));
      acc += b2f((unsigned short)((wv >> 16) & 0xffff)) *
             b2f((unsigned short)((xv >> 16) & 0xffff));
      acc += b2f((unsigned short)((wv >> 32) & 0xffff)) *
             b2f((unsigned short)((xv >> 32) & 0xffff));
      acc += b2f((unsigned short)(wv >> 48)) * b2f((unsigned short)(xv >> 48));
    }
  }
  for (uint32_t off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffffffffffULL, acc, off);
  if (lane == 0) partials[(size_t)blockIdx.y * N + n] = acc;
}

// Phase 2: y[n] = sum of the chunk partials. One thread per output.
__global__ void k_gemv_combine(const float* __restrict__ partials, float* __restrict__ y,
                               uint32_t N, uint32_t chunks) {
  const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float a = 0.0f;
  for (uint32_t c = 0; c < chunks; ++c) a += partials[(size_t)c * N + n];
  y[n] = a;
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
               float* split_scratch, float* prefill_scratch, uint32_t pos0, uint32_t S,
               uint32_t nh, uint32_t nkv, uint32_t hd, float scale) {
  const uint32_t T = pos0 + S;
  if (S > kAttnSplitMaxS) {
    // v4 split-T prefill (throughput-bound regime): fully independent
    // (query-tile, key-chunk, head) blocks + combine on the shared scratch.
    const uint32_t max_chunks = (T + kPrefillChunkKeys - 1) / kPrefillChunkKeys;
    const dim3 g((S + 63) / 64, max_chunks, nh);
    const dim3 gc(S, nh);
    if (hd == 64) {
      k_prefill_split<64><<<g, 64>>>(qkv, kcache, vcache, prefill_scratch, pos0, S, nh,
                                     nkv, scale, max_chunks);
      k_prefill_combine<64><<<gc, 128>>>((const float*)prefill_scratch,
                                         (unsigned short*)attn_bf16, pos0, S, nh, max_chunks);
    } else if (hd == 128) {
      k_prefill_split<128><<<g, 64>>>(qkv, kcache, vcache, prefill_scratch, pos0, S, nh,
                                      nkv, scale, max_chunks);
      k_prefill_combine<128><<<gc, 128>>>((const float*)prefill_scratch,
                                          (unsigned short*)attn_bf16, pos0, S, nh,
                                          max_chunks);
    } else {
      throw std::runtime_error(
          "gpu_rocm::kernels::attention: v4 prefill requires head_dim in {64, 128}");
    }
  } else {
    // v2 flash-decode (all T): the machine is latency-bound here, so split
    // the cache rows and merge partials. Measured to beat v1 at every
    // decode shape except T=1 (by ~3us of launch overhead).
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
  }
}

uint32_t gemv_chunks(uint32_t N, uint32_t K) {
  // CTA target: enough (row-block, chunk) blocks to occupy the GCD. Swept
  // over the 8B shapes in research/decode-roofline ([6] section): 1024 ->
  // 13.37 ms/token floor, 2048 -> 13.37, 4096 -> 12.99 (kept), 8192 -> 14.44
  // (gup regresses past 2 chunks). Chunk count clamped so each chunk stays
  // >= ~1024 wide (splitting thinner only adds partials traffic).
  constexpr uint32_t kTargetCTAs = 4096;
  const uint32_t row_blocks = (N + 7) / 8;
  uint32_t chunks = (kTargetCTAs + row_blocks - 1) / row_blocks;
  const uint32_t max_chunks = K >= 1024 ? K / 1024 : 1;
  if (chunks < 1) chunks = 1;
  if (chunks > max_chunks) chunks = max_chunks;
  return chunks;
}

namespace {
void gemv_launch(const void* W, const void* x, float* y, float* partials, uint32_t N,
                 uint32_t K, bool x_f32) {
  if (N == 0 || K == 0) throw std::runtime_error("gpu_rocm::kernels::gemv: empty shape");
  if (K % 4)
    throw std::runtime_error(
        "gpu_rocm::kernels::gemv: K must be a multiple of 4 (uint64/float4 loads)");
  const uint32_t chunks = gemv_chunks(N, K);
  const uint32_t CK = (((K + chunks - 1) / chunks) + 3) & ~3u;  // multiple of 4
  const dim3 grid((N + 7) / 8, chunks);
  if (x_f32)
    k_gemv_partial<true><<<grid, 256>>>((const unsigned short*)W, x, partials, N, K, CK);
  else
    k_gemv_partial<false><<<grid, 256>>>((const unsigned short*)W, x, partials, N, K,
                                         CK);
  k_gemv_combine<<<grid_for(N, 256), 256>>>(partials, y, N, chunks);
}
}  // namespace

void gemv_x_f32(const void* W, const float* x, float* y, float* partials, uint32_t N,
                uint32_t K) {
  gemv_launch(W, x, y, partials, N, K, true);
}

void gemv_x_bf16(const void* W, const void* x, float* y, float* partials, uint32_t N,
                 uint32_t K) {
  gemv_launch(W, x, y, partials, N, K, false);
}

}  // namespace alice::gpu_rocm::kernels
