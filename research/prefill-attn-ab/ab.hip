// Prefill attention scheme A/B experiment (research harness, NOT engine code).
//
// Single-head-style causal attention kernels over fp32 Q/K/V:
//   query s of S attends keys [0, T-S+s]; scale = 1/sqrt(hd).
//   grid.y = head h (H heads for machine-fill during timing), blockIdx.x =
//   query (v1) or query tile (A, B).
//
//   v1: engine baseline. One block per (query, head), 128 threads,
//       lane-per-element, block reduction per key row, online softmax.
//   v3A:  streaming single-pass, thread-per-query. q in registers, acc in
//       registers (hd=64) or LDS (hd=128: q+acc can't share the register
//       file), online rescale per key. No chunk structure, no syncthreads.
//   v3B:  two-phase per 64-key chunk (user's scheme). Raw score tile in LDS
//       during the K phase (q row staged in regs from an LDS q tile),
//       deferred exp(score - chunk_max) in the V phase (acc row staged in
//       regs from an LDS acc tile) — phase-separated registers, one online
//       rescale per chunk. Score rows are thread-private: no per-chunk
//       __syncthreads.
//   v4:  split-T prefill (FA-2-style split-KV). Each block owns ONE key
//       chunk for a query tile (thread-per-query): raw scores to an LDS tile
//       in key-major layout (bank-conflict-free in both phases), deferred exp
//       with the single-chunk max, partials (acc, max, sum-exp) to GLOBAL
//       memory, merged by a small combine kernel. No staged q/acc tiles, no
//       running softmax, no __syncthreads, 16 KB LDS -> 4 blocks/CU.
//
// Query-tile sizes are per-scheme per-hd best-occupancy picks (documented in
// README): A64=64, A128=32, B64=32, B128=16.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#define HIP_CHECK(x)                                                                     \
  do {                                                                                   \
    hipError_t e_ = (x);                                                                 \
    if (e_ != hipSuccess) {                                                              \
      std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__,  \
                   __LINE__);                                                            \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

namespace {

// ---------------------------------------------------------------- v1 (engine baseline)

template <unsigned NT>
__device__ float blk_sum(float v, float* shared) {
  shared[threadIdx.x] = v;
  __syncthreads();
  for (unsigned s = NT / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  return shared[0];
}

template <uint32_t HD>
__global__ void k_v1(const float* __restrict__ q, const float* __restrict__ kc,
                     const float* __restrict__ vc, float* __restrict__ out, uint32_t S,
                     uint32_t T, float scale) {
  __shared__ float shared[128];
  const uint32_t s = blockIdx.x;
  const float* qh = q + ((size_t)blockIdx.y * S + s) * HD;
  const float* kh = kc + (size_t)blockIdx.y * T * HD;
  const float* vh = vc + (size_t)blockIdx.y * T * HD;
  const uint32_t Tk = T - S + s + 1;
  const uint32_t i = threadIdx.x;
  const float q_i = i < HD ? qh[i] : 0.0f;
  float m = -INFINITY, l = 0.0f, acc = 0.0f;
  for (uint32_t t = 0; t < Tk; ++t) {
    const float part = i < HD ? q_i * kh[(size_t)t * HD + i] : 0.0f;
    const float dot = blk_sum<128>(part, shared);
    const float sc = dot * scale;
    const float nm = fmaxf(m, sc);
    const float f = expf(m - nm), p = expf(sc - nm);
    l = l * f + p;
    if (i < HD) acc = acc * f + p * vh[(size_t)t * HD + i];
    m = nm;
  }
  if (i < HD) out[((size_t)blockIdx.y * S + s) * HD + i] = acc / l;
}

// ---------------------------------------------------------------- scheme v3A (streaming)

template <uint32_t HD, uint32_t QT, bool ACC_LDS>
__global__ void k_v3a(const float* __restrict__ q, const float* __restrict__ kc,
                    const float* __restrict__ vc, float* __restrict__ out, uint32_t S,
                    uint32_t T, float scale) {
  __shared__ float sacc[ACC_LDS ? QT * HD : 1];
  if (threadIdx.x >= QT) return;  // blocks are 64 threads; QT may be smaller
  const uint32_t s = blockIdx.x * QT + threadIdx.x;
  const bool live = s < S;
  const uint32_t Tk = live ? (T - S + s + 1) : 0;
  const float* kh = kc + (size_t)blockIdx.y * T * HD;
  const float* vh = vc + (size_t)blockIdx.y * T * HD;

  float q_r[HD];
  const float* qrow = q + ((size_t)blockIdx.y * S + (live ? s : 0)) * HD;
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) q_r[i] = live ? qrow[i] : 0.0f;
  float acc_r[ACC_LDS ? 1 : HD];
  if constexpr (ACC_LDS) {
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) sacc[threadIdx.x * HD + i] = 0.0f;
  } else {
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) acc_r[i] = 0.0f;
  }

  float m = -INFINITY, l = 0.0f;
  for (uint32_t t = 0; t < Tk; ++t) {
    const float* krow = kh + (size_t)t * HD;  // broadcast: all lanes same address
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < HD; i += 4) {
      d0 += q_r[i] * krow[i];
      d1 += q_r[i + 1] * krow[i + 1];
      d2 += q_r[i + 2] * krow[i + 2];
      d3 += q_r[i + 3] * krow[i + 3];
    }
    const float sc = ((d0 + d1) + (d2 + d3)) * scale;
    const float nm = fmaxf(m, sc);
    const float f = expf(m - nm), p = expf(sc - nm);
    l = l * f + p;
    const float* vrow = vh + (size_t)t * HD;  // broadcast
    if constexpr (ACC_LDS) {
#pragma unroll
      for (uint32_t i = 0; i < HD; ++i)
        sacc[threadIdx.x * HD + i] = sacc[threadIdx.x * HD + i] * f + p * vrow[i];
    } else {
#pragma unroll
      for (uint32_t i = 0; i < HD; ++i) acc_r[i] = acc_r[i] * f + p * vrow[i];
    }
    m = nm;
  }
  if (!live) return;
  float* orow = out + ((size_t)blockIdx.y * S + s) * HD;
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i)
    orow[i] = (ACC_LDS ? sacc[threadIdx.x * HD + i] : acc_r[i]) / l;
}

// Scheme v3A2 (bonus variant for hd=128): q tile in LDS, acc in registers.
// Halves the per-key LDS traffic of v3A's hd=128 form (q reads instead of
// acc read+write); acc 128 regs fits since q is not co-resident.
template <uint32_t HD, uint32_t QT>
__global__ void k_v3a2(const float* __restrict__ q, const float* __restrict__ kc,
                     const float* __restrict__ vc, float* __restrict__ out, uint32_t S,
                     uint32_t T, float scale) {
  __shared__ float sq[QT * HD];
  for (uint32_t idx = threadIdx.x; idx < QT * HD; idx += 64) {
    const uint32_t r = idx / HD, c = idx % HD;
    const uint32_t gs = blockIdx.x * QT + r;
    sq[idx] = gs < S ? q[((size_t)blockIdx.y * S + gs) * HD + c] : 0.0f;
  }
  __syncthreads();  // the only one in the kernel
  if (threadIdx.x >= QT) return;
  const uint32_t s = blockIdx.x * QT + threadIdx.x;
  const bool live = s < S;
  const uint32_t Tk = live ? (T - S + s + 1) : 0;
  const float* kh = kc + (size_t)blockIdx.y * T * HD;
  const float* vh = vc + (size_t)blockIdx.y * T * HD;

  float acc[HD];
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) acc[i] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  for (uint32_t t = 0; t < Tk; ++t) {
    const float* krow = kh + (size_t)t * HD;  // broadcast
    const float* qrow = sq + threadIdx.x * HD;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < HD; i += 4) {
      d0 += qrow[i] * krow[i];
      d1 += qrow[i + 1] * krow[i + 1];
      d2 += qrow[i + 2] * krow[i + 2];
      d3 += qrow[i + 3] * krow[i + 3];
    }
    const float sc = ((d0 + d1) + (d2 + d3)) * scale;
    const float nm = fmaxf(m, sc);
    const float f = expf(m - nm), p = expf(sc - nm);
    l = l * f + p;
    const float* vrow = vh + (size_t)t * HD;  // broadcast
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) acc[i] = acc[i] * f + p * vrow[i];
    m = nm;
  }
  if (!live) return;
  float* orow = out + ((size_t)blockIdx.y * S + s) * HD;
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) orow[i] = acc[i] / l;
}

// ---------------------------------------------------------------- scheme v3B (two-phase)

template <uint32_t HD, uint32_t QT, uint32_t CK>
__global__ void k_v3b(const float* __restrict__ q, const float* __restrict__ kc,
                    const float* __restrict__ vc, float* __restrict__ out, uint32_t S,
                    uint32_t T, float scale) {
  __shared__ float sq[QT * HD];   // q tile, staged once (coalesced)
  __shared__ float ssc[QT * CK];  // raw score tile (thread-private rows)
  __shared__ float sac[QT * HD];  // acc tile (persistent; regs only inside V phase)
  const uint32_t s = blockIdx.x * QT + threadIdx.x;
  const bool live = s < S;
  const uint32_t Tk = live ? (T - S + s + 1) : 0;
  const float* kh = kc + (size_t)blockIdx.y * T * HD;
  const float* vh = vc + (size_t)blockIdx.y * T * HD;

  for (uint32_t idx = threadIdx.x; idx < QT * HD; idx += 64) {
    const uint32_t r = idx / HD, c = idx % HD;
    const uint32_t gs = blockIdx.x * QT + r;
    sq[idx] = gs < S ? q[((size_t)blockIdx.y * S + gs) * HD + c] : 0.0f;
    sac[idx] = 0.0f;
  }
  __syncthreads();  // the only one in the kernel
  if (threadIdx.x >= QT) return;  // after the barrier: per-thread LDS rows are QT-sized

  float m = -INFINITY, l = 0.0f;
  for (uint32_t c0 = 0; c0 < Tk; c0 += CK) {
    const uint32_t c1 = min(Tk, c0 + CK);
    // K phase: q row into registers, dots -> score tile, track chunk max.
    float qr[HD];
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) qr[i] = sq[threadIdx.x * HD + i];
    float mc = -INFINITY;
    for (uint32_t t = c0; t < c1; ++t) {
      const float* krow = kh + (size_t)t * HD;  // broadcast
      float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
      for (uint32_t i = 0; i < HD; i += 4) {
        d0 += qr[i] * krow[i];
        d1 += qr[i + 1] * krow[i + 1];
        d2 += qr[i + 2] * krow[i + 2];
        d3 += qr[i + 3] * krow[i + 3];
      }
      const float sc = ((d0 + d1) + (d2 + d3)) * scale;
      ssc[threadIdx.x * CK + (t - c0)] = sc;
      mc = fmaxf(mc, sc);
    }
    // Merge: rescale history into the new max's units (first chunk: f=0).
    const float M = fmaxf(m, mc);
    const float f = expf(m - M);
    l *= f;
    // V phase: acc row into registers, deferred exp, accumulate, store back.
    float ar[HD];
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) ar[i] = sac[threadIdx.x * HD + i] * f;
    for (uint32_t t = c0; t < c1; ++t) {
      const float p = expf(ssc[threadIdx.x * CK + (t - c0)] - M);
      l += p;
      const float* vrow = vh + (size_t)t * HD;  // broadcast
#pragma unroll
      for (uint32_t i = 0; i < HD; ++i) ar[i] += p * vrow[i];
    }
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) sac[threadIdx.x * HD + i] = ar[i];
    m = M;
  }
  if (!live) return;
  float* orow = out + ((size_t)blockIdx.y * S + s) * HD;
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) orow[i] = sac[threadIdx.x * HD + i] / l;
}

// ---------------------------------------------------------------- v2 (engine flash-decode)

// Engine decode kernel, ported for decode-shape timing: split-T with
// in-warp reductions — one block per (split, query, head), 4 warps of 64
// lanes stride the split's rows (lane-per-element dots, __shfl reduction),
// per-split partials merged by a combine kernel. fp32 out (engine: bf16).
template <uint32_t HD>
__global__ void k_v2_split(const float* __restrict__ q, const float* __restrict__ kc,
                           const float* __restrict__ vc, float* __restrict__ partials,
                           uint32_t S, uint32_t T, float scale, uint32_t rows_per_split) {
  __shared__ float warp_max[4], warp_sum[4], warp_acc[4][HD];
  const uint32_t query = blockIdx.y;
  const uint32_t keys_for_query = T - S + query + 1;
  const uint32_t split = blockIdx.x;
  const uint32_t row_begin = split * rows_per_split;
  const uint32_t row_end = min(keys_for_query, row_begin + rows_per_split);
  const float* q_head = q + ((size_t)blockIdx.z * S + query) * HD;
  const float* k_head = kc + (size_t)blockIdx.z * T * HD;
  const float* v_head = vc + (size_t)blockIdx.z * T * HD;
  const uint32_t lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
  constexpr uint32_t EPL = (HD + 63) / 64;  // elements per lane (hd <= 128)

  float q_reg[EPL];
  for (uint32_t e = 0; e < EPL; ++e) {
    const uint32_t i = lane + e * 64;
    q_reg[e] = i < HD ? q_head[i] : 0.0f;
  }
  float m = -INFINITY, l = 0.0f, acc[EPL] = {0.0f};
  for (uint32_t t = row_begin + warp; t < row_end; t += 4) {
    const float* k_row = k_head + (size_t)t * HD;
    float part = 0.0f;
    for (uint32_t e = 0; e < EPL; ++e) {
      const uint32_t i = lane + e * 64;
      if (i < HD) part += q_reg[e] * k_row[i];
    }
    for (uint32_t off = 32; off > 0; off >>= 1)
      part += __shfl_down_sync(0xffffffffffffffffULL, part, off);
    const float dot = __shfl_sync(0xffffffffffffffffULL, part, 0);
    const float sc = dot * scale;
    const float nm = fmaxf(m, sc);
    const float f = expf(m - nm), p = expf(sc - nm);
    l = l * f + p;
    const float* v_row = v_head + (size_t)t * HD;
    for (uint32_t e = 0; e < EPL; ++e) {
      const uint32_t i = lane + e * 64;
      if (i < HD) acc[e] = acc[e] * f + p * v_row[i];
    }
    m = nm;
  }

  if (lane == 0) {
    warp_max[warp] = m;
    warp_sum[warp] = l;
  }
  for (uint32_t e = 0; e < EPL; ++e) {
    const uint32_t i = lane + e * 64;
    if (i < HD) warp_acc[warp][i] = acc[e];
  }
  __syncthreads();
  if (warp != 0) return;
  const float M = fmaxf(fmaxf(warp_max[0], warp_max[1]), fmaxf(warp_max[2], warp_max[3]));
  float* out = partials + (((size_t)blockIdx.z * S + query) * gridDim.x + split) * (HD + 2);
  if (M == -INFINITY) {  // empty split
    if (lane == 0) out[HD + 1] = 0.0f;
    return;
  }
  float w[4], L = 0.0f;
  for (uint32_t j = 0; j < 4; ++j) {
    w[j] = expf(warp_max[j] - M);
    L += warp_sum[j] * w[j];
  }
  for (uint32_t e = 0; e < EPL; ++e) {
    const uint32_t i = lane + e * 64;
    if (i < HD)
      out[i] = warp_acc[0][i] * w[0] + warp_acc[1][i] * w[1] + warp_acc[2][i] * w[2] +
               warp_acc[3][i] * w[3];
  }
  if (lane == 0) {
    out[HD] = M;
    out[HD + 1] = L;
  }
}

template <uint32_t HD>
__global__ void k_v2_combine(const float* __restrict__ partials, float* __restrict__ out,
                             uint32_t S, uint32_t splits) {
  const uint32_t query = blockIdx.x;
  const float* base = partials + ((size_t)blockIdx.y * S + query) * splits * (HD + 2);
  float M = -INFINITY, L = 0.0f;
  for (uint32_t sp = 0; sp < splits; ++sp) {
    const float l = base[sp * (HD + 2) + HD + 1];
    if (l > 0.0f) M = fmaxf(M, base[sp * (HD + 2) + HD]);
  }
  for (uint32_t sp = 0; sp < splits; ++sp) {
    const float l = base[sp * (HD + 2) + HD + 1];
    if (l > 0.0f) L += l * expf(base[sp * (HD + 2) + HD] - M);
  }
  const uint32_t i = threadIdx.x;
  if (i < HD) {
    float a = 0.0f;
    for (uint32_t sp = 0; sp < splits; ++sp) {
      const float l = base[sp * (HD + 2) + HD + 1];
      if (l > 0.0f) a += base[sp * (HD + 2) + i] * expf(base[sp * (HD + 2) + HD] - M);
    }
    out[((size_t)blockIdx.y * S + query) * HD + i] = a / L;
  }
}

// ---------------------------------------------------------------- v4 (split-T prefill)

// Main kernel: one block per (query-tile, key-chunk, head); one thread per
// query. Each block computes its chunk's partial softmax output with no
// history at all: raw dots into the LDS score tile (key-major layout:
// consecutive threads hit consecutive banks in both phases), one deferred
// exp pass once the chunk max is known, then the partial (acc, max, sum_exp)
// goes to global memory. Fully-masked chunk-blocks (causality) return before
// touching K/V. Zero __syncthreads: every score row is thread-private.
template <uint32_t HD, uint32_t QT, uint32_t CK>
__global__ void __launch_bounds__(64, 4)  // 256-VGPR budget: no acc scratch spill
    k_v4(const float* __restrict__ q, const float* __restrict__ kc,
                     const float* __restrict__ vc, float* __restrict__ partials,
                     uint32_t S, uint32_t T, float scale, uint32_t max_chunks) {
  __shared__ float scores[CK * QT];  // key-major: scores[QT * key + thread]
  const uint32_t query = blockIdx.x * QT + threadIdx.x;
  if (query >= S) return;
  const uint32_t keys_for_query = T - S + query + 1;  // causal bound (exclusive)
  const uint32_t chunk_start = blockIdx.y * CK;
  if (chunk_start >= keys_for_query) return;  // fully-masked chunk-block
  const uint32_t chunk_end = min(chunk_start + CK, keys_for_query);
  const float* k_head = kc + (size_t)blockIdx.z * T * HD;
  const float* v_head = vc + (size_t)blockIdx.z * T * HD;

  // Dot phase: q row into registers, stream K rows (all lanes on the same
  // key -> broadcast), write raw scores to the LDS tile, track the chunk max.
  float q_reg[HD];
  const float* q_row = q + ((size_t)blockIdx.z * S + query) * HD;
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) q_reg[i] = q_row[i];
  float chunk_max = -INFINITY;
  for (uint32_t t = chunk_start; t < chunk_end; ++t) {
    const float* k_row = k_head + (size_t)t * HD;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < HD; i += 4) {
      d0 += q_reg[i] * k_row[i];
      d1 += q_reg[i + 1] * k_row[i + 1];
      d2 += q_reg[i + 2] * k_row[i + 2];
      d3 += q_reg[i + 3] * k_row[i + 3];
    }
    const float score = ((d0 + d1) + (d2 + d3)) * scale;
    scores[QT * (t - chunk_start) + threadIdx.x] = score;
    chunk_max = fmaxf(chunk_max, score);
  }

  // Value phase: q registers are dead (phase-separated register file).
  // Deferred exp — every argument <= 0 because the chunk max is known.
  float acc[HD];
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) acc[i] = 0.0f;
  float sum_exp = 0.0f;
  for (uint32_t t = chunk_start; t < chunk_end; ++t) {
    const float p = expf(scores[QT * (t - chunk_start) + threadIdx.x] - chunk_max);
    sum_exp += p;
    const float* v_row = v_head + (size_t)t * HD;  // broadcast
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) acc[i] += p * v_row[i];
  }

  // Partial: [acc(HD), chunk_max, sum_exp]. The combine only reads the
  // chunks valid for each query, so unwritten slots here are never touched.
  float* out =
      partials + (((size_t)blockIdx.z * S + query) * max_chunks + blockIdx.y) * (HD + 2);
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) out[i] = acc[i];
  out[HD] = chunk_max;
  out[HD + 1] = sum_exp;
}

// v4R variant (user's idea): score tile in per-thread REGISTERS instead of
// LDS. scores[CK] is wavefront-uniform-indexed (all lanes iterate the same
// delta_t), the case CDNA2 GPR index mode covers; whether LLVM selects it
// or spills to local scratch is verified via the resource report (see
// README). Peak register pressure: HD + CK + helpers ~= 208 (hd=128) -> 4
// wavefronts/CU; ~150 (hd=64) -> up to 6 blocks/CU (v4-LDS was LDS-capped
// at 4). Zero LDS, zero bank questions.
template <uint32_t HD, uint32_t QT, uint32_t CK>
__global__ void __launch_bounds__(64, 4)
    k_v4r(const float* __restrict__ q, const float* __restrict__ kc,
          const float* __restrict__ vc, float* __restrict__ partials, uint32_t S,
          uint32_t T, float scale, uint32_t max_chunks) {
  const uint32_t query = blockIdx.x * QT + threadIdx.x;
  if (query >= S) return;
  const uint32_t keys_for_query = T - S + query + 1;
  const uint32_t chunk_start = blockIdx.y * CK;
  if (chunk_start >= keys_for_query) return;
  const uint32_t chunk_len = min(CK, keys_for_query - chunk_start);
  const float* k_base = kc + ((size_t)blockIdx.z * T + chunk_start) * HD;
  const float* v_base = vc + ((size_t)blockIdx.z * T + chunk_start) * HD;

  float scores[CK];
  float vec[HD];  // q during the dot phase, reused as acc in the value phase
  const float* q_row = q + ((size_t)blockIdx.z * S + query) * HD;
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) vec[i] = q_row[i];
  // Rolled loop, runtime bound, no unroll pragma: delta_t is a
  // wavefront-uniform scalar, so scores[delta_t] is a candidate for
  // CDNA2 GPR index mode (s_set_gpr_idx_on). Verified in the disassembly.
  float chunk_max = -INFINITY;
  for (uint32_t delta_t = 0; delta_t < chunk_len; ++delta_t) {
    const float* k_row = k_base + (size_t)delta_t * HD;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < HD; i += 4) {
      d0 += vec[i] * k_row[i];
      d1 += vec[i + 1] * k_row[i + 1];
      d2 += vec[i + 2] * k_row[i + 2];
      d3 += vec[i + 3] * k_row[i + 3];
    }
    const float score = ((d0 + d1) + (d2 + d3)) * scale;
    scores[delta_t] = score;
    chunk_max = fmaxf(chunk_max, score);
  }

#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) vec[i] = 0.0f;  // q is dead; vec becomes acc
  float sum_exp = 0.0f;
  for (uint32_t delta_t = 0; delta_t < chunk_len; ++delta_t) {
    const float p = expf(scores[delta_t] - chunk_max);
    sum_exp += p;
    const float* v_row = v_base + (size_t)delta_t * HD;
#pragma unroll
    for (uint32_t i = 0; i < HD; ++i) vec[i] += p * v_row[i];
  }

  float* out =
      partials + (((size_t)blockIdx.z * S + query) * max_chunks + blockIdx.y) * (HD + 2);
#pragma unroll
  for (uint32_t i = 0; i < HD; ++i) out[i] = vec[i];
  out[HD] = chunk_max;
  out[HD + 1] = sum_exp;
}

// Combine: one block per (query, head); lane i owns output element i. Merge
// the chunk partials with the usual max-rescale rule, divide once at the end.
template <uint32_t HD, uint32_t CK>
__global__ void k_v4_combine(const float* __restrict__ partials, float* __restrict__ out,
                             uint32_t S, uint32_t T, uint32_t max_chunks) {
  const uint32_t query = blockIdx.x;
  const uint32_t num_chunks = (T - S + query + 1 + CK - 1) / CK;  // causal-valid chunks
  const float* base = partials + ((size_t)blockIdx.y * S + query) * max_chunks * (HD + 2);
  float max_score = -INFINITY;
  for (uint32_t c = 0; c < num_chunks; ++c)
    max_score = fmaxf(max_score, base[c * (HD + 2) + HD]);
  float total_exp = 0.0f;
  for (uint32_t c = 0; c < num_chunks; ++c)
    total_exp += base[c * (HD + 2) + HD + 1] * expf(base[c * (HD + 2) + HD] - max_score);
  const uint32_t i = threadIdx.x;
  if (i < HD) {
    float acc = 0.0f;
    for (uint32_t c = 0; c < num_chunks; ++c)
      acc += base[c * (HD + 2) + i] * expf(base[c * (HD + 2) + HD] - max_score);
    out[((size_t)blockIdx.y * S + query) * HD + i] = acc / total_exp;
  }
}

// ---------------------------------------------------------------- host harness

struct Case {
  uint32_t H, S, T, hd;
  std::vector<float> q, k, v, ref;
};

Case load_case(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  char magic[4];
  Case c;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "ATT1", 4) != 0 ||
      std::fread(&c.H, 4, 1, f) != 1 || std::fread(&c.S, 4, 1, f) != 1 ||
      std::fread(&c.T, 4, 1, f) != 1 || std::fread(&c.hd, 4, 1, f) != 1) {
    std::fprintf(stderr, "bad case file %s\n", path);
    std::exit(1);
  }
  auto read = [&](size_t n) {
    std::vector<float> x(n);
    if (std::fread(x.data(), 4, n, f) != n) {
      std::fprintf(stderr, "truncated case file %s\n", path);
      std::exit(1);
    }
    return x;
  };
  c.q = read((size_t)c.H * c.S * c.hd);
  c.k = read((size_t)c.H * c.T * c.hd);
  c.v = read((size_t)c.H * c.T * c.hd);
  c.ref = read((size_t)c.H * c.S * c.hd);
  std::fclose(f);
  return c;
}

using Launch = void (*)(const float*, const float*, const float*, float*, uint32_t,
                        uint32_t, uint32_t, float);

template <uint32_t HD>
void launch_one(const char* scheme, const Case& c, const float* dq, const float* dk,
                const float* dv, float* dout, float* dpart, uint32_t max_chunks) {
  const float scale = 1.0f / sqrtf(float(HD));
  const dim3 g1(c.S, c.H);
  if (std::strcmp(scheme, "v1") == 0) {
    k_v1<HD><<<g1, 128>>>(dq, dk, dv, dout, c.S, c.T, scale);
  } else if (std::strcmp(scheme, "v3A") == 0) {
    if constexpr (HD == 64) {
      const dim3 g((c.S + 63) / 64, c.H);
      k_v3a<64, 64, false><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
    } else {
      const dim3 g((c.S + 63) / 64, c.H);  // sacc 32KB -> 2 blocks/CU, full wavefront
      k_v3a<128, 64, true><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
    }
  } else if (std::strcmp(scheme, "v3A2") == 0) {
    const dim3 g((c.S + 63) / 64, c.H);  // sq 32KB (hd=128) -> 2 blocks/CU
    if constexpr (HD == 64)
      k_v3a2<64, 64><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
    else
      k_v3a2<128, 64><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
  } else if (std::strcmp(scheme, "v3B") == 0) {
    if constexpr (HD == 64) {
      const dim3 g((c.S + 63) / 64, c.H);  // LDS 48KB -> 1 block/CU, full wavefront
      k_v3b<64, 64, 64><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
    } else {
      // LDS 40KB -> 1 block/CU, half wavefront: v3B's LDS appetite
      // (q+scores+acc) at hd=128 can't have full wavefronts AND occupancy.
      const dim3 g((c.S + 31) / 32, c.H);
      k_v3b<128, 32, 64><<<g, 64>>>(dq, dk, dv, dout, c.S, c.T, scale);
    }
  } else if (std::strcmp(scheme, "v2") == 0) {
    uint32_t splits = (c.T + 63) / 64, rows_per_split = 64;
    if (splits > 64) {  // keep full coverage of [0, T)
      splits = 64;
      rows_per_split = (c.T + splits - 1) / splits;
    }
    const dim3 g(splits, c.S, c.H);
    k_v2_split<HD><<<g, 256>>>(dq, dk, dv, dpart, c.S, c.T, scale, rows_per_split);
    const dim3 gc(c.S, c.H);
    k_v2_combine<HD><<<gc, 128>>>(dpart, dout, c.S, splits);
  } else {  // v4 / v4R
    const dim3 g((c.S + 63) / 64, max_chunks, c.H);
    if (std::strcmp(scheme, "v4R") == 0)
      k_v4r<HD, 64, 64><<<g, 64>>>(dq, dk, dv, dpart, c.S, c.T, scale, max_chunks);
    else
      k_v4<HD, 64, 64><<<g, 64>>>(dq, dk, dv, dpart, c.S, c.T, scale, max_chunks);
    const dim3 gc(c.S, c.H);
    k_v4_combine<HD, 64><<<gc, 128>>>(dpart, dout, c.S, c.T, max_chunks);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <case.bin> [--bench N]\n", argv[0]);
    return 1;
  }
  const Case c = load_case(argv[1]);
  const bool write_out = c.H == 1;  // outs only needed for correctness cases
  float *dq, *dk, *dv, *dout;
  HIP_CHECK(hipMalloc(&dq, c.q.size() * 4));
  HIP_CHECK(hipMalloc(&dk, c.k.size() * 4));
  HIP_CHECK(hipMalloc(&dv, c.v.size() * 4));
  HIP_CHECK(hipMalloc(&dout, c.q.size() * 4));
  HIP_CHECK(hipMemcpy(dq, c.q.data(), c.q.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dk, c.k.data(), c.k.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dv, c.v.data(), c.v.size() * 4, hipMemcpyHostToDevice));

  // useful KV bytes actually read: H * sum_s Tk(s) * hd * 2 * 4
  const double kv_bytes =
      double(c.H) * (double(c.S) * (c.T - c.S / 2.0 + 0.5)) * c.hd * 2 * 4;

  const uint32_t max_chunks = (c.T + 63) / 64;
  float* dpart = nullptr;  // v4 partials: H*S*max_chunks*(hd+2) floats
  HIP_CHECK(hipMalloc(&dpart, (size_t)c.H * c.S * max_chunks * (c.hd + 2) * 4));

  for (const char* scheme : {"v1", "v2", "v3A", "v3A2", "v3B", "v4", "v4R"}) {
    auto run = [&] {
      if (c.hd == 64)
        launch_one<64>(scheme, c, dq, dk, dv, dout, dpart, max_chunks);
      else
        launch_one<128>(scheme, c, dq, dk, dv, dout, dpart, max_chunks);
    };
    run();  // correctness pass (also warms up)
    HIP_CHECK(hipDeviceSynchronize());
    if (write_out) {
      std::vector<float> host(c.q.size());
      HIP_CHECK(hipMemcpy(host.data(), dout, host.size() * 4, hipMemcpyDeviceToHost));
      std::string path = std::string(argv[1]) + ".out_" + scheme;
      FILE* f = std::fopen(path.c_str(), "wb");
      std::fwrite(host.data(), 4, host.size(), f);
      std::fclose(f);
    }
    // timing: adaptive iteration count targeting ~0.3 s per scheme
    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));
    HIP_CHECK(hipEventRecord(ev0));
    run();
    HIP_CHECK(hipEventRecord(ev1));
    HIP_CHECK(hipEventSynchronize(ev1));
    float est_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&est_ms, ev0, ev1));
    const int iters = est_ms <= 0.0f ? 3 : std::max(3, std::min(200, int(300.0f / est_ms)));
    HIP_CHECK(hipEventRecord(ev0));
    for (int i = 0; i < iters; ++i) run();
    HIP_CHECK(hipEventRecord(ev1));
    HIP_CHECK(hipEventSynchronize(ev1));
    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, ev0, ev1));
    const double ms = total_ms / iters;
    std::printf("hd=%-4uT=%-5uS=%-5uH=%-3u %-3s %8.3f ms  %7.1f GB/s KV\n", c.hd, c.T, c.S,
                c.H, scheme, ms, kv_bytes / ms / 1e6);
    HIP_CHECK(hipEventDestroy(ev0));
    HIP_CHECK(hipEventDestroy(ev1));
  }
  return 0;
}
