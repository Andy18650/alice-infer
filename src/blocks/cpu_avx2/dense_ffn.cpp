// CPU_AVX2 implementation of DenseFfnBF16 — the first self-contained
// optimized operator (design §10). Fully private weight storage and math
// (no shared toolbox); per-TU -mavx2 -mfma with cpuid-guarded dispatch.
//
// Structure:
//   * aligned 64B private storage with parallel first-touch (NUMA-local
//     residency without mbind: the thread that streams a page at forward()
//     time is, under static scheduling + same thread count, the one that
//     faulted it in at construction).
//   * S-dispatch (GEMV vs GEMM are different physics — llama.cpp does the
//     same: tinyBLAS refuses m<4 and falls back):
//       - S == 1 (decode): dot-form for all three projections, parallel
//         over output rows; every thread streams CONTIGUOUS weight rows.
//       - S > 1 (prefill): fused gate/up/down chunk loop (the [S, I]
//         intermediate never exists) whose chunk loop over I is the
//         K-blocking, with a tinyBLAS-style REGISTER-TILE microkernel
//         (gemm_bloc<RM,RN>, reference/sgemm.cpp): RM*RN independent dot
//         accumulators per tile hide FMA latency by construction and
//         reuse every loaded vector from registers (A row x RN, B row
//         x RM). Both operands are k-contiguous, so B (weights) is
//         consumed ROW BY ROW — output-tile parallelism partitions
//         weights by whole rows = contiguous per-thread streams. The
//         transposed w_down of the previous revision (and its strided
//         column-block pattern + 50% extra storage) is RETIRED: it was
//         a workaround for not having a real microkernel.
//   * private [S, d] output accumulator for the fused path (fusion cannot
//     accumulate into io while later chunks still read the input from
//     io); copied back at the end to fulfil the in-place contract.
//   * threading: one persistent omp parallel region, static omp for
//     throughout (no dynamic scheduling by decision).
//
// Numerics (user-signed-off, design §10): cross-impl equivalence is
// tolerance-based (chunked + lane-group accumulation reorder sums vs the
// oracle). Within-impl: deterministic per (S, thread count) — tiles are
// independent, chunk accumulation is fixed ascending order; row
// independence always holds; incremental==batched is TOLERANCE-based
// because the S==1 dot-form and S>1 tile paths sum in different orders.

#include "dense_ffn.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(_OPENMP)
#define ALICE_AVX2_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define ALICE_AVX2_OMP_PARALLEL _Pragma("omp parallel")
#define ALICE_AVX2_OMP_FOR _Pragma("omp for")
#else
#define ALICE_AVX2_OMP_PARALLEL_FOR
#define ALICE_AVX2_OMP_PARALLEL
#define ALICE_AVX2_OMP_FOR
#endif

namespace alice::cpu_avx2 {
namespace {

constexpr size_t kAlign = 64;     // cache line; AVX2 32B loads are a subset
constexpr uint32_t kChunk = 256;  // intermediate rows per fused chunk (K-block)

constexpr size_t round_up(size_t n, size_t m) { return (n + m - 1) / m * m; }

template <typename T>
AlignedPtr<T> alloc_aligned(size_t n) {
  // std::aligned_alloc requires size to be a multiple of the alignment.
  const size_t bytes = round_up(n * sizeof(T), kAlign);
  void* p = std::aligned_alloc(kAlign, bytes);
  if (!p) throw std::bad_alloc();
  return AlignedPtr<T>(static_cast<T*>(p));
}

// --- isolated inner kernels -------------------------------------------------
// Step 3: AVX2+FMA intrinsics (this TU is compiled with -mavx2 -mfma on x86;
// the scalar versions remain as the portable fallback and oracle-A/B).
//
// bf16 -> fp32 conversion is the shift trick: bf16 is the top 16 bits of
// fp32, so widen u16->u32 and shift left 16. Zen3 has no native bf16.
//
// NUMERICS NOTE: the multi-accumulator dot and the register tile both sum
// in lane-group order, not strict ascending order. Per (impl, S, thread-
// count) the schedule is fixed and deterministic, but the S==1 dot-form
// down-pass and the S>1 tile down-pass no longer produce bit-identical
// sums — within-impl incremental==batched is TOLERANCE-based here
// (cross-impl already was; policy in design §10). Row independence is
// unaffected.

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

inline __m256 load8_bf16_as_f32(const bf16* p) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

inline float hsum256(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

// Dot product over a weight row (decode path). Four independent vector
// accumulators (32 lanes in flight) break the scalar dependency chain.
float dot_bf16_f32(const float* x, const bf16* w, uint32_t n) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
  uint32_t i = 0;
  for (; i + 32 <= n; i += 32) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), load8_bf16_as_f32(w + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), load8_bf16_as_f32(w + i + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), load8_bf16_as_f32(w + i + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), load8_bf16_as_f32(w + i + 24), acc3);
  }
  for (; i + 8 <= n; i += 8)
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), load8_bf16_as_f32(w + i), acc0);
  acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
  float acc = hsum256(acc0);
  for (; i < n; ++i) acc += x[i] * w[i].to_f32();
  return acc;
}

// Register-tile microkernel (tinyBLAS-style; reference/sgemm.cpp): computes
// the RMxRN output tile at (m0, n0) of C = A * B^T (+C if accumulate), where
// BOTH operands are k-contiguous (A [m,k] fp32, B [n,k] bf16, PyTorch
// [out,in] weight layout). Each of the RM*RN outputs is an independent dot
// with its own vector accumulator: RM*RN = 12 chains hide FMA latency by
// construction, and every loaded vector is reused from registers (A row x
// RN outputs, B row x RM outputs). hsum once per tile at the end. Because
// B is consumed ROW BY ROW, parallelizing over output tiles partitions
// weights by whole rows — contiguous per-thread streams, no transpose.
template <int RM, int RN>
void gemm_bloc(const float* A, size_t lda, const bf16* B, size_t ldb, float* C,
               size_t ldc, uint32_t m0, uint32_t n0, uint32_t k, bool accumulate) {
  __m256 Cv[RN][RM];
  for (int j = 0; j < RN; ++j)
    for (int i = 0; i < RM; ++i) Cv[j][i] = _mm256_setzero_ps();
  const uint32_t k8 = k & ~7u;
  for (uint32_t l = 0; l < k8; l += 8) {
    __m256 Bv[RN];  // RM=4 > RN=3: keep B in regs (12 acc + 3 Bv + 1 Av = 16)
    for (int j = 0; j < RN; ++j)
      Bv[j] = load8_bf16_as_f32(B + size_t(n0 + j) * ldb + l);
    for (int i = 0; i < RM; ++i) {
      const __m256 Av = _mm256_loadu_ps(A + size_t(m0 + i) * lda + l);
      for (int j = 0; j < RN; ++j) Cv[j][i] = _mm256_fmadd_ps(Av, Bv[j], Cv[j][i]);
    }
  }
  for (int j = 0; j < RN; ++j)
    for (int i = 0; i < RM; ++i) {
      float s = hsum256(Cv[j][i]);
      for (uint32_t l = k8; l < k; ++l)  // scalar k tail
        s += A[size_t(m0 + i) * lda + l] * B[size_t(n0 + j) * ldb + l].to_f32();
      float& c = C[size_t(m0 + i) * ldc + (n0 + j)];
      if (accumulate) c += s; else c = s;
    }
}

#else  // portable scalar fallback

float dot_bf16_f32(const float* x, const bf16* w, uint32_t n) {
  float acc = 0.0f;
  for (uint32_t i = 0; i < n; ++i) acc += x[i] * w[i].to_f32();
  return acc;
}

template <int RM, int RN>
void gemm_bloc(const float* A, size_t lda, const bf16* B, size_t ldb, float* C,
               size_t ldc, uint32_t m0, uint32_t n0, uint32_t k, bool accumulate) {
  for (int j = 0; j < RN; ++j)
    for (int i = 0; i < RM; ++i) {
      float acc = 0.0f;
      for (uint32_t l = 0; l < k; ++l)
        acc += A[size_t(m0 + i) * lda + l] * B[size_t(n0 + j) * ldb + l].to_f32();
      float& c = C[size_t(m0 + i) * ldc + (n0 + j)];
      if (accumulate) c += acc; else c = acc;
    }
}

#endif

// Tile-grid driver: processes flat tile index range [t0, t1) of the
// (ceil(m/4) x ceil(n/3)) grid, dispatching edge tiles to smaller
// instantiations (RM 1..4 x RN 1..3). Consecutive t = consecutive m-tiles
// within one n-tile, so a thread's static range reuses the same B rows.
void gemm_tiles(const float* A, size_t lda, const bf16* B, size_t ldb, float* C,
                size_t ldc, uint32_t m, uint32_t n, uint32_t k, bool accumulate,
                uint32_t t0, uint32_t t1) {
  const uint32_t mtiles = (m + 3) / 4;
  const uint32_t ntiles = (n + 2) / 3;
  (void)ntiles;
  for (uint32_t t = t0; t < t1; ++t) {
    const uint32_t m0 = (t % mtiles) * 4, n0 = (t / mtiles) * 3;
    const uint32_t rm = m - m0 < 4 ? m - m0 : 4;
    const uint32_t rn = n - n0 < 3 ? n - n0 : 3;
    switch (rm * 10 + rn) {
      case 43: gemm_bloc<4, 3>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
      case 42: gemm_bloc<4, 2>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
      case 41: gemm_bloc<4, 1>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
      case 33: gemm_bloc<3, 3>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
      case 32: gemm_bloc<3, 2>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
      case 31: gemm_bloc<3, 1>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
      case 23: gemm_bloc<2, 3>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
      case 22: gemm_bloc<2, 2>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
      case 21: gemm_bloc<2, 1>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
      case 13: gemm_bloc<1, 3>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
      case 12: gemm_bloc<1, 2>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
      case 11: gemm_bloc<1, 1>(A, lda, B, ldb, C, ldc, m0, n0, k, accumulate); break;
    }
  }
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Row-copy with parallel first-touch (static schedule matches the
// forward()-time parallelization).
void copy_rows_first_touch(BF16View src, bf16* dst, size_t dst_ld) {
  ALICE_AVX2_OMP_PARALLEL_FOR
  for (uint32_t r = 0; r < src.rows; ++r)
    std::memcpy(dst + size_t(r) * dst_ld, src.data + size_t(r) * src.stride(),
                src.cols * sizeof(bf16));
}

void zero_first_touch(float* p, size_t n) {
  ALICE_AVX2_OMP_PARALLEL_FOR
  for (size_t i = 0; i < n; ++i) p[i] = 0.0f;
}

}  // namespace

DenseFfnBF16CpuAvx2::DenseFfnBF16CpuAvx2(BF16View w_gate, BF16View w_up,
                                         BF16View w_down, const ModelConfig& mc,
                                         const EngineConfig& ec)
    : d_(mc.hidden_size), inter_(mc.intermediate_size), mpc_(ec.max_prefill_chunk) {
  if (w_gate.rows != inter_ || w_gate.cols != d_ || w_up.rows != inter_ ||
      w_up.cols != d_ || w_down.rows != d_ || w_down.cols != inter_)
    throw std::invalid_argument("DenseFfnBF16CpuAvx2: weight shape mismatch vs config");

  ldw_ = round_up(d_, kAlign / sizeof(bf16));
  ldi_ = round_up(inter_, kAlign / sizeof(bf16));
  wg_ = alloc_aligned<bf16>(size_t(inter_) * ldw_);
  wu_ = alloc_aligned<bf16>(size_t(inter_) * ldw_);
  wd_ = alloc_aligned<bf16>(size_t(d_) * ldi_);
  copy_rows_first_touch(w_gate, wg_.get(), ldw_);
  copy_rows_first_touch(w_up, wu_.get(), ldw_);
  copy_rows_first_touch(w_down, wd_.get(), ldi_);
  full_act_ = alloc_aligned<float>(inter_);

  gatec_ = alloc_aligned<float>(size_t(mpc_) * kChunk);
  upc_ = alloc_aligned<float>(size_t(mpc_) * kChunk);
  actc_ = alloc_aligned<float>(size_t(mpc_) * kChunk);
  out_ = alloc_aligned<float>(size_t(mpc_) * d_);
  zero_first_touch(gatec_.get(), size_t(mpc_) * kChunk);
  zero_first_touch(upc_.get(), size_t(mpc_) * kChunk);
  zero_first_touch(actc_.get(), size_t(mpc_) * kChunk);
  zero_first_touch(out_.get(), size_t(mpc_) * d_);
}

void DenseFfnBF16CpuAvx2::forward(F32View io) {
  const uint32_t S = io.rows;
  if (S > mpc_)
    throw std::runtime_error("DenseFfnBF16CpuAvx2: prefill chunk exceeded");
  if (io.cols != d_) throw std::invalid_argument("DenseFfnBF16CpuAvx2: io must be [S, d]");

  const float* x = io.data;
  const size_t xld = io.stride();

  if (S == 1) {  // --- decode: dot-form, contiguous streaming, 2 regions ---
    ALICE_AVX2_OMP_PARALLEL_FOR
    for (uint32_t j = 0; j < inter_; ++j) {
      const bf16* wg = wg_.get() + size_t(j) * ldw_;
      const bf16* wu = wu_.get() + size_t(j) * ldw_;
      full_act_[j] = silu(dot_bf16_f32(x, wg, d_)) * dot_bf16_f32(x, wu, d_);
    }
    ALICE_AVX2_OMP_PARALLEL_FOR
    for (uint32_t i = 0; i < d_; ++i)
      io.data[i] = dot_bf16_f32(full_act_.get(), wd_.get() + size_t(i) * ldi_, inter_);
    return;
  }

  // --- prefill: fused chunk loop with register-tile GEMM --------------------
  // The chunk loop over I is the K-blocking; all three projections run the
  // same microkernel. One persistent parallel region (static omp for
  // throughout — no dynamic scheduling), implicit barriers between phases.
  ALICE_AVX2_OMP_PARALLEL
  {
    ALICE_AVX2_OMP_FOR
    for (size_t e = 0; e < size_t(S) * d_; ++e) out_[e] = 0.0f;

    for (uint32_t j0 = 0; j0 < inter_; j0 += kChunk) {
      const uint32_t je = std::min(j0 + kChunk, inter_);
      const uint32_t cn = je - j0;
      const uint32_t mt = (S + 3) / 4, nt_c = (cn + 2) / 3, nt_d = (d_ + 2) / 3;

      // gate / up: C = x . Wg[chunk]^T, C = x . Wu[chunk]^T  (chunk cols only)
      ALICE_AVX2_OMP_FOR
      for (uint32_t t = 0; t < mt * nt_c; ++t)
        gemm_tiles(x, xld, wg_.get() + size_t(j0) * ldw_, ldw_, gatec_.get(), kChunk,
                   S, cn, d_, false, t, t + 1);
      ALICE_AVX2_OMP_FOR
      for (uint32_t t = 0; t < mt * nt_c; ++t)
        gemm_tiles(x, xld, wu_.get() + size_t(j0) * ldw_, ldw_, upc_.get(), kChunk,
                   S, cn, d_, false, t, t + 1);

      // act[s][c] = silu(gate) * up   (layout [S][kChunk]: k-contiguous
      // rows — note row stride kChunk, NOT cn: flat indexing breaks on the
      // tail chunk where cn < kChunk)
      ALICE_AVX2_OMP_FOR
      for (uint32_t s = 0; s < S; ++s) {
        float* a = actc_.get() + size_t(s) * kChunk;
        const float* g = gatec_.get() + size_t(s) * kChunk;
        const float* u = upc_.get() + size_t(s) * kChunk;
        for (uint32_t c = 0; c < cn; ++c) a[c] = silu(g[c]) * u[c];
      }

      // down: out_ += act . Wd[:, chunk]^T  — B = row-major wd_, column
      // offset j0; each thread's i-tiles read WHOLE wd_ rows unit-stride.
      ALICE_AVX2_OMP_FOR
      for (uint32_t t = 0; t < mt * nt_d; ++t)
        gemm_tiles(actc_.get(), kChunk, wd_.get() + j0, ldi_, out_.get(), d_, S, d_,
                   cn, true, t, t + 1);
    }

    // Fulfil the in-place io contract.
    ALICE_AVX2_OMP_FOR
    for (uint32_t s = 0; s < S; ++s)
      std::memcpy(io.data + size_t(s) * io.stride(), out_.get() + size_t(s) * d_,
                  d_ * sizeof(float));
  }
}

}  // namespace alice::cpu_avx2
