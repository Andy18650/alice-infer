// CPU_AVX2 implementation of DenseFfnBF16.
//
// Step 2 of the AVX2 plan: STRUCTURE ONLY, scalar inner loops —
//   * aligned private weight storage with parallel first-touch (NUMA-local
//     residency without mbind: the thread that streams a page at forward()
//     time is, under static scheduling + same thread count, the one that
//     faulted it in at construction),
//   * TWO schedules dispatched on S, never on arithmetic order:
//       - S == 1 (decode): dot-form for all three projections, parallel over
//         output rows — every thread streams CONTIGUOUS weight rows (the
//         oracle's DRAM-friendly pattern). Measured: a column-block-parallel
//         down-pass (64B slices at 4KB stride) collapses DRAM efficiency
//         and costs ~3.5x decode at 48 threads; hence this path.
//       - S > 1 (prefill): fused gate/up/down chunk loop; the [S, I]
//         intermediate never exists, w_down is consumed TRANSPOSED so the
//         down-pass streams contiguous rows ("out += act_j * WdT[j][:]")
//         instead of strided columns.
//     Both paths accumulate every output element over j in ASCENDING order
//     with one add per j — the identical FP op sequence — so
//     incremental==batched stays BIT-EXACT (verified by test).
//   * private [S, d] output accumulator for the fused path, because fusion
//     cannot accumulate into io while later chunks still read the input
//     from io; copied back at the end to fulfil the in-place contract.
//   * w_down is kept in BOTH layouts (row-major for the dot path,
//     transposed for the fused path): +50% FFN weight storage, flagged for
//     the memory audit.
//
// Numerics (user-signed-off, design §10): chunked accumulation changes
// summation order vs the oracle, so cross-impl equivalence is
// tolerance-based. Determinism per fixed schedule and incremental==batched
// bit-equality ARE preserved (see above).

#include "dense_ffn.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(_OPENMP)
#define ALICE_AVX2_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define ALICE_AVX2_OMP_PARALLEL_FOR_C2 _Pragma("omp parallel for collapse(2)")
#else
#define ALICE_AVX2_OMP_PARALLEL_FOR
#define ALICE_AVX2_OMP_PARALLEL_FOR_C2
#endif

namespace alice::cpu_avx2 {
namespace {

constexpr size_t kAlign = 64;       // cache line; AVX2 32B loads are a subset
constexpr uint32_t kChunk = 256;    // intermediate rows per fused chunk
constexpr uint32_t kColBlock = 32;  // down-pass column tile (64B of bf16)

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
// NUMERICS NOTE: the multi-accumulator dot sums in lane-group order, not
// strict ascending order. Per (impl, S, thread-count) the schedule is fixed
// and deterministic, but the S==1 dot-form down-pass and the S>1 chunked
// axpy down-pass no longer produce bit-identical sums — within-impl
// incremental==batched becomes TOLERANCE-based here (cross-impl already
// was; policy in design §10). Row independence is unaffected.

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

// Dot product over a weight row. Four independent vector accumulators (32
// lanes in flight) break the scalar dependency chain: Zen3 FMA latency 4,
// throughput 2/cycle — a single accumulator starves both pipes.
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

void axpy_bf16(float* out, float a, const bf16* w, uint32_t n) {
  const __m256 va = _mm256_set1_ps(a);
  uint32_t i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm256_storeu_ps(out + i,
                     _mm256_fmadd_ps(va, load8_bf16_as_f32(w + i), _mm256_loadu_ps(out + i)));
    _mm256_storeu_ps(out + i + 8, _mm256_fmadd_ps(va, load8_bf16_as_f32(w + i + 8),
                                                  _mm256_loadu_ps(out + i + 8)));
  }
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(out + i,
                     _mm256_fmadd_ps(va, load8_bf16_as_f32(w + i), _mm256_loadu_ps(out + i)));
  for (; i < n; ++i) out[i] += a * w[i].to_f32();
}

#else  // portable scalar fallback (single accumulator, strict order)

float dot_bf16_f32(const float* x, const bf16* w, uint32_t n) {
  float acc = 0.0f;
  for (uint32_t i = 0; i < n; ++i) acc += x[i] * w[i].to_f32();
  return acc;
}

void axpy_bf16(float* out, float a, const bf16* w, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) out[i] += a * w[i].to_f32();
}

#endif

float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Row-copy with parallel first-touch (static schedule matches the
// forward()-time parallelization).
void copy_rows_first_touch(BF16View src, bf16* dst, size_t dst_ld) {
  ALICE_AVX2_OMP_PARALLEL_FOR
  for (uint32_t r = 0; r < src.rows; ++r)
    std::memcpy(dst + size_t(r) * dst_ld, src.data + size_t(r) * src.stride(),
                src.cols * sizeof(bf16));
}

// Transposed copy for w_down ([d, I] -> dst [I, ld]); parallel first-touch
// over destination rows.
void copy_transposed_first_touch(BF16View src, bf16* dst, size_t dst_ld) {
  ALICE_AVX2_OMP_PARALLEL_FOR
  for (uint32_t j = 0; j < src.cols; ++j)
    for (uint32_t i = 0; i < src.rows; ++i)
      dst[size_t(j) * dst_ld + i] = src.data[size_t(i) * src.stride() + j];
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
  wdt_ = alloc_aligned<bf16>(size_t(inter_) * ldw_);
  wd_ = alloc_aligned<bf16>(size_t(d_) * ldi_);  // row-major copy for the dot path
  copy_rows_first_touch(w_gate, wg_.get(), ldw_);
  copy_rows_first_touch(w_up, wu_.get(), ldw_);
  copy_transposed_first_touch(w_down, wdt_.get(), ldw_);
  copy_rows_first_touch(w_down, wd_.get(), ldi_);
  full_act_ = alloc_aligned<float>(inter_);

  act_ = alloc_aligned<float>(size_t(kChunk) * mpc_);
  out_ = alloc_aligned<float>(size_t(mpc_) * d_);
  zero_first_touch(act_.get(), size_t(kChunk) * mpc_);
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

  // --- prefill: fused chunk loop -------------------------------------------
  ALICE_AVX2_OMP_PARALLEL_FOR
  for (size_t e = 0; e < size_t(S) * d_; ++e) out_[e] = 0.0f;

  for (uint32_t j0 = 0; j0 < inter_; j0 += kChunk) {
    const uint32_t je = std::min(j0 + kChunk, inter_);
    const uint32_t cn = je - j0;

    // Phase 1: act[c][s] = silu(x_s . Wg[j0+c]) * (x_s . Wu[j0+c]).
    // Parallel over chunk rows c; weight row stays hot for all S dots.
    ALICE_AVX2_OMP_PARALLEL_FOR
    for (uint32_t c = 0; c < cn; ++c) {
      const bf16* wg = wg_.get() + size_t(j0 + c) * ldw_;
      const bf16* wu = wu_.get() + size_t(j0 + c) * ldw_;
      float* a = act_.get() + size_t(c) * mpc_;
      for (uint32_t s = 0; s < S; ++s) {
        const float* xr = x + size_t(s) * xld;
        a[s] = silu(dot_bf16_f32(xr, wg, d_)) * dot_bf16_f32(xr, wu, d_);
      }
    }

    // Phase 2: out[s][i] += sum_c act[c][s] * WdT[j0+c][i], fixed ascending
    // c order per output element (thread-count-independent bits).
    ALICE_AVX2_OMP_PARALLEL_FOR_C2
    for (uint32_t s = 0; s < S; ++s)
      for (uint32_t ib = 0; ib < d_; ib += kColBlock) {
        float* o = out_.get() + size_t(s) * d_ + ib;
        const uint32_t n = std::min(kColBlock, d_ - ib);
        for (uint32_t c = 0; c < cn; ++c)
          axpy_bf16(o, act_[size_t(c) * mpc_ + s],
                    wdt_.get() + size_t(j0 + c) * ldw_ + ib, n);
      }
  }

  // Fulfil the in-place io contract.
  if (io.stride() == d_) {
    std::memcpy(io.data, out_.get(), size_t(S) * d_ * sizeof(float));
  } else {
    for (uint32_t s = 0; s < S; ++s)
      std::memcpy(io.data + size_t(s) * io.stride(), out_.get() + size_t(s) * d_,
                  d_ * sizeof(float));
  }
}

}  // namespace alice::cpu_avx2
