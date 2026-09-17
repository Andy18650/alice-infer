#pragma once

// Naive CPU_GENERIC kernels — the "functional" tier of the taxonomy
// (design §5): stateless, everything passed per call. These are the
// permanent test oracle; clarity over speed, always.

#include <cstdint>
#include <cstring>

#include "alice/manifest.hpp"  // RopeScaling
#include "alice/tensor.hpp"

// Intra-op parallelism is the implementation's private business (design
// §10): OpenMP when available, transparent sequential fallback otherwise.
// Parallelism is over OUTPUT ELEMENTS ONLY — every reduction stays
// sequential within a thread, so results are bit-identical to the naive
// sequential oracle. OMP_NUM_THREADS=1 reproduces the baseline exactly.
#if defined(_OPENMP)
#define ALICE_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define ALICE_OMP_PARALLEL_FOR
#endif

namespace alice::cpu_generic {

// out[s][o] = sum_i in[s][i] * float(w[o][i]).
// in: fp32 [S, I]; w: bf16 [O, I]; out: fp32 [S, O]. fp32 accumulation.
// `in` and `out` must not alias. OpenMP-parallel over output columns o.
void matmul_wt(const float* in, const bf16* w, float* out, uint32_t S, uint32_t I,
               uint32_t O);

// x[s][:] /= rms(x[s][:]) (+eps), then *= g[:]. In-place; x is fp32 [S, d].
void rmsnorm_inplace(float* x, uint32_t S, uint32_t d, const float* g, float eps);

void copy_rows(float* dst, const float* src, uint32_t S, uint32_t d);  // dst = src
void add_rows(float* dst, const float* src, uint32_t S, uint32_t d);   // dst += src

// RoPE tables, HF/Llama convention: cos_t[p*half + i] = cos(p * inv_freq[i]),
// inv_freq[i] = theta^(-2i/head_dim), with optional llama3 frequency scaling
// (low band divided by factor, high band unchanged, smooth between).
void build_rope_tables(float* cos_t, float* sin_t, uint32_t max_seq, uint32_t head_dim,
                       float theta, bool has_scaling, const RopeScaling& rs);

// Rotate one head vector in place at one position (half-split pairing:
// x[i] with x[i+half]).
void rope_rotate(float* x, uint32_t head_dim, const float* cos_row, const float* sin_row);

float silu(float x);
void softmax(float* x, uint32_t n);  // in place, numerically stabilized

}  // namespace alice::cpu_generic
