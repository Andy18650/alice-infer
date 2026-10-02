#pragma once

// Naive host-side math helpers: the ground-truth oracle for the GPU SIMT
// kernels (test_gpu_kernels) + the host-side RoPE table builder used by the
// gpu_rocm attention block. Stateless, everything passed per call; clarity
// over speed. (The retired host-block reference impls used to live alongside
// these; naive reference impls for new block types are now done as GPU
// kernels, not host blocks.)

#include <cstdint>
#include <optional>

#include "rope_scaling.hpp"  // RopeScaling (a build_rope_tables param)

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

namespace alice::host_math {

// x[s][:] /= rms(x[s][:]) (+eps), then *= g[:]. In-place; x is fp32 [S, d].
void rmsnorm_inplace(float* x, uint32_t S, uint32_t d, const float* g, float eps);

void add_rows(float* dst, const float* src, uint32_t S, uint32_t d);   // dst += src

// RoPE tables, HF/Llama convention: cos_t[p*half + i] = cos(p * inv_freq[i]),
// inv_freq[i] = theta^(-2i/head_dim), with optional llama3 frequency scaling
// (low band divided by factor, high band unchanged, smooth between).
// `rs` present => apply scaling; absent => plain theta tables.
void build_rope_tables(float* cos_t, float* sin_t, uint32_t max_seq, uint32_t head_dim,
                       float theta, const std::optional<RopeScaling>& rs);

// Rotate one head vector in place at one position (half-split pairing:
// x[i] with x[i+half]).
void rope_rotate(float* x, uint32_t head_dim, const float* cos_row, const float* sin_row);

float silu(float x);

}  // namespace alice::host_math
