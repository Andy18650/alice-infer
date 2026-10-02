#pragma once

// Host-side ground-truth reference math for the GPU SIMT-kernel tests
// (test_gpu_kernels): each device kernel is compared against these naive
// host implementations — the "two watches" oracle (design §13). Stateless,
// everything passed per call; clarity over speed.
//
// TEST-ONLY code, fully separated from production (lives under tests/oracle/,
// not src/). The production RoPE table builder lives in
// src/blocks/gpu_rocm/gqa_attention.cpp and is covered end-to-end by
// test_hf_block_oracle; this is the independent host reference for the
// kernel unit tests. The rope table builder here omits llama3 scaling (the
// kernel tests don't exercise it; the scaling path is covered by the
// attention block oracle).

#include <cstdint>

namespace oracle {

// x[s][:] /= rms(x[s][:]) (+eps), then *= g[:]. In-place; x is fp32 [S, d].
void rmsnorm_inplace(float* x, uint32_t S, uint32_t d, const float* g, float eps);

void add_rows(float* dst, const float* src, uint32_t S, uint32_t d);   // dst += src

// RoPE tables, HF/Llama convention (no scaling): cos_t[p*half + i] =
// cos(p * inv_freq[i]), inv_freq[i] = theta^(-2i/head_dim).
void build_rope_tables(float* cos_t, float* sin_t, uint32_t max_seq, uint32_t head_dim,
                       float theta);

// Rotate one head vector in place at one position (half-split pairing:
// x[i] with x[i+half]).
void rope_rotate(float* x, uint32_t head_dim, const float* cos_row, const float* sin_row);

float silu(float x);

}  // namespace oracle
