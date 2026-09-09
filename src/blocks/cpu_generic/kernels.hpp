#pragma once

// Naive CPU_GENERIC kernels — the "functional" tier of the taxonomy
// (design §5): stateless, everything passed per call. These are the
// permanent test oracle; clarity over speed, always.

#include <cstdint>
#include <cstring>

#include "alice/manifest.hpp"  // RopeScaling
#include "alice/tensor.hpp"

namespace alice::cpu_generic {

inline float bf16_to_f32(uint16_t b) {
  uint32_t bits = uint32_t(b) << 16;  // bf16 = top 16 bits of fp32
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// out[s][o] = sum_i in[s][i] * bf16(w[o][i]).
// in: fp32 [S, I]; w: bf16 [O, I]; out: fp32 [S, O]. fp32 accumulation.
// `in` and `out` must not alias.
void matmul_wt(const TensorView& in, const TensorView& w, TensorView& out);

// x[s][:] /= rms(x[s][:]) (+eps), then *= g[:]. In-place; x is fp32 [S, d].
void rmsnorm_inplace(TensorView& x, const float* g, float eps);

void copy_rows(TensorView& dst, const TensorView& src);  // dst = src (same shape)
void add_rows(TensorView& dst, const TensorView& src);   // dst += src

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
