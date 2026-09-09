#include "kernels.hpp"

#include <cmath>
#include <stdexcept>

namespace alice::cpu_generic {

void matmul_wt(F32View in, BF16View w, F32View out) {
  const uint32_t S = in.rows, I = in.cols, O = w.rows;
  if (w.cols != I || out.rows != S || out.cols != O)
    throw std::invalid_argument("cpu_generic: matmul shape mismatch");
  // o outer: one parallel region, O-way parallelism; each (s, o) reduction
  // stays sequential inside a thread (bit-exact vs the sequential oracle).
  ALICE_OMP_PARALLEL_FOR
  for (uint32_t o = 0; o < O; ++o) {
    const bf16* w_row = w.data + size_t(o) * w.stride();
    for (uint32_t s = 0; s < S; ++s) {
      const float* in_row = in.data + size_t(s) * in.stride();
      float acc = 0.0f;
      for (uint32_t i = 0; i < I; ++i) acc += in_row[i] * w_row[i].to_f32();
      out.data[size_t(s) * out.stride() + o] = acc;
    }
  }
}

void rmsnorm_inplace(F32View x, const float* g, float eps) {
  for (uint32_t s = 0; s < x.rows; ++s) {
    float* row = x.data + size_t(s) * x.stride();
    float ss = 0.0f;
    for (uint32_t i = 0; i < x.cols; ++i) ss += row[i] * row[i];
    const float inv = 1.0f / std::sqrt(ss / float(x.cols) + eps);
    for (uint32_t i = 0; i < x.cols; ++i) row[i] = row[i] * inv * g[i];
  }
}

void copy_rows(F32View dst, F32View src) {
  if (dst.rows != src.rows || dst.cols != src.cols)
    throw std::invalid_argument("cpu_generic: copy shape mismatch");
  for (uint32_t s = 0; s < src.rows; ++s)
    std::memcpy(dst.data + size_t(s) * dst.stride(), src.data + size_t(s) * src.stride(),
                src.cols * sizeof(float));
}

void add_rows(F32View dst, F32View src) {
  if (dst.rows != src.rows || dst.cols != src.cols)
    throw std::invalid_argument("cpu_generic: add shape mismatch");
  for (uint32_t s = 0; s < src.rows; ++s) {
    float* d = dst.data + size_t(s) * dst.stride();
    const float* r = src.data + size_t(s) * src.stride();
    for (uint32_t i = 0; i < src.cols; ++i) d[i] += r[i];
  }
}

void build_rope_tables(float* cos_t, float* sin_t, uint32_t max_seq, uint32_t head_dim,
                       float theta, bool has_scaling, const RopeScaling& rs) {
  const uint32_t half = head_dim / 2;
  const double ln_theta = std::log(double(theta));
  for (uint32_t i = 0; i < half; ++i) {
    double f = std::exp(-(2.0 * i / double(head_dim)) * ln_theta);
    if (has_scaling) {
      const double wavelen = 2.0 * M_PI / f;
      const double orig = double(rs.original_max_position_embeddings);
      if (wavelen > orig / double(rs.low_freq_factor)) {
        f /= double(rs.factor);  // low-frequency band
      } else if (wavelen >= orig / double(rs.high_freq_factor)) {
        // medium band: smooth interpolation
        const double s =
            (orig / wavelen - double(rs.low_freq_factor)) /
            (double(rs.high_freq_factor) - double(rs.low_freq_factor));
        f = (1.0 - s) * f / double(rs.factor) + s * f;
      }  // else high-frequency band: unchanged
    }
    for (uint32_t p = 0; p < max_seq; ++p) {
      cos_t[size_t(p) * half + i] = float(std::cos(double(p) * f));
      sin_t[size_t(p) * half + i] = float(std::sin(double(p) * f));
    }
  }
}

void rope_rotate(float* x, uint32_t head_dim, const float* cos_row, const float* sin_row) {
  const uint32_t half = head_dim / 2;
  for (uint32_t i = 0; i < half; ++i) {
    const float x1 = x[i], x2 = x[i + half];
    x[i] = x1 * cos_row[i] - x2 * sin_row[i];
    x[i + half] = x2 * cos_row[i] + x1 * sin_row[i];
  }
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

void softmax(float* x, uint32_t n) {
  float mx = x[0];
  for (uint32_t i = 1; i < n; ++i) mx = std::max(mx, x[i]);
  float sum = 0.0f;
  for (uint32_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);
    sum += x[i];
  }
  const float inv = 1.0f / sum;
  for (uint32_t i = 0; i < n; ++i) x[i] *= inv;
}

}  // namespace alice::cpu_generic
