#include "kernels.hpp"

#include <cmath>
#include <stdexcept>

namespace alice::cpu_generic {

void matmul_wt(const float* in, const bf16* w, float* out, uint32_t S, uint32_t I,
               uint32_t O) {
  // o outer: one parallel region, O-way parallelism; each (s, o) reduction
  // stays sequential inside a thread (bit-exact vs the sequential oracle).
  ALICE_OMP_PARALLEL_FOR
  for (uint32_t o = 0; o < O; ++o) {
    const bf16* w_row = w + size_t(o) * I;
    for (uint32_t s = 0; s < S; ++s) {
      const float* in_row = in + size_t(s) * I;
      float acc = 0.0f;
      for (uint32_t i = 0; i < I; ++i) acc += in_row[i] * w_row[i].to_f32();
      out[size_t(s) * O + o] = acc;
    }
  }
}

void rmsnorm_inplace(float* x, uint32_t S, uint32_t d, const float* g, float eps) {
  for (uint32_t s = 0; s < S; ++s) {
    float* row = x + size_t(s) * d;
    float ss = 0.0f;
    for (uint32_t i = 0; i < d; ++i) ss += row[i] * row[i];
    const float inv = 1.0f / std::sqrt(ss / float(d) + eps);
    for (uint32_t i = 0; i < d; ++i) row[i] = row[i] * inv * g[i];
  }
}

void copy_rows(float* dst, const float* src, uint32_t S, uint32_t d) {
  for (uint32_t s = 0; s < S; ++s)
    std::memcpy(dst + size_t(s) * d, src + size_t(s) * d, d * sizeof(float));
}

void add_rows(float* dst, const float* src, uint32_t S, uint32_t d) {
  for (uint32_t s = 0; s < S; ++s) {
    float* dd = dst + size_t(s) * d;
    const float* r = src + size_t(s) * d;
    for (uint32_t i = 0; i < d; ++i) dd[i] += r[i];
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
