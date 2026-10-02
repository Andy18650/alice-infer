#include "math.hpp"

#include <cmath>

namespace oracle {

void rmsnorm_inplace(float* x, uint32_t S, uint32_t d, const float* g, float eps) {
  for (uint32_t s = 0; s < S; ++s) {
    float* row = x + size_t(s) * d;
    float ss = 0.0f;
    for (uint32_t i = 0; i < d; ++i) ss += row[i] * row[i];
    const float inv = 1.0f / std::sqrt(ss / float(d) + eps);
    for (uint32_t i = 0; i < d; ++i) row[i] = row[i] * inv * g[i];
  }
}

void add_rows(float* dst, const float* src, uint32_t S, uint32_t d) {
  for (uint32_t s = 0; s < S; ++s) {
    float* dd = dst + size_t(s) * d;
    const float* r = src + size_t(s) * d;
    for (uint32_t i = 0; i < d; ++i) dd[i] += r[i];
  }
}

void build_rope_tables(float* cos_t, float* sin_t, uint32_t max_seq, uint32_t head_dim,
                       float theta) {
  const uint32_t half = head_dim / 2;
  const double ln_theta = std::log(double(theta));
  for (uint32_t i = 0; i < half; ++i) {
    const double f = std::exp(-(2.0 * i / double(head_dim)) * ln_theta);
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

}  // namespace oracle
