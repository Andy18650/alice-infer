#include "alice/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace alice {

int32_t Sampler::sample(const TensorView& logits) {
  if (logits.dtype != DType::F32 || logits.rows != 1)
    throw std::invalid_argument("Sampler: logits must be fp32 [1, vocab]");
  const uint32_t V = logits.cols;
  const float* in = logits.as<float>();

  if (cfg_.temperature == 0.0f)  // greedy
    return int32_t(std::max_element(in, in + V) - in);

  work_.assign(in, in + V);
  const float inv_t = 1.0f / cfg_.temperature;
  for (auto& x : work_) x *= inv_t;

  // softmax (stabilized)
  const float mx = *std::max_element(work_.begin(), work_.end());
  float sum = 0.0f;
  for (auto& x : work_) {
    x = std::exp(x - mx);
    sum += x;
  }
  for (auto& x : work_) x /= sum;

  // candidate pool: indices sorted by descending probability
  std::vector<uint32_t> idx(V);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](uint32_t a, uint32_t b) { return work_[a] > work_[b]; });

  size_t keep = V;
  if (cfg_.top_k > 0 && cfg_.top_k < keep) keep = cfg_.top_k;
  if (cfg_.top_p < 1.0f) {
    float cum = 0.0f;
    size_t cut = 0;
    while (cut < keep && cum < cfg_.top_p) cum += work_[idx[cut++]];
    keep = cut;
  }

  float total = 0.0f;
  for (size_t i = 0; i < keep; ++i) total += work_[idx[i]];
  std::uniform_real_distribution<float> u(0.0f, total);
  const float r = u(rng_);
  float acc = 0.0f;
  for (size_t i = 0; i < keep; ++i) {
    acc += work_[idx[i]];
    if (r < acc) return int32_t(idx[i]);
  }
  return int32_t(idx[keep - 1]);  // rounding guard
}

}  // namespace alice
