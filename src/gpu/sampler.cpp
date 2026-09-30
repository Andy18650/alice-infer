#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime_api.h>

#include "kernels.hpp"

namespace alice::gpu {

namespace {
#define HIP_CHECK(x)                                                                    \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess)                                                               \
      throw std::runtime_error(std::string("gpu::Sampler: ") + #x + " failed: " +      \
                               hipGetErrorString(e_));                                  \
  } while (0)
}  // namespace

Sampler::Sampler(const SamplingConfig& cfg) : cfg_(cfg), rng_(cfg.seed) {
  HIP_CHECK(hipMalloc(&id_dev_, sizeof(int32_t)));
}

Sampler::~Sampler() { (void)hipFree(id_dev_); }

int32_t Sampler::sample(const float* logits, uint32_t vocab) {
  if (cfg_.temperature == 0.0f) {
    kernels::argmax(logits, vocab, id_dev_);
    int32_t id = 0;
    HIP_CHECK(hipMemcpy(&id, id_dev_, sizeof(int32_t), hipMemcpyDeviceToHost));
    return id;
  }
  logits_host_.resize(vocab);
  HIP_CHECK(hipMemcpy(logits_host_.data(), logits, vocab * sizeof(float),
                      hipMemcpyDeviceToHost));
  return sample_host(logits_host_.data(), vocab);
}

int32_t Sampler::sample_host(const float* logits, uint32_t vocab) {
  const uint32_t V = vocab;
  const float* in = logits;

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

}  // namespace alice::gpu
