// GPU Sampler tests: verify that the GPU-resident sampler (device argmax
// for greedy, D2H + host sample for temperature > 0) produces identical
// results to a direct CPU reference.

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include "../src/gpu/sampler.hpp"

using alice::SamplingConfig;
using alice::gpu::Sampler;

namespace {

float* dev_upload(const std::vector<float>& host) {
  float* p = nullptr;
  if (hipMalloc(&p, host.size() * sizeof(float)) != hipSuccess)
    throw std::bad_alloc();
  if (hipMemcpy(p, host.data(), host.size() * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    throw std::runtime_error("H2D failed");
  return p;
}

}  // namespace

TEST(GpuSampler, GreedyMatchesCpuArgmax) {
  // The GPU sampler at temperature=0 must return the same index as
  // std::max_element on the host.
  const uint32_t vocab = 1024;
  std::mt19937 rng(42);
  std::vector<float> logits(vocab);
  for (auto& v : logits) v = float(rng()) / float(rng.max()) * 2.f - 1.f;

  const int32_t cpu_id =
      int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());

  SamplingConfig cfg;  // temperature 0 (greedy)
  cfg.seed = 0;
  Sampler sampler(cfg);

  float* dev = dev_upload(logits);
  const int32_t gpu_id = sampler.sample(dev, vocab);
  hipFree(dev);

  EXPECT_EQ(gpu_id, cpu_id);
}

TEST(GpuSampler, GreedyMatchesCpuArgmaxSmallVocab) {
  // Edge case: very small vocabulary (the kernel must handle V < warp).
  const uint32_t vocab = 7;
  std::vector<float> logits = {0.1f, -0.5f, 0.3f, 0.9f, -0.2f, 0.8f, 0.05f};

  const int32_t cpu_id =
      int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());

  SamplingConfig cfg;
  cfg.seed = 0;
  Sampler sampler(cfg);

  float* dev = dev_upload(logits);
  const int32_t gpu_id = sampler.sample(dev, vocab);
  hipFree(dev);

  EXPECT_EQ(gpu_id, cpu_id);
}

TEST(GpuSampler, GreedyDeterministicAcrossCalls) {
  // Greedy sampling must be deterministic (no RNG involved).
  const uint32_t vocab = 512;
  std::mt19937 rng(99);
  std::vector<float> logits(vocab);
  for (auto& v : logits) v = float(rng()) / float(rng.max());

  SamplingConfig cfg;
  cfg.seed = 0;
  Sampler sampler(cfg);

  float* dev = dev_upload(logits);
  const int32_t id1 = sampler.sample(dev, vocab);
  const int32_t id2 = sampler.sample(dev, vocab);
  hipFree(dev);

  EXPECT_EQ(id1, id2);
}

TEST(GpuSampler, TemperatureMatchesCpuSample) {
  // Temperature > 0 path: D2H logits, then host-side sampling.
  // Verify against a pure CPU implementation using the same seed.
  const uint32_t vocab = 256;
  const float temperature = 0.8f;
  const uint64_t seed = 12345;

  std::mt19937 rng(42);
  std::vector<float> logits(vocab);
  for (auto& v : logits) v = float(rng()) / float(rng.max()) * 2.f - 1.f;

  // GPU sampler
  SamplingConfig cfg;
  cfg.temperature = temperature;
  cfg.seed = seed;
  Sampler sampler(cfg);

  float* dev = dev_upload(logits);
  const int32_t gpu_id = sampler.sample(dev, vocab);
  hipFree(dev);

  // CPU reference: same sampling logic as sample_host
  std::vector<float> work(logits);
  const float inv_t = 1.0f / temperature;
  for (auto& x : work) x *= inv_t;
  const float mx = *std::max_element(work.begin(), work.end());
  float sum = 0.0f;
  for (auto& x : work) { x = std::exp(x - mx); sum += x; }
  for (auto& x : work) x /= sum;

  std::vector<uint32_t> idx(vocab);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](uint32_t a, uint32_t b) { return work[a] > work[b]; });

  size_t keep = vocab;  // top_k=0, top_p=1.0 → no pruning
  float total = 0.0f;
  for (size_t i = 0; i < keep; ++i) total += work[idx[i]];
  std::mt19937_64 cpu_rng(seed);
  std::uniform_real_distribution<float> u(0.0f, total);
  const float r = u(cpu_rng);
  float acc = 0.0f;
  int32_t cpu_id = int32_t(idx[keep - 1]);
  for (size_t i = 0; i < keep; ++i) {
    acc += work[idx[i]];
    if (r < acc) { cpu_id = int32_t(idx[i]); break; }
  }

  EXPECT_EQ(gpu_id, cpu_id);
}
