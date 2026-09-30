// GPU Sampler tests: verify that gpu::Sampler produces identical results
// to the original host alice::Sampler, preserved in tests/oracle/ as a
// frozen git-extracted oracle. The oracle is the code that shipped as
// src/sampler.cpp before the merge — any divergence is a transcription bug.

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include "oracle/sampler.hpp"
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

TEST(GpuSampler, GreedyMatchesOracle) {
  const uint32_t vocab = 1024;
  std::mt19937 rng(42);
  std::vector<float> logits(vocab);
  for (auto& v : logits) v = float(rng()) / float(rng.max()) * 2.f - 1.f;

  SamplingConfig cfg;  // temperature 0 (greedy)
  oracle::Sampler ref(cfg);
  const int32_t cpu_id = ref.sample(logits.data(), vocab);

  Sampler sampler(cfg);
  float* dev = dev_upload(logits);
  const int32_t gpu_id = sampler.sample(dev, vocab);
  hipFree(dev);

  EXPECT_EQ(gpu_id, cpu_id);
}

TEST(GpuSampler, GreedyMatchesOracleSmallVocab) {
  const uint32_t vocab = 7;
  std::vector<float> logits = {0.1f, -0.5f, 0.3f, 0.9f, -0.2f, 0.8f, 0.05f};

  SamplingConfig cfg;
  oracle::Sampler ref(cfg);
  const int32_t cpu_id = ref.sample(logits.data(), vocab);

  Sampler sampler(cfg);
  float* dev = dev_upload(logits);
  const int32_t gpu_id = sampler.sample(dev, vocab);
  hipFree(dev);

  EXPECT_EQ(gpu_id, cpu_id);
}

TEST(GpuSampler, GreedyDeterministicAcrossCalls) {
  const uint32_t vocab = 512;
  std::mt19937 rng(99);
  std::vector<float> logits(vocab);
  for (auto& v : logits) v = float(rng()) / float(rng.max());

  SamplingConfig cfg;
  Sampler sampler(cfg);

  float* dev = dev_upload(logits);
  const int32_t id1 = sampler.sample(dev, vocab);
  const int32_t id2 = sampler.sample(dev, vocab);
  hipFree(dev);

  EXPECT_EQ(id1, id2);
}

TEST(GpuSampler, TemperatureMatchesOracle) {
  const uint32_t vocab = 256;
  const float temperature = 0.8f;
  const uint64_t seed = 12345;

  std::mt19937 rng(42);
  std::vector<float> logits(vocab);
  for (auto& v : logits) v = float(rng()) / float(rng.max()) * 2.f - 1.f;

  SamplingConfig cfg;
  cfg.temperature = temperature;
  cfg.seed = seed;

  oracle::Sampler ref(cfg);
  const int32_t cpu_id = ref.sample(logits.data(), vocab);

  Sampler sampler(cfg);
  float* dev = dev_upload(logits);
  const int32_t gpu_id = sampler.sample(dev, vocab);
  hipFree(dev);

  EXPECT_EQ(gpu_id, cpu_id);
}

TEST(GpuSampler, TopKMatchesOracle) {
  const uint32_t vocab = 256;
  const float temperature = 0.8f;
  const uint64_t seed = 54321;

  std::mt19937 rng(77);
  std::vector<float> logits(vocab);
  for (auto& v : logits) v = float(rng()) / float(rng.max()) * 2.f - 1.f;

  SamplingConfig cfg;
  cfg.temperature = temperature;
  cfg.top_k = 40;
  cfg.seed = seed;

  oracle::Sampler ref(cfg);
  const int32_t cpu_id = ref.sample(logits.data(), vocab);

  Sampler sampler(cfg);
  float* dev = dev_upload(logits);
  const int32_t gpu_id = sampler.sample(dev, vocab);
  hipFree(dev);

  EXPECT_EQ(gpu_id, cpu_id);
}

TEST(GpuSampler, TopPMatchesOracle) {
  const uint32_t vocab = 256;
  const float temperature = 0.8f;
  const uint64_t seed = 98765;

  std::mt19937 rng(88);
  std::vector<float> logits(vocab);
  for (auto& v : logits) v = float(rng()) / float(rng.max()) * 2.f - 1.f;

  SamplingConfig cfg;
  cfg.temperature = temperature;
  cfg.top_p = 0.9f;
  cfg.seed = seed;

  oracle::Sampler ref(cfg);
  const int32_t cpu_id = ref.sample(logits.data(), vocab);

  Sampler sampler(cfg);
  float* dev = dev_upload(logits);
  const int32_t gpu_id = sampler.sample(dev, vocab);
  hipFree(dev);

  EXPECT_EQ(gpu_id, cpu_id);
}
