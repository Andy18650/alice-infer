#pragma once

// GPU-resident Sampler: temperature 0 (greedy) argmaxes on device —
// four bytes cross back per step. Nonzero temperatures D2H the logits
// and sample on the host (policy code stays on the host; interim until
// on-device sampling lands with the RNG design).

#include <cstdint>
#include <random>
#include <vector>

#include "alice/engine_config.hpp"

#include <hip/hip_runtime_api.h>

namespace alice::gpu {

class Sampler {
 public:
  explicit Sampler(const SamplingConfig& cfg);
  ~Sampler();

  // logits: device fp32 [vocab]. Returns the sampled token id (host).
  int32_t sample(const float* logits, uint32_t vocab);

 private:
  // Host-side sampling (temperature > 0 path). logits: host fp32 [vocab].
  int32_t sample_host(const float* logits, uint32_t vocab);

  SamplingConfig cfg_;
  std::mt19937_64 rng_;
  std::vector<float> work_;        // scratch for host sampling, sized on first use
  std::vector<float> logits_host_; // D2H staging for the temperature path
  int32_t* id_dev_ = nullptr;
};

}  // namespace alice::gpu
