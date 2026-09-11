#pragma once

// GPU-resident Sampler: temperature 0 (tests, greedy) argmaxes on device —
// four bytes cross back per step. Nonzero temperatures D2H the logits and
// reuse the host Sampler (policy code stays on the host; interim until
// on-device sampling lands with the RNG design).

#include <cstdint>
#include <vector>

#include "alice/sampler.hpp"

namespace alice::gpu {

class Sampler {
 public:
  explicit Sampler(const SamplingConfig& cfg);
  ~Sampler();

  // logits: device fp32 [vocab]. Returns the sampled token id (host).
  int32_t sample(const float* logits, uint32_t vocab);

 private:
  SamplingConfig cfg_;
  alice::Sampler host_sampler_;     // temperature path
  std::vector<float> logits_host_;  // D2H staging for the temperature path
  int32_t* id_dev_ = nullptr;
};

}  // namespace alice::gpu
