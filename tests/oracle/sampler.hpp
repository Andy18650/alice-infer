#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "alice/engine_config.hpp"

// Frozen from git at the commit before the gpu::Sampler merge (HEAD~1).
// The oracle tests compare gpu::Sampler against THIS code to catch
// transcription bugs in the merge.

namespace oracle {

// FP32 logits -> token id (design §7.3). Config fixed at construction (maps
// to the session API's auxiliary new_session parameters); temperature 0 =
// greedy/argmax for tests. Owns the RNG — the first mutable non-weight state
// in the pipeline, acceptable under single-session.
class Sampler {
 public:
  explicit Sampler(alice::SamplingConfig cfg) : cfg_(cfg), rng_(cfg.seed) {}

  // logits: fp32 [vocab] (borrowed host memory).
  int32_t sample(const float* logits, uint32_t vocab);

 private:
  alice::SamplingConfig cfg_;
  std::mt19937_64 rng_;
  std::vector<float> work_;  // scratch, sized on first use
};

}  // namespace oracle
