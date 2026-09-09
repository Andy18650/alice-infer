#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "alice/engine_config.hpp"
#include "alice/tensor.hpp"

namespace alice {

// FP32 logits -> token id (design §7.3). Config fixed at construction (maps
// to the session API's auxiliary new_session parameters); temperature 0 =
// greedy/argmax for tests. Owns the RNG — the first mutable non-weight state
// in the pipeline, acceptable under single-session.
class Sampler {
 public:
  explicit Sampler(SamplingConfig cfg) : cfg_(cfg), rng_(cfg.seed) {}

  // logits: fp32 [1, vocab].
  int32_t sample(F32View logits);

 private:
  SamplingConfig cfg_;
  std::mt19937_64 rng_;
  std::vector<float> work_;  // scratch, sized on first use
};

}  // namespace alice
