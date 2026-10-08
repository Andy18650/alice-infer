#pragma once

#include <cstdint>

namespace alice {

struct SamplingConfig {
  float temperature = 0.0f;  // 0 = greedy (argmax); tests use this
  uint32_t top_k = 0;        // 0 = disabled
  float top_p = 1.0f;        // 1.0 = disabled
  uint64_t seed = 0;
};

// Construction-time, engine-wide parameters (design §8.1: packaging, not a
// designed object yet).
struct ModelConfig {
  uint32_t max_seq_len = 8192;        // state sized for this; exceeding = hard error
  uint32_t max_prefill_chunk = 512;   // forward() with more tokens = hard error
  uint32_t tp_degree = 1;             // tensor-parallel degree (1 = single-GCD)
  bool decode_graph = true;           // S=1 graph replay; false = eager A/B baseline
};

}  // namespace alice
