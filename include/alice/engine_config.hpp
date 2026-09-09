#pragma once

#include <cstdint>

#include "alice/block.hpp"

namespace alice {

struct SamplingConfig {
  float temperature = 0.0f;  // 0 = greedy (argmax); tests use this
  uint32_t top_k = 0;        // 0 = disabled
  float top_p = 1.0f;        // 1.0 = disabled
  uint64_t seed = 0;
};

// Construction-time engine parameters (design §7.1: passed to the
// factory/loader; the struct is just packaging).
struct EngineConfig {
  uint32_t max_seq_len = 8192;        // state sized for this; exceeding = hard error
  uint32_t max_prefill_chunk = 512;   // forward() with more tokens = hard error
  HardwareClass hardware = HardwareClass::CPU_GENERIC;
  SamplingConfig sampling;
};

}  // namespace alice
