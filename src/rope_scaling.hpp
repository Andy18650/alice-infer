#pragma once

// llama3-type RoPE frequency-scaling params (design §3). A parameter of
// host_math::build_rope_tables (the function that interprets these fields)
// and the optional payload of GqaAttentionConfig::rope_scaling. Leaf header
// (no deps) so the host-math oracle and the attention block config can both
// include it without pulling each other in.

#include <cstdint>

namespace alice {

struct RopeScaling {
  float factor = 1.0f;
  float low_freq_factor = 1.0f;
  float high_freq_factor = 1.0f;
  uint32_t original_max_position_embeddings = 0;
};

}  // namespace alice
