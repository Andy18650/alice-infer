#pragma once

// llama3-type RoPE frequency-scaling params (design §3). A parameter of the
// RoPE table builder (prod: src/blocks/gpu_rocm/gqa_attention.cpp) and the
// optional payload of GqaAttentionConfig::rope_scaling. Leaf header (no deps)
// so the prod block and the attention config can both include it without
// pulling each other in.

#include <cstdint>

namespace alice {

struct RopeScaling {
  float factor = 1.0f;
  float low_freq_factor = 1.0f;
  float high_freq_factor = 1.0f;
  uint32_t original_max_position_embeddings = 0;
};

}  // namespace alice
