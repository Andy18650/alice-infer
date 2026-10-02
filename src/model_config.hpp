#pragma once

// Model architecture config (design §3): the model's static shape knobs —
// hidden_size, num_heads, head_dim, rms_norm_eps, rope, token ids. Used by
// every block constructor + the block factories (built from the descriptor's
// per-spec config) + the REPL (bos/eos). This is a BLOCK-LAYER concept, not a
// manifest concept: it lived in manifest.hpp historically and is relocated
// here so the block layer doesn't drag in the manifest (and transitively
// safetensors) just for these shapes.

#include <cstdint>
#include <vector>

namespace alice {

struct RopeScaling {  // llama3-type scaling; presence flagged in ModelConfig
  float factor = 1.0f;
  float low_freq_factor = 1.0f;
  float high_freq_factor = 1.0f;
  uint32_t original_max_position_embeddings = 0;
};

struct ModelConfig {
  uint32_t hidden_size = 0;
  uint32_t intermediate_size = 0;
  // No num_layers field (2026-09 round 1 of the manifest diet): it was
  // written by the generator, parsed, and consumed by nothing — the block
  // list IS the layer composition; pre_norms count is validated against
  // blocks.size().
  uint32_t num_heads = 0;
  uint32_t num_kv_heads = 0;
  uint32_t head_dim = 0;
  uint32_t vocab_size = 0;
  float rms_norm_eps = 1e-5f;
  float rope_theta = 500000.0f;
  bool has_rope_scaling = false;
  RopeScaling rope_scaling;
  // Token metadata (frontend concern; engine ignores them). Optional.
  uint32_t bos_token_id = 128000;
  std::vector<uint32_t> eos_token_ids;
};

}  // namespace alice
