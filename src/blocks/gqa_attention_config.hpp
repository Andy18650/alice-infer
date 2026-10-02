#pragma once

// gqa_attention block-type config (design §8): the static shapes + RoPE
// params a GQA-attention block needs. Impl-agnostic: every implementation
// of gqa_attention (gpu_rocm today, future archs later) takes this from its
// factory. This is the block TYPE's contract, not any one impl's, so it
// lives at the block layer (not under gpu_rocm/) and a second impl can
// reuse it without depending on the first. Plain POD; the JSON binding
// (NLOHMANN_DEFINE, minus the conditional rope_scaling group which is read
// manually) lives in the factory .cpp.

#include <cstdint>
#include <optional>

#include "../rope_scaling.hpp"  // alice::RopeScaling

namespace alice::block {

struct GqaAttentionConfig {
  uint32_t io_dim = 0;       // hidden_size
  uint32_t qk_dim = 0;       // query/key head dim
  uint32_t v_dim = 0;        // value head dim (v1: == qk_dim)
  uint32_t num_q_heads = 0;
  uint32_t num_kv_heads = 0;
  double rope_theta = 0.0;
  // Absent (nullopt) => no llama3 frequency scaling. Conditional JSON, so
  // it is populated manually in the factory (not via NLOHMANN_DEFINE).
  std::optional<RopeScaling> rope_scaling;
};

}  // namespace alice::block
