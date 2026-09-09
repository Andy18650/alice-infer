#pragma once

#include <memory>

#include "alice/block.hpp"
#include "alice/engine_config.hpp"
#include "alice/manifest.hpp"
#include "alice/tensor.hpp"

namespace alice {

// Abstract block class for (GQA causal self-attention) × (bf16 weights).
// Stateful: owns the KV cache; position is a private counter (design §6).
//
// Constructor contract, IDENTICAL for every implementation:
//   (TensorView wq, TensorView wk, TensorView wv, TensorView wo,
//    const ModelConfig&, const EngineConfig&)
// Weights arrive validated, bf16, shapes [nh*hd, d] / [nkv*hd, d] /
// [nkv*hd, d] / [d, nh*hd]; considered FREED when the constructor returns.
class GqaAttentionBF16 : public StatefulBlock {
 public:
  static std::unique_ptr<GqaAttentionBF16> create(HardwareClass hw, TensorView wq,
                                                  TensorView wk, TensorView wv,
                                                  TensorView wo, const ModelConfig& mc,
                                                  const EngineConfig& ec);
};

}  // namespace alice
