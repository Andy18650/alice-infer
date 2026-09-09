#pragma once

#include <memory>

#include "alice/block.hpp"
#include "alice/engine_config.hpp"
#include "alice/manifest.hpp"
#include "alice/tensor.hpp"

namespace alice {

// Abstract block class for (dense SwiGLU FFN) × (bf16 weights) — design §7.1:
// one abstract class per block type × weight dtype.
//
// Constructor contract, IDENTICAL for every implementation:
//   (TensorView w_gate, TensorView w_up, TensorView w_down,
//    const ModelConfig&, const EngineConfig&)
// Weights arrive validated (manifest audit), bf16, shapes
// [I, d] / [I, d] / [d, I]. The constructor copies/converts/packs them into
// private buffers; the passed views are considered FREED on return.
class DenseFfnBF16 : public Block {
 public:
  // Factory with hardware-class fallback chain (CPU_GENERIC only for now).
  static std::unique_ptr<DenseFfnBF16> create(HardwareClass hw, TensorView w_gate,
                                              TensorView w_up, TensorView w_down,
                                              const ModelConfig& mc,
                                              const EngineConfig& ec);
};

}  // namespace alice
