#include "alice/gqa_attention_bf16.hpp"

#include <stdexcept>

#include "cpu_generic/gqa_attention.hpp"

namespace alice {

std::unique_ptr<GqaAttentionBF16> GqaAttentionBF16::create(HardwareClass hw, TensorView wq,
                                                           TensorView wk, TensorView wv,
                                                           TensorView wo,
                                                           const ModelConfig& mc,
                                                           const EngineConfig& ec) {
  switch (hw) {
    case HardwareClass::CPU_GENERIC:
      return std::make_unique<cpu_generic::GqaAttentionBF16CpuGeneric>(wq, wk, wv, wo, mc,
                                                                       ec);
    default:
      throw std::invalid_argument(
          "GqaAttentionBF16::create: hardware class not implemented (fallback chain ends "
          "at CPU_GENERIC)");
  }
}

}  // namespace alice
