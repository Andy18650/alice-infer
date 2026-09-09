#include "alice/gqa_attention_bf16.hpp"

#include <stdexcept>

#include "cpu_generic/gqa_attention.hpp"

namespace alice {

std::unique_ptr<GqaAttentionBF16> GqaAttentionBF16::create(HardwareClass hw, BF16View wq,
                                                           BF16View wk, BF16View wv,
                                                           BF16View wo,
                                                           const ModelConfig& mc,
                                                           const EngineConfig& ec) {
  switch (hw) {
    // Fallback chain (design §7.1): a hardware class with no attention
    // implementation yet resolves to the best available one below it.
    case HardwareClass::CPU_AVX2:  // no AVX2 attention impl yet
      [[fallthrough]];
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
