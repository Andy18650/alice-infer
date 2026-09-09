#include "alice/dense_ffn_bf16.hpp"

#include <stdexcept>

#include "cpu_generic/dense_ffn.hpp"

namespace alice {

std::unique_ptr<DenseFfnBF16> DenseFfnBF16::create(HardwareClass hw, BF16View w_gate,
                                                   BF16View w_up, BF16View w_down,
                                                   const ModelConfig& mc,
                                                   const EngineConfig& ec) {
  switch (hw) {
    case HardwareClass::CPU_GENERIC:
      return std::make_unique<cpu_generic::DenseFfnBF16CpuGeneric>(w_gate, w_up, w_down, mc,
                                                                   ec);
    default:
      throw std::invalid_argument(
          "DenseFfnBF16::create: hardware class not implemented (fallback chain ends at "
          "CPU_GENERIC)");
  }
}

}  // namespace alice
