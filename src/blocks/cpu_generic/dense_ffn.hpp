#pragma once

// CPU reference (naive oracle) dense FFN — development-only reference
// implementation (design §3/§8.1): the test oracle and the compute core of
// gpu::CpuReferenceBlock. Never production compute.
// Weights bf16 [inter, d] x2 + [d, inter]; SiLU-gated MLP.

#include <memory>

#include "alice/engine_config.hpp"
#include "../../manifest.hpp"
#include "alice/tensor.hpp"

namespace alice::cpu_generic {

class DenseFfnBF16CpuGeneric final {
 public:
  // Takes ownership of the staged weights (moved in, shape-checked).
  DenseFfnBF16CpuGeneric(Tensor<bf16> w_gate, Tensor<bf16> w_up, Tensor<bf16> w_down,
                         const ModelConfig& mc, const EngineConfig& ec);

  // io: fp32 [S, d] host, in-place (block output overwrites input).
  void forward(float* io, uint32_t S);

 private:
  uint32_t d_ = 0, inter_ = 0, mpc_ = 0;
  Tensor<bf16> wg_, wu_, wd_;  // private weights (moved in)

  std::unique_ptr<float[]> gate_, up_;  // transients [mpc, inter]
};

}  // namespace alice::cpu_generic
