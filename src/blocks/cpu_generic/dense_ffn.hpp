#pragma once

// CPU reference (naive oracle) dense FFN — development-only reference
// implementation (design §3/§8.1): the test oracle and the dev-only
// ALICE_CPU_REF_BLOCKS swap-in.  Never production compute.
// Weights bf16 [inter, d] x2 + [d, inter]; SiLU-gated MLP.
//
// Inherits gpu::Block so it can be placed directly into the GPU-resident
// model loop; forward() does an internal D2H -> host compute -> H2D round
// trip (the same work the former CpuReferenceBlock adapter did).

#include <memory>
#include <vector>

#include "alice/engine_config.hpp"
#include "../../gpu/block.hpp"
#include "../../manifest.hpp"
#include "alice/tensor.hpp"

namespace alice::cpu_generic {

class DenseFfnBF16 final : public gpu::Block {
 public:
  // Takes ownership of the staged weights (moved in, shape-checked).
  DenseFfnBF16(Tensor<bf16> w_gate, Tensor<bf16> w_up, Tensor<bf16> w_down,
               const ModelConfig& mc, const EngineConfig& ec);

  // io: fp32 [S, d] device pointer, in-place (block output overwrites input).
  // Internally: D2H staging, host compute, H2D result.
  void forward(float* io, uint32_t S) override;

 private:
  uint32_t d_ = 0, inter_ = 0, mpc_ = 0;
  Tensor<bf16> wg_, wu_, wd_;  // private weights (moved in)

  std::unique_ptr<float[]> gate_, up_;  // transients [mpc, inter]
  std::vector<float> host_io_;          // D2H/H2D staging [mpc, d]
};

}  // namespace alice::cpu_generic
