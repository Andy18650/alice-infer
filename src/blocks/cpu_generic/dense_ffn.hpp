#pragma once

// CPU reference (naive oracle) dense FFN — development-only reference
// implementation (design §3/§8.1): the test oracle for kernels/blocks and
// the compute core of gpu::CpuReferenceBlock. Never production compute.
// Weights kept in bf16 (storage contract), fp32 accumulation in matmul.

#include <memory>

#include "alice/engine_config.hpp"
#include "alice/manifest.hpp"
#include "alice/tensor.hpp"

namespace alice::cpu_generic {

class DenseFfnBF16CpuGeneric final {
 public:
  // Views are validated staging buffers, considered freed on return.
  DenseFfnBF16CpuGeneric(BF16View w_gate, BF16View w_up, BF16View w_down,
                         const ModelConfig& mc, const EngineConfig& ec);

  // io: fp32 [S, d] host, in-place (block output overwrites input).
  void forward(F32View io);

 private:
  uint32_t d_ = 0, inter_ = 0, mpc_ = 0;

  std::unique_ptr<bf16[]> wg_, wu_, wd_;  // private copies
  BF16View wgv_, wuv_, wdv_;

  std::unique_ptr<float[]> gate_, up_;  // transients [mpc, I] fp32 (up_ reused for act)
};

}  // namespace alice::cpu_generic
