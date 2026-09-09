#pragma once

// CPU_GENERIC (naive oracle) implementation of DenseFfnBF16.
// Weights kept in bf16 (storage contract), fp32 accumulation in matmul.

#include "alice/dense_ffn_bf16.hpp"

namespace alice::cpu_generic {

class DenseFfnBF16CpuGeneric final : public DenseFfnBF16 {
 public:
  // Implements the abstract-class constructor contract; views freed on return.
  DenseFfnBF16CpuGeneric(TensorView w_gate, TensorView w_up, TensorView w_down,
                         const ModelConfig& mc, const EngineConfig& ec);

  void forward(TensorView io) override;

 private:
  uint32_t d_ = 0, inter_ = 0, mpc_ = 0;

  std::unique_ptr<std::byte[]> wg_, wu_, wd_;  // private bf16 copies
  TensorView wgv_, wuv_, wdv_;

  std::unique_ptr<float[]> gate_, up_;  // transients [mpc, I] fp32 (up_ reused for act)
};

}  // namespace alice::cpu_generic
