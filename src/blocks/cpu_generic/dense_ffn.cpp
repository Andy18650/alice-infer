#include "dense_ffn.hpp"

#include <stdexcept>

#include "kernels.hpp"

namespace alice::cpu_generic {

DenseFfnBF16CpuGeneric::DenseFfnBF16CpuGeneric(Tensor<bf16> w_gate, Tensor<bf16> w_up,
                                               Tensor<bf16> w_down, const ModelConfig& mc,
                                               const EngineConfig& ec)
    : d_(mc.hidden_size), inter_(mc.intermediate_size), mpc_(ec.max_prefill_chunk) {
  if (w_gate.rows() != inter_ || w_gate.cols() != d_ || w_up.rows() != inter_ ||
      w_up.cols() != d_ || w_down.rows() != d_ || w_down.cols() != inter_)
    throw std::invalid_argument("DenseFfnBF16CpuGeneric: weight shape mismatch vs config");
  wg_ = std::move(w_gate);
  wu_ = std::move(w_up);
  wd_ = std::move(w_down);
  gate_ = std::make_unique<float[]>(size_t(mpc_) * inter_);
  up_ = std::make_unique<float[]>(size_t(mpc_) * inter_);
}

void DenseFfnBF16CpuGeneric::forward(float* io, uint32_t S) {
  if (S > mpc_) throw std::runtime_error("DenseFfnBF16CpuGeneric: prefill chunk exceeded");

  float* gate = gate_.get();
  float* up = up_.get();
  matmul_wt(io, wg_.data(), gate, S, d_, inter_);
  matmul_wt(io, wu_.data(), up, S, d_, inter_);
  // elementwise activation: silu(gate) * up
  ALICE_OMP_PARALLEL_FOR
  for (size_t e = 0; e < size_t(S) * inter_; ++e) up[e] = silu(gate[e]) * up[e];
  matmul_wt(up, wd_.data(), io, S, inter_, d_);  // reads up, writes io: no aliasing
}

}  // namespace alice::cpu_generic
