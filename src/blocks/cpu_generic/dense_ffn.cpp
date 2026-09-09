#include "dense_ffn.hpp"

#include <cstring>
#include <stdexcept>

#include "kernels.hpp"

namespace alice::cpu_generic {

namespace {

// Copies a validated weight view into a fresh private buffer; returns the
// buffer and sets `view` to describe the copy.
template <typename T>
std::unique_ptr<T[]> copy_weight(TView<T> src, TView<T>& view, const char* what) {
  if (src.stride() != src.cols)
    throw std::invalid_argument(std::string(what) + ": strided weights unsupported");
  auto buf = std::make_unique<T[]>(src.numel());
  std::memcpy(buf.get(), src.data, src.nbytes());
  view = TView<T>{buf.get(), src.rows, src.cols, 0};
  return buf;
}

}  // namespace

DenseFfnBF16CpuGeneric::DenseFfnBF16CpuGeneric(BF16View w_gate, BF16View w_up,
                                               BF16View w_down, const ModelConfig& mc,
                                               const EngineConfig& ec)
    : d_(mc.hidden_size), inter_(mc.intermediate_size), mpc_(ec.max_prefill_chunk) {
  if (w_gate.rows != inter_ || w_gate.cols != d_ || w_up.rows != inter_ ||
      w_up.cols != d_ || w_down.rows != d_ || w_down.cols != inter_)
    throw std::invalid_argument("DenseFfnBF16CpuGeneric: weight shape mismatch vs config");
  wg_ = copy_weight(w_gate, wgv_, "DenseFfnBF16CpuGeneric");
  wu_ = copy_weight(w_up, wuv_, "DenseFfnBF16CpuGeneric");
  wd_ = copy_weight(w_down, wdv_, "DenseFfnBF16CpuGeneric");
  gate_ = std::make_unique<float[]>(size_t(mpc_) * inter_);
  up_ = std::make_unique<float[]>(size_t(mpc_) * inter_);
}

void DenseFfnBF16CpuGeneric::forward(F32View io) {
  const uint32_t S = io.rows;
  if (S > mpc_) throw std::runtime_error("DenseFfnBF16CpuGeneric: prefill chunk exceeded");
  if (io.cols != d_)
    throw std::invalid_argument("DenseFfnBF16CpuGeneric: io must be [S, d]");

  F32View gate{gate_.get(), S, inter_, 0};
  F32View up{up_.get(), S, inter_, 0};
  matmul_wt(io, wgv_, gate);
  matmul_wt(io, wuv_, up);
  // elementwise activation (buffers are contiguous): silu(gate) * up
  ALICE_OMP_PARALLEL_FOR
  for (size_t e = 0; e < size_t(S) * inter_; ++e) up_[e] = silu(gate_[e]) * up_[e];
  matmul_wt(up, wdv_, io);  // reads up_, writes io: no aliasing
}

}  // namespace alice::cpu_generic
