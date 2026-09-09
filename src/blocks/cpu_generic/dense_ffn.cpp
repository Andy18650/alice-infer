#include "dense_ffn.hpp"

#include <cstring>
#include <stdexcept>

#include "kernels.hpp"

namespace alice::cpu_generic {

namespace {

// Copies a validated weight view into a fresh private buffer; returns the
// buffer and sets `view` to describe the copy.
std::unique_ptr<std::byte[]> copy_weight(const TensorView& src, TensorView& view) {
  if (src.dtype != DType::BF16)
    throw std::invalid_argument("DenseFfnBF16CpuGeneric: weights must be bf16");
  if (src.stride() != src.cols)
    throw std::invalid_argument("DenseFfnBF16CpuGeneric: strided weights unsupported");
  auto buf = std::make_unique<std::byte[]>(src.nbytes());
  std::memcpy(buf.get(), src.data, src.nbytes());
  view = TensorView{buf.get(), src.dtype, src.rows, src.cols, 0};
  return buf;
}

}  // namespace

DenseFfnBF16CpuGeneric::DenseFfnBF16CpuGeneric(TensorView w_gate, TensorView w_up,
                                               TensorView w_down, const ModelConfig& mc,
                                               const EngineConfig& ec)
    : d_(mc.hidden_size), inter_(mc.intermediate_size), mpc_(ec.max_prefill_chunk) {
  if (w_gate.rows != inter_ || w_gate.cols != d_ || w_up.rows != inter_ ||
      w_up.cols != d_ || w_down.rows != d_ || w_down.cols != inter_)
    throw std::invalid_argument("DenseFfnBF16CpuGeneric: weight shape mismatch vs config");
  wg_ = copy_weight(w_gate, wgv_);
  wu_ = copy_weight(w_up, wuv_);
  wd_ = copy_weight(w_down, wdv_);
  gate_ = std::make_unique<float[]>(size_t(mpc_) * inter_);
  up_ = std::make_unique<float[]>(size_t(mpc_) * inter_);
}

void DenseFfnBF16CpuGeneric::forward(TensorView io) {
  const uint32_t S = io.rows;
  if (S > mpc_) throw std::runtime_error("DenseFfnBF16CpuGeneric: prefill chunk exceeded");
  if (io.cols != d_ || io.dtype != DType::F32)
    throw std::invalid_argument("DenseFfnBF16CpuGeneric: io must be fp32 [S, d]");

  TensorView gate{gate_.get(), DType::F32, S, inter_, 0};
  TensorView up{up_.get(), DType::F32, S, inter_, 0};
  matmul_wt(io, wgv_, gate);
  matmul_wt(io, wuv_, up);
  for (uint32_t s = 0; s < S; ++s) {
    float* g = gate.as<float>() + size_t(s) * inter_;
    float* u = up.as<float>() + size_t(s) * inter_;
    for (uint32_t i = 0; i < inter_; ++i) u[i] = silu(g[i]) * u[i];
  }
  matmul_wt(up, wdv_, io);  // reads up_, writes io: no aliasing
}

}  // namespace alice::cpu_generic
