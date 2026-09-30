#include "dense_ffn.hpp"

#include <stdexcept>
#include <string>

#include <hip/hip_runtime_api.h>

#include "kernels.hpp"

namespace alice::cpu_generic {

namespace {
#define HIP_CHECK(x)                                                                    \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess)                                                               \
      throw std::runtime_error(std::string("cpu_generic::DenseFfnBF16: ") + #x +       \
                               " failed: " + hipGetErrorString(e_));                    \
  } while (0)
}  // namespace

DenseFfnBF16::DenseFfnBF16(Tensor<bf16> w_gate, Tensor<bf16> w_up,
                           Tensor<bf16> w_down, const ModelConfig& mc,
                           const EngineConfig& ec)
    : d_(mc.hidden_size), inter_(mc.intermediate_size), mpc_(ec.max_prefill_chunk) {
  if (w_gate.rows() != inter_ || w_gate.cols() != d_ || w_up.rows() != inter_ ||
      w_up.cols() != d_ || w_down.rows() != d_ || w_down.cols() != inter_)
    throw std::invalid_argument("DenseFfnBF16: weight shape mismatch vs config");
  wg_ = std::move(w_gate);
  wu_ = std::move(w_up);
  wd_ = std::move(w_down);
  gate_ = std::make_unique<float[]>(size_t(mpc_) * inter_);
  up_ = std::make_unique<float[]>(size_t(mpc_) * inter_);
  host_io_.resize(size_t(mpc_) * d_);
}

void DenseFfnBF16::forward(float* io, uint32_t S) {
  if (S == 0 || S > mpc_)
    throw std::runtime_error("cpu_generic::DenseFfnBF16::forward: S out of range");
  const size_t n = size_t(S) * d_;
  HIP_CHECK(hipMemcpy(host_io_.data(), io, n * sizeof(float), hipMemcpyDeviceToHost));

  float* gate = gate_.get();
  float* up = up_.get();
  matmul_wt(host_io_.data(), wg_.data(), gate, S, d_, inter_);
  matmul_wt(host_io_.data(), wu_.data(), up, S, d_, inter_);
  ALICE_OMP_PARALLEL_FOR
  for (size_t e = 0; e < size_t(S) * inter_; ++e) up[e] = silu(gate[e]) * up[e];
  matmul_wt(up, wd_.data(), host_io_.data(), S, inter_, d_);

  HIP_CHECK(hipMemcpy(io, host_io_.data(), n * sizeof(float), hipMemcpyHostToDevice));
}

#undef HIP_CHECK

}  // namespace alice::cpu_generic
