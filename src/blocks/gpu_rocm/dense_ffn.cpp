#include "dense_ffn.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include "platform.hpp"
#include "kernels.hpp"

namespace alice::gpu_rocm {

namespace {
#define HIP_CHECK(x)                                                                    \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess)                                                               \
      throw std::runtime_error(std::string("gpu::DenseFfnBF16: ") + #x + " failed: " + \
                               hipGetErrorString(e_));                                  \
  } while (0)
#define RB_CHECK(x)                                                                     \
  do {                                                                                  \
    rocblas_status s_ = (x);                                                            \
    if (s_ != rocblas_status_success)                                                   \
      throw std::runtime_error(std::string("gpu::DenseFfnBF16: rocBLAS error ") +       \
                               std::to_string(int(s_)));                                \
  } while (0)
}  // namespace

DenseFfnBF16::DenseFfnBF16(Tensor<bf16> w_gate, Tensor<bf16> w_up, Tensor<bf16> w_down,
                           const ModelConfig& mc, const EngineConfig& ec)
    : d_(mc.hidden_size), inter_(mc.intermediate_size), mpc_(ec.max_prefill_chunk) {
  if (w_gate.rows() != inter_ || w_gate.cols() != d_ || w_up.rows() != inter_ ||
      w_up.cols() != d_ || w_down.rows() != d_ || w_down.cols() != inter_)
    throw std::invalid_argument("gpu::DenseFfnBF16: weight shape mismatch vs config");

  Platform& p = platform();
  HIP_CHECK(hipSetDevice(p.device));

  // Fuse [Wg;Wu] on host, one H2D (same layout the pre-migration block used).
  std::vector<bf16> gup(size_t(2) * inter_ * d_);
  std::memcpy(gup.data(), w_gate.data(), size_t(inter_) * d_ * sizeof(bf16));
  std::memcpy(gup.data() + size_t(inter_) * d_, w_up.data(), size_t(inter_) * d_ * sizeof(bf16));

  HIP_CHECK(hipMalloc(&dw_gup_, gup.size() * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&dw_down_, size_t(d_) * inter_ * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&dx_, size_t(mpc_) * d_ * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&dgu_, size_t(mpc_) * 2 * inter_ * sizeof(float)));
  HIP_CHECK(hipMalloc(&dact_, size_t(mpc_) * inter_ * sizeof(bf16)));
  HIP_CHECK(hipMemcpy(dw_gup_, gup.data(), gup.size() * sizeof(bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dw_down_, w_down.data(), size_t(d_) * inter_ * sizeof(bf16),
                      hipMemcpyHostToDevice));
}

DenseFfnBF16::~DenseFfnBF16() {
  (void)hipFree(dw_gup_);
  (void)hipFree(dw_down_);
  (void)hipFree(dx_);
  (void)hipFree(dgu_);
  (void)hipFree(dact_);
}

void DenseFfnBF16::forward(float* io, uint32_t S) {
  if (S == 0 || S > mpc_)
    throw std::runtime_error("gpu::DenseFfnBF16::forward: S out of range");
  Platform& p = platform();
  HIP_CHECK(hipSetDevice(p.device));

  kernels::f32_to_bf16(io, dx_, size_t(S) * d_);
  const float alpha = 1.0f, beta = 0.0f;
  // gate/up fused: C'[2I, S] = [Wg;Wu] . x^T (col-major view of row-major)
  RB_CHECK(rocblas_gemm_ex(p.handle, rocblas_operation_transpose, rocblas_operation_none,
                           2 * inter_, S, d_, &alpha, dw_gup_, rocblas_datatype_bf16_r, d_,
                           dx_, rocblas_datatype_bf16_r, d_, &beta, dgu_,
                           rocblas_datatype_f32_r, 2 * inter_, dgu_,
                           rocblas_datatype_f32_r, 2 * inter_, rocblas_datatype_f32_r,
                           rocblas_gemm_algo_standard, 0, 0));
  kernels::silu_mul(dgu_, dact_, S, inter_);
  // down, straight into io: C'[d, S] = Wd . act^T
  RB_CHECK(rocblas_gemm_ex(p.handle, rocblas_operation_transpose, rocblas_operation_none,
                           d_, S, inter_, &alpha, dw_down_, rocblas_datatype_bf16_r,
                           inter_, dact_, rocblas_datatype_bf16_r, inter_, &beta, io,
                           rocblas_datatype_f32_r, d_, io, rocblas_datatype_f32_r, d_,
                           rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
}

}  // namespace alice::gpu
