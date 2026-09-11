#include "lm_head.hpp"

#include <stdexcept>
#include <string>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include "../blocks/gpu_rocblas/platform.hpp"
#include "kernels.hpp"

namespace alice::gpu {

namespace {
#define HIP_CHECK(x)                                                                    \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess)                                                               \
      throw std::runtime_error(std::string("gpu::LmHead: ") + #x + " failed: " +       \
                               hipGetErrorString(e_));                                  \
  } while (0)
#define RB_CHECK(x)                                                                     \
  do {                                                                                  \
    rocblas_status s_ = (x);                                                            \
    if (s_ != rocblas_status_success)                                                   \
      throw std::runtime_error(std::string("gpu::LmHead: rocBLAS error ") +             \
                               std::to_string(int(s_)));                                \
  } while (0)
}  // namespace

LmHead::LmHead(BF16View weight, uint32_t max_prefill_chunk)
    : vocab_(weight.rows), d_(weight.cols), mpc_(max_prefill_chunk) {
  gpu_rocblas::Platform& p = gpu_rocblas::platform();
  HIP_CHECK(hipSetDevice(p.device));
  HIP_CHECK(hipMalloc(&w_, weight.nbytes()));
  HIP_CHECK(hipMalloc(&dx_, size_t(mpc_) * d_ * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&logits_, size_t(mpc_) * vocab_ * sizeof(float)));
  HIP_CHECK(hipMemcpy(w_, weight.data, weight.nbytes(), hipMemcpyHostToDevice));
}

LmHead::~LmHead() {
  (void)hipFree(w_);
  (void)hipFree(dx_);
  (void)hipFree(logits_);
}

const float* LmHead::project_last(const float* io, uint32_t S) {
  if (S == 0 || S > mpc_) throw std::runtime_error("gpu::LmHead: S out of range");
  gpu_rocblas::Platform& p = gpu_rocblas::platform();
  HIP_CHECK(hipSetDevice(p.device));
  kernels::f32_to_bf16(io + size_t(S - 1) * d_, dx_, d_);
  const float alpha = 1.0f, beta = 0.0f;
  // logits[V] = W . x  (C'[V, 1] col-major)
  RB_CHECK(rocblas_gemm_ex(p.handle, rocblas_operation_transpose, rocblas_operation_none,
                           vocab_, 1, d_, &alpha, w_, rocblas_datatype_bf16_r, d_, dx_,
                           rocblas_datatype_bf16_r, d_, &beta, logits_,
                           rocblas_datatype_f32_r, vocab_, logits_,
                           rocblas_datatype_f32_r, vocab_, rocblas_datatype_f32_r,
                           rocblas_gemm_algo_standard, 0, 0));
  return logits_;
}

const float* LmHead::project_all(const float* io, uint32_t S) {
  if (S == 0 || S > mpc_) throw std::runtime_error("gpu::LmHead: S out of range");
  gpu_rocblas::Platform& p = gpu_rocblas::platform();
  HIP_CHECK(hipSetDevice(p.device));
  kernels::f32_to_bf16(io, dx_, size_t(S) * d_);
  const float alpha = 1.0f, beta = 0.0f;
  // C'[V, S] col-major = row-major [S, V]
  RB_CHECK(rocblas_gemm_ex(p.handle, rocblas_operation_transpose, rocblas_operation_none,
                           vocab_, S, d_, &alpha, w_, rocblas_datatype_bf16_r, d_, dx_,
                           rocblas_datatype_bf16_r, d_, &beta, logits_,
                           rocblas_datatype_f32_r, vocab_, logits_,
                           rocblas_datatype_f32_r, vocab_, rocblas_datatype_f32_r,
                           rocblas_gemm_algo_standard, 0, 0));
  return logits_;
}

}  // namespace alice::gpu
