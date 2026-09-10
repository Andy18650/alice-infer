#include "dense_ffn.hpp"

#include <cstring>
#include <stdexcept>

#include "platform.hpp"

namespace alice::gpu_rocblas {

namespace {

#define HIP_CHECK(x)                                                                     \
  do {                                                                                   \
    const hipError_t e_ = (x);                                                           \
    if (e_ != hipSuccess)                                                                \
      throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(e_) +      \
                               " at " __FILE__ ":" + std::to_string(__LINE__));          \
  } while (0)
#define RB_CHECK(x)                                                                      \
  do {                                                                                   \
    if ((x) != rocblas_status_success)                                                   \
      throw std::runtime_error("rocBLAS error at " __FILE__ ":" +                        \
                               std::to_string(__LINE__));                                \
  } while (0)

// act = silu(gate) * up, bf16-rounded. gu is col-major [2I, S] (== row-major
// [S, 2I]); act col-major [I, S]. Truncating fp32->bf16 matches host
// bf16::from_f32. No includes: hipRTC compiles this standalone.
const char* kSiluMulSrc = R"KERNEL(
extern "C" __global__ void silu_mul(const float* gu, unsigned short* act,
                                    unsigned int I, unsigned long n) {
  const unsigned long idx = blockIdx.x * (unsigned long)blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const unsigned int s = (unsigned int)(idx / I);
  const unsigned int i = (unsigned int)(idx % I);
  const float* row = gu + (unsigned long)s * 2 * I;
  const float g = row[i];
  const float u = row[I + i];
  const float f = g / (1.0f + expf(-g)) * u;
  act[idx] = (unsigned short)(__float_as_uint(f) >> 16);
}
)KERNEL";

}  // namespace

DenseFfnBF16GpuRocblas::DenseFfnBF16GpuRocblas(BF16View w_gate, BF16View w_up,
                                               BF16View w_down, const ModelConfig& mc,
                                               const EngineConfig& ec)
    : d_(mc.hidden_size), inter_(mc.intermediate_size), mpc_(ec.max_prefill_chunk) {
  if (w_gate.rows != inter_ || w_gate.cols != d_ || w_up.rows != inter_ ||
      w_up.cols != d_ || w_down.rows != d_ || w_down.cols != inter_)
    throw std::invalid_argument("DenseFfnBF16GpuRocblas: weight shape mismatch");
  Platform& p = platform();
  HIP_CHECK(hipSetDevice(p.device));

  HIP_CHECK(hipMalloc(&dw_gup_, size_t(2) * inter_ * d_ * 2));
  HIP_CHECK(hipMalloc(&dw_down_, size_t(d_) * inter_ * 2));
  HIP_CHECK(hipMalloc(&dx_, size_t(d_) * mpc_ * 2));
  HIP_CHECK(hipMalloc(&dgu_, size_t(2) * inter_ * mpc_ * 4));
  HIP_CHECK(hipMalloc(&dact_, size_t(inter_) * mpc_ * 2));
  HIP_CHECK(hipMalloc(&dout_, size_t(d_) * mpc_ * 4));
  HIP_CHECK(hipHostMalloc(&hb_x_, size_t(mpc_) * d_ * 2, hipHostMallocDefault));
  HIP_CHECK(hipHostMalloc(&hb_out_, size_t(mpc_) * d_ * 4, hipHostMallocDefault));

  // Stage weights to the device; the host views are freed on return.
  HIP_CHECK(hipMemcpy(dw_gup_, w_gate.data, size_t(inter_) * d_ * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(static_cast<char*>(dw_gup_) + size_t(inter_) * d_ * 2, w_up.data,
                      size_t(inter_) * d_ * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dw_down_, w_down.data, size_t(d_) * inter_ * 2, hipMemcpyHostToDevice));

  rtc_kernel("silu_mul", kSiluMulSrc);  // compile now, not on first forward
}

DenseFfnBF16GpuRocblas::~DenseFfnBF16GpuRocblas() {
  hipFree(dw_gup_);
  hipFree(dw_down_);
  hipFree(dx_);
  hipFree(dgu_);
  hipFree(dact_);
  hipFree(dout_);
  if (hb_x_) hipHostFree(hb_x_);
  if (hb_out_) hipHostFree(hb_out_);
}

void DenseFfnBF16GpuRocblas::forward(F32View io) {
  const uint32_t S = io.rows;
  if (S == 0 || S > mpc_)
    throw std::runtime_error("DenseFfnBF16GpuRocblas::forward: S out of range");
  Platform& p = platform();
  HIP_CHECK(hipSetDevice(p.device));

  // x: host fp32 -> pinned bf16 (truncating conversion, matches from_f32)
#if defined(_OPENMP)
#pragma omp parallel for if (S > 8)
#endif
  for (uint32_t s = 0; s < S; ++s) {
    const float* xr = io.data + size_t(s) * io.stride();
    for (uint32_t i = 0; i < d_; ++i) hb_x_[size_t(s) * d_ + i] = bf16::from_f32(xr[i]);
  }
  HIP_CHECK(hipMemcpy(dx_, hb_x_, size_t(S) * d_ * 2, hipMemcpyHostToDevice));

  const float alpha = 1.0f, beta = 0.0f;
  // gate/up fused: C'[2I, S] = [Wg;Wu] . x^T  (col-major view of row-major)
  RB_CHECK(rocblas_gemm_ex(p.handle, rocblas_operation_transpose, rocblas_operation_none,
                           2 * inter_, S, d_, &alpha, dw_gup_, rocblas_datatype_bf16_r,
                           d_, dx_, rocblas_datatype_bf16_r, d_, &beta, dgu_,
                           rocblas_datatype_f32_r, 2 * inter_, dgu_,
                           rocblas_datatype_f32_r, 2 * inter_, rocblas_datatype_f32_r,
                           rocblas_gemm_algo_standard, 0, 0));

  // act = silu(gate) * up (device elementwise)
  const unsigned long n = (unsigned long)inter_ * S;
  unsigned int I = inter_;
  void* args[] = {&dgu_, &dact_, &I, const_cast<unsigned long*>(&n)};
  const unsigned int block = 256;
  HIP_CHECK(hipModuleLaunchKernel(rtc_kernel("silu_mul", kSiluMulSrc),
                                  (unsigned int)((n + block - 1) / block), 1, 1, block, 1,
                                  1, 0, nullptr, args, nullptr));

  // down: C'[d, S] = Wd . act^T
  RB_CHECK(rocblas_gemm_ex(p.handle, rocblas_operation_transpose, rocblas_operation_none,
                           d_, S, inter_, &alpha, dw_down_, rocblas_datatype_bf16_r,
                           inter_, dact_, rocblas_datatype_bf16_r, inter_, &beta, dout_,
                           rocblas_datatype_f32_r, d_, dout_, rocblas_datatype_f32_r, d_,
                           rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));

  HIP_CHECK(hipMemcpy(hb_out_, dout_, size_t(S) * d_ * 4, hipMemcpyDeviceToHost));
#if defined(_OPENMP)
#pragma omp parallel for if (S > 8)
#endif
  for (uint32_t s = 0; s < S; ++s)
    std::memcpy(io.data + size_t(s) * io.stride(), hb_out_ + size_t(s) * d_,
                d_ * sizeof(float));
}

}  // namespace alice::gpu_rocblas
