#include "dense_ffn.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include "../../gpu/resources.hpp"
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

DenseFfnBF16::DenseFfnBF16(TensorView<bf16> w_gate, TensorView<bf16> w_up, TensorView<bf16> w_down,
                           const block::DenseFfnConfig& cfg, const ModelConfig& mc,
                           gpu::Resources& res)
    : stream_(res.stream()), d_(cfg.io_dim), mpc_(mc.max_prefill_chunk) {
  // Derive the tensor-parallel-local intermediate dim from the staged views.
  if (w_gate.cols() != d_ || w_up.cols() != d_ || w_down.rows() != d_ ||
      w_gate.rows() != w_up.rows() || w_down.cols() != w_gate.rows())
    throw std::invalid_argument("gpu::DenseFfnBF16: weight shape mismatch vs config");
  inter_ = w_gate.rows();
  if (inter_ == 0)
    throw std::invalid_argument("gpu::DenseFfnBF16: bad intermediate dim");

  handle_ = res.handle();

  // Fuse [Wg;Wu] on host, one H2D (same layout the pre-migration block used).
  std::vector<bf16> gup(size_t(2) * inter_ * d_);
  std::memcpy(gup.data(), w_gate.data(), size_t(inter_) * d_ * sizeof(bf16));
  std::memcpy(gup.data() + size_t(inter_) * d_, w_up.data(), size_t(inter_) * d_ * sizeof(bf16));

  dw_gup_ = res.alloc(gup.size() * sizeof(bf16));
  dw_down_ = res.alloc(size_t(d_) * inter_ * sizeof(bf16));
  dx_ = res.alloc(size_t(mpc_) * d_ * sizeof(bf16));
  dgu_ = static_cast<float*>(res.alloc(size_t(mpc_) * 2 * inter_ * sizeof(float)));
  dact_ = res.alloc(size_t(mpc_) * inter_ * sizeof(bf16));
  // S=1 GEMV partials: the larger of the two sites (gup / down).
  const size_t gemv_floats = std::max(kernels::gemv_partial_floats(2 * inter_, d_),
                                      kernels::gemv_partial_floats(d_, inter_));
  dgemv_ = static_cast<float*>(res.alloc(gemv_floats * sizeof(float)));
  // w_down may be a strided row-parallel shard: gather to contiguous host
  // memory before the single H2D copy.
  std::vector<bf16> down_contig(size_t(d_) * inter_);
  for (uint32_t r = 0; r < d_; ++r) {
    std::memcpy(down_contig.data() + size_t(r) * inter_,
                w_down.data() + size_t(r) * w_down.stride(), inter_ * sizeof(bf16));
  }

  // Column-major repack (2026-10, research/decode-roofline [10]): the S=1
  // GEMV reads W_cm[k*N+n] as uint32 row-pairs (coalesced 256 B wave
  // accesses); the S>1 gemm_ex consumes the same layout with op=N, lda=N.
  // Temporary device staging belongs to weight loading, not named layer
  // state. The fused gate/up matrix is the larger upload. Copies and repacks
  // use the rank stream; complete the repack before releasing staging.
  void* staging_ptr = nullptr;
  HIP_CHECK(hipMalloc(&staging_ptr, gup.size() * sizeof(bf16)));
  std::unique_ptr<void, decltype(&hipFree)> staging(staging_ptr, &hipFree);
  HIP_CHECK(hipMemcpyAsync(staging.get(), gup.data(), gup.size() * sizeof(bf16),
                           hipMemcpyHostToDevice, stream_));
  kernels::repack_to_col_major(stream_, staging.get(), dw_gup_, 2 * inter_, d_);
  HIP_CHECK(hipMemcpyAsync(staging.get(), down_contig.data(),
                           down_contig.size() * sizeof(bf16),
                           hipMemcpyHostToDevice, stream_));
  kernels::repack_to_col_major(stream_, staging.get(), dw_down_, d_, inter_);
  HIP_CHECK(hipStreamSynchronize(stream_));
  // staging's RAII deleter calls hipFree on success or exception.
}

DenseFfnBF16::~DenseFfnBF16() {
  // Device VRAM is owned by gpu::Resources (freed at teardown).
}

void DenseFfnBF16::forward(float* io, uint32_t S) {
  if (S == 0 || S > mpc_)
    throw std::runtime_error("gpu::DenseFfnBF16::forward: S out of range");

  if (S == 1) {
    // Decode dispatch (design §15; the S=1 GEMV reads the column-major
    // weights — see kernels.hpp; the fp32 stream is quantized in-register
    // (gup site), so the f32_to_bf16 launch disappears from the decode
    // path. The gup combine is fused with SiLU+mul: one launch, no dgu_
    // round-trip.)
    kernels::gemv_cm_partial_x_f32(stream_, dw_gup_, io, dgemv_, 2 * inter_, d_);
    kernels::gemv_cm_combine_silu(stream_, dgemv_, dact_, inter_, d_);
    kernels::gemv_cm_x_bf16(stream_, dw_down_, dact_, io, dgemv_, d_, inter_);
    return;
  }

  kernels::f32_to_bf16(stream_, io, dx_, size_t(S) * d_);
  const float alpha = 1.0f, beta = 0.0f;
  // gate/up fused, col-major weights: C'[2I, S] = [Wg;Wu]_cm . x^T (op=N)
  RB_CHECK(rocblas_gemm_ex(handle_, rocblas_operation_none, rocblas_operation_none,
                           2 * inter_, S, d_, &alpha, dw_gup_, rocblas_datatype_bf16_r,
                           2 * inter_, dx_, rocblas_datatype_bf16_r, d_, &beta, dgu_,
                           rocblas_datatype_f32_r, 2 * inter_, dgu_,
                           rocblas_datatype_f32_r, 2 * inter_, rocblas_datatype_f32_r,
                           rocblas_gemm_algo_standard, 0, 0));
  kernels::silu_mul(stream_, dgu_, dact_, S, inter_);
  // down, straight into io, col-major weights: C'[d, S] = Wd_cm . act^T
  RB_CHECK(rocblas_gemm_ex(handle_, rocblas_operation_none, rocblas_operation_none,
                           d_, S, inter_, &alpha, dw_down_, rocblas_datatype_bf16_r,
                           d_, dact_, rocblas_datatype_bf16_r, inter_, &beta, io,
                           rocblas_datatype_f32_r, d_, io, rocblas_datatype_f32_r, d_,
                           rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
}

}  // namespace alice::gpu
