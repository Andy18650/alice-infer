#include "gqa_attention.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include "../cpu_generic/kernels.hpp"
#include "platform.hpp"
#include "kernels.hpp"

namespace alice::gpu_rocm {

namespace {
#define HIP_CHECK(x)                                                                    \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess)                                                               \
      throw std::runtime_error(std::string("gpu::GqaAttentionBF16: ") + #x +           \
                               " failed: " + hipGetErrorString(e_));                    \
  } while (0)
#define RB_CHECK(x)                                                                     \
  do {                                                                                  \
    rocblas_status s_ = (x);                                                            \
    if (s_ != rocblas_status_success)                                                   \
      throw std::runtime_error(std::string("gpu::GqaAttentionBF16: rocBLAS error ") +   \
                               std::to_string(int(s_)));                                \
  } while (0)
}  // namespace

GqaAttentionBF16::GqaAttentionBF16(BF16View wq, BF16View wk, BF16View wv, BF16View wo,
                                   const ModelConfig& mc, const EngineConfig& ec,
                                   float* prefill_scratch)
    : nh_(mc.num_heads),
      nkv_(mc.num_kv_heads),
      hd_(mc.head_dim),
      d_(mc.hidden_size),
      group_(mc.num_heads / mc.num_kv_heads),
      max_seq_(ec.max_seq_len),
      mpc_(ec.max_prefill_chunk),
      scale_(1.0f / std::sqrt(float(mc.head_dim))) {
  const uint32_t qd = nh_ * hd_, kvd = nkv_ * hd_;
  if (wq.rows != qd || wq.cols != d_ || wk.rows != kvd || wk.cols != d_ ||
      wv.rows != kvd || wv.cols != d_ || wo.rows != d_ || wo.cols != qd)
    throw std::invalid_argument("gpu::GqaAttentionBF16: weight shape mismatch vs config");
  if (nh_ % nkv_ != 0 || hd_ > 128)
    throw std::invalid_argument("gpu::GqaAttentionBF16: bad head config");

  Platform& p = platform();
  HIP_CHECK(hipSetDevice(p.device));

  // Fuse [Wq;Wk;Wv] on host, one H2D.
  std::vector<bf16> qkv(size_t(qd + 2 * kvd) * d_);
  std::memcpy(qkv.data(), wq.data, size_t(qd) * d_ * sizeof(bf16));
  std::memcpy(qkv.data() + size_t(qd) * d_, wk.data, size_t(kvd) * d_ * sizeof(bf16));
  std::memcpy(qkv.data() + size_t(qd + kvd) * d_, wv.data, size_t(kvd) * d_ * sizeof(bf16));

  HIP_CHECK(hipMalloc(&dw_qkv_, qkv.size() * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&dw_o_, size_t(d_) * qd * sizeof(bf16)));
  HIP_CHECK(hipMemcpy(dw_qkv_, qkv.data(), qkv.size() * sizeof(bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dw_o_, wo.data, size_t(d_) * qd * sizeof(bf16), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&dkcache_, size_t(max_seq_) * kvd * sizeof(float)));
  HIP_CHECK(hipMalloc(&dvcache_, size_t(max_seq_) * kvd * sizeof(float)));

  const uint32_t half = hd_ / 2;
  std::vector<float> cos_t(size_t(max_seq_) * half), sin_t(size_t(max_seq_) * half);
  cpu_generic::build_rope_tables(cos_t.data(), sin_t.data(), max_seq_, hd_, mc.rope_theta,
                                 mc.has_rope_scaling, mc.rope_scaling);
  HIP_CHECK(hipMalloc(&drope_cos_, cos_t.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&drope_sin_, sin_t.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(drope_cos_, cos_t.data(), cos_t.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(drope_sin_, sin_t.data(), sin_t.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&dx_, size_t(mpc_) * d_ * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&dqkv_, size_t(mpc_) * (qd + 2 * kvd) * sizeof(float)));
  HIP_CHECK(hipMalloc(&dattn_, size_t(mpc_) * qd * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&dsplit_, size_t(kernels::kAttnMaxSplits) * nh_ *
                                    kernels::kAttnSplitMaxS * (hd_ + 2) * sizeof(float)));
  prefill_scratch_ = prefill_scratch;
  if (prefill_scratch_ == nullptr)
    throw std::invalid_argument("gpu_rocm::GqaAttentionBF16: null prefill scratch");
}

GqaAttentionBF16::~GqaAttentionBF16() {
  (void)hipFree(dw_qkv_);
  (void)hipFree(dw_o_);
  (void)hipFree(dkcache_);
  (void)hipFree(dvcache_);
  (void)hipFree(drope_cos_);
  (void)hipFree(drope_sin_);
  (void)hipFree(dx_);
  (void)hipFree(dqkv_);
  (void)hipFree(dattn_);
  (void)hipFree(dsplit_);
}

void GqaAttentionBF16::forward(float* io, uint32_t S) {
  if (S == 0 || S > mpc_)
    throw std::runtime_error("gpu::GqaAttentionBF16::forward: S out of range");
  if (pos_ + S > max_seq_)
    throw std::runtime_error("gpu::GqaAttentionBF16::forward: max_seq_len exceeded");
  Platform& p = platform();
  HIP_CHECK(hipSetDevice(p.device));
  const uint32_t qd = nh_ * hd_, kvd = nkv_ * hd_, qkvd = qd + 2 * kvd;

  kernels::f32_to_bf16(io, dx_, size_t(S) * d_);
  const float alpha = 1.0f, beta = 0.0f;
  // fused QKV: C'[qkvd, S] = [Wq;Wk;Wv] . x^T
  RB_CHECK(rocblas_gemm_ex(p.handle, rocblas_operation_transpose, rocblas_operation_none,
                           qkvd, S, d_, &alpha, dw_qkv_, rocblas_datatype_bf16_r, d_, dx_,
                           rocblas_datatype_bf16_r, d_, &beta, dqkv_,
                           rocblas_datatype_f32_r, qkvd, dqkv_, rocblas_datatype_f32_r,
                           qkvd, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
  kernels::rope_append(dqkv_, dkcache_, dvcache_, drope_cos_, drope_sin_, pos_, S, nh_,
                       nkv_, hd_);
  kernels::attention(dqkv_, dkcache_, dvcache_, dattn_, dsplit_, prefill_scratch_, pos_,
                     S, nh_, nkv_, hd_, scale_);
  // o_proj into io: C'[d, S] = Wo . attn^T
  RB_CHECK(rocblas_gemm_ex(p.handle, rocblas_operation_transpose, rocblas_operation_none,
                           d_, S, qd, &alpha, dw_o_, rocblas_datatype_bf16_r, qd, dattn_,
                           rocblas_datatype_bf16_r, qd, &beta, io,
                           rocblas_datatype_f32_r, d_, io, rocblas_datatype_f32_r, d_,
                           rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
  pos_ += S;
}

}  // namespace alice::gpu
