#include "gqa_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

#include "../../host_math.hpp"
#include "../../gpu/resources.hpp"
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

GqaAttentionBF16::GqaAttentionBF16(TensorView<bf16> wq, TensorView<bf16> wk, TensorView<bf16> wv,
                                   TensorView<bf16> wo, const block::GqaAttentionConfig& cfg,
                                   const EngineConfig& ec, gpu::Resources& res)
    : nh_(cfg.num_q_heads),
      nkv_(cfg.num_kv_heads),
      hd_(cfg.qk_dim),
      d_(cfg.io_dim),
      group_(cfg.num_q_heads / cfg.num_kv_heads),
      max_seq_(ec.max_seq_len),
      mpc_(ec.max_prefill_chunk),
      scale_(1.0f / std::sqrt(float(cfg.qk_dim))) {
  const uint32_t qd = nh_ * hd_, kvd = nkv_ * hd_;
  if (wq.rows() != qd || wq.cols() != d_ || wk.rows() != kvd || wk.cols() != d_ ||
      wv.rows() != kvd || wv.cols() != d_ || wo.rows() != d_ || wo.cols() != qd)
    throw std::invalid_argument("gpu::GqaAttentionBF16: weight shape mismatch vs config");
  if (nh_ % nkv_ != 0 || hd_ > 128)
    throw std::invalid_argument("gpu::GqaAttentionBF16: bad head config");

  handle_ = res.handle();

  // Fuse [Wq;Wk;Wv] on host, one H2D.
  std::vector<bf16> qkv(size_t(qd + 2 * kvd) * d_);
  std::memcpy(qkv.data(), wq.data(), size_t(qd) * d_ * sizeof(bf16));
  std::memcpy(qkv.data() + size_t(qd) * d_, wk.data(), size_t(kvd) * d_ * sizeof(bf16));
  std::memcpy(qkv.data() + size_t(qd + kvd) * d_, wv.data(), size_t(kvd) * d_ * sizeof(bf16));

  dw_qkv_ = res.alloc(qkv.size() * sizeof(bf16));
  dw_o_ = res.alloc(size_t(d_) * qd * sizeof(bf16));
  HIP_CHECK(hipMemcpy(dw_qkv_, qkv.data(), qkv.size() * sizeof(bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dw_o_, wo.data(), size_t(d_) * qd * sizeof(bf16), hipMemcpyHostToDevice));

  dkcache_ = static_cast<float*>(res.alloc(size_t(max_seq_) * kvd * sizeof(float)));
  dvcache_ = static_cast<float*>(res.alloc(size_t(max_seq_) * kvd * sizeof(float)));

  const uint32_t half = hd_ / 2;
  std::vector<float> cos_t(size_t(max_seq_) * half), sin_t(size_t(max_seq_) * half);
  host_math::build_rope_tables(cos_t.data(), sin_t.data(), max_seq_, hd_,
                                 float(cfg.rope_theta), cfg.rope_scaling);
  drope_cos_ = static_cast<float*>(res.alloc(cos_t.size() * sizeof(float)));
  drope_sin_ = static_cast<float*>(res.alloc(sin_t.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(drope_cos_, cos_t.data(), cos_t.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(drope_sin_, sin_t.data(), sin_t.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  dx_ = res.alloc(size_t(mpc_) * d_ * sizeof(bf16));
  dqkv_ = static_cast<float*>(res.alloc(size_t(mpc_) * (qd + 2 * kvd) * sizeof(float)));
  dattn_ = res.alloc(size_t(mpc_) * qd * sizeof(bf16));
  dsplit_ = static_cast<float*>(res.alloc(size_t(kernels::kAttnMaxSplits) * nh_ *
                                          kernels::kAttnSplitMaxS * (hd_ + 2) * sizeof(float)));
  // S=1 GEMV partials: the larger of the two sites (qkv / o_proj).
  const size_t gemv_floats = std::max(
      kernels::gemv_partial_floats(qd + 2 * kvd, d_), kernels::gemv_partial_floats(d_, qd));
  dgemv_ = static_cast<float*>(res.alloc(gemv_floats * sizeof(float)));
  prefill_scratch_ = res.scratch();
  if (prefill_scratch_ == nullptr)
    throw std::invalid_argument("gpu_rocm::GqaAttentionBF16: null prefill scratch "
                                "(Resources::init_scratch not called?)");
}

GqaAttentionBF16::~GqaAttentionBF16() {
  // Device VRAM is owned by gpu::Resources (freed at teardown).
}

void GqaAttentionBF16::forward(float* io, uint32_t S) {
  if (S == 0 || S > mpc_)
    throw std::runtime_error("gpu::GqaAttentionBF16::forward: S out of range");
  if (pos_ + S > max_seq_)
    throw std::runtime_error("gpu::GqaAttentionBF16::forward: max_seq_len exceeded");
  const uint32_t qd = nh_ * hd_, kvd = nkv_ * hd_, qkvd = qd + 2 * kvd;

  if (S == 1) {
    // Decode dispatch (design §15; measured in research/decode-roofline:
    // gemm_ex S=1 runs the FFN shapes at 313-727 GB/s; the split-K GEMV at
    // 946-1265). The fp32 stream is quantized in-register (qkv site), so
    // the f32_to_bf16 launch disappears from the decode path.
    kernels::gemv_x_f32(dw_qkv_, io, dqkv_, dgemv_, qkvd, d_);
    kernels::rope_append(dqkv_, dkcache_, dvcache_, drope_cos_, drope_sin_, pos_, S, nh_,
                         nkv_, hd_);
    kernels::attention(dqkv_, dkcache_, dvcache_, dattn_, dsplit_, prefill_scratch_,
                       pos_, S, nh_, nkv_, hd_, scale_);
    kernels::gemv_x_bf16(dw_o_, dattn_, io, dgemv_, d_, qd);
    pos_ += S;
    return;
  }

  kernels::f32_to_bf16(io, dx_, size_t(S) * d_);
  const float alpha = 1.0f, beta = 0.0f;
  // fused QKV: C'[qkvd, S] = [Wq;Wk;Wv] . x^T
  RB_CHECK(rocblas_gemm_ex(handle_, rocblas_operation_transpose, rocblas_operation_none,
                           qkvd, S, d_, &alpha, dw_qkv_, rocblas_datatype_bf16_r, d_, dx_,
                           rocblas_datatype_bf16_r, d_, &beta, dqkv_,
                           rocblas_datatype_f32_r, qkvd, dqkv_, rocblas_datatype_f32_r,
                           qkvd, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
  kernels::rope_append(dqkv_, dkcache_, dvcache_, drope_cos_, drope_sin_, pos_, S, nh_,
                       nkv_, hd_);
  kernels::attention(dqkv_, dkcache_, dvcache_, dattn_, dsplit_, prefill_scratch_, pos_,
                     S, nh_, nkv_, hd_, scale_);
  // o_proj into io: C'[d, S] = Wo . attn^T
  RB_CHECK(rocblas_gemm_ex(handle_, rocblas_operation_transpose, rocblas_operation_none,
                           d_, S, qd, &alpha, dw_o_, rocblas_datatype_bf16_r, qd, dattn_,
                           rocblas_datatype_bf16_r, qd, &beta, io,
                           rocblas_datatype_f32_r, d_, io, rocblas_datatype_f32_r, d_,
                           rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
  pos_ += S;
}

}  // namespace alice::gpu
