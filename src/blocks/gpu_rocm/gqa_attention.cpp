#include "gqa_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
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

// Host-side RoPE table builder (design §3): cos_t[p*half+i] =
// cos(p*inv_freq[i]), inv_freq[i] = theta^(-2i/head_dim), with optional
// llama3 frequency scaling (low band / factor, high band unchanged, smooth
// between). The block builds these on the host at construction and H2Ds them;
// the device kernels consume the tables (they don't rebuild). The
// independent test oracle lives in tests/oracle/math.cpp.
void build_rope_tables(float* cos_t, float* sin_t, uint32_t max_seq, uint32_t head_dim,
                       float theta, const std::optional<RopeScaling>& rs) {
  const uint32_t half = head_dim / 2;
  const double ln_theta = std::log(double(theta));
  for (uint32_t i = 0; i < half; ++i) {
    double f = std::exp(-(2.0 * i / double(head_dim)) * ln_theta);
    if (rs) {
      const double wavelen = 2.0 * M_PI / f;
      const double orig = double(rs->original_max_position_embeddings);
      if (wavelen > orig / double(rs->low_freq_factor)) {
        f /= double(rs->factor);  // low-frequency band
      } else if (wavelen >= orig / double(rs->high_freq_factor)) {
        // medium band: smooth interpolation
        const double s =
            (orig / wavelen - double(rs->low_freq_factor)) /
            (double(rs->high_freq_factor) - double(rs->low_freq_factor));
        f = (1.0 - s) * f / double(rs->factor) + s * f;
      }  // else high-frequency band: unchanged
    }
    for (uint32_t p = 0; p < max_seq; ++p) {
      cos_t[size_t(p) * half + i] = float(std::cos(double(p) * f));
      sin_t[size_t(p) * half + i] = float(std::sin(double(p) * f));
    }
  }
}
}  // namespace

GqaAttentionBF16::GqaAttentionBF16(TensorView<bf16> wq, TensorView<bf16> wk, TensorView<bf16> wv,
                                   TensorView<bf16> wo, const block::GqaAttentionConfig& cfg,
                                   const ModelConfig& mc, gpu::Resources& res)
    : stream_(res.stream()),
      hd_(cfg.qk_dim),
      d_(cfg.io_dim),
      max_seq_(mc.max_seq_len),
      mpc_(mc.max_prefill_chunk),
      scale_(1.0f / std::sqrt(float(cfg.qk_dim))) {
  // Derive the tensor-parallel-local head counts from the staged weight
  // views. The factory shards heads evenly, so q/k/v head ratios are preserved.
  if (wq.cols() != d_ || wk.cols() != d_ || wv.cols() != d_ || wo.rows() != d_ ||
      wq.rows() % hd_ != 0 || wk.rows() % hd_ != 0 || wo.cols() != wq.rows())
    throw std::invalid_argument("gpu::GqaAttentionBF16: weight shape mismatch vs config");
  nh_ = wq.rows() / hd_;
  nkv_ = wk.rows() / hd_;
  if (nh_ == 0 || nkv_ == 0 || nh_ % nkv_ != 0 || hd_ > 128)
    throw std::invalid_argument("gpu::GqaAttentionBF16: bad head config");
  const uint32_t qd = nh_ * hd_, kvd = nkv_ * hd_;

  handle_ = res.handle();

  // Fuse [Wq;Wk;Wv] on host, one H2D.
  std::vector<bf16> qkv(size_t(qd + 2 * kvd) * d_);
  std::memcpy(qkv.data(), wq.data(), size_t(qd) * d_ * sizeof(bf16));
  std::memcpy(qkv.data() + size_t(qd) * d_, wk.data(), size_t(kvd) * d_ * sizeof(bf16));
  std::memcpy(qkv.data() + size_t(qd + kvd) * d_, wv.data(), size_t(kvd) * d_ * sizeof(bf16));

  dw_qkv_ = res.alloc(qkv.size() * sizeof(bf16));
  dw_o_ = res.alloc(size_t(d_) * qd * sizeof(bf16));
  // wo may be a strided row-parallel shard: gather to contiguous host memory
  // before the single H2D copy.
  std::vector<bf16> wo_contig(size_t(d_) * qd);
  for (uint32_t r = 0; r < d_; ++r) {
    std::memcpy(wo_contig.data() + size_t(r) * qd,
                wo.data() + size_t(r) * wo.stride(), qd * sizeof(bf16));
  }

  // Column-major repack (2026-10, research/decode-roofline [10]): the S=1
  // GEMV reads W_cm[k*N+n] as uint32 row-pairs (coalesced 256 B wave
  // accesses); the S>1 gemm_ex consumes the same layout with op=N, lda=N.
  // H2D row-major into the shared staging, repack kernel writes the final
  // col-major weight. Stream-ordered; staging is immediately reusable.
  const uint32_t qkvd_ = qd + 2 * kvd;
  void* staging = res.named_at_least("wpack_stage", qkv.size() * sizeof(bf16));
  HIP_CHECK(hipMemcpy(staging, qkv.data(), qkv.size() * sizeof(bf16),
                      hipMemcpyHostToDevice));
  kernels::repack_to_col_major(stream_, staging, dw_qkv_, qkvd_, d_);
  HIP_CHECK(hipMemcpy(staging, wo_contig.data(), wo_contig.size() * sizeof(bf16),
                      hipMemcpyHostToDevice));
  kernels::repack_to_col_major(stream_, staging, dw_o_, d_, qd);

  dkcache_ = static_cast<float*>(res.alloc(size_t(max_seq_) * kvd * sizeof(float)));
  dvcache_ = static_cast<float*>(res.alloc(size_t(max_seq_) * kvd * sizeof(float)));

  const uint32_t half = hd_ / 2;
  std::vector<float> cos_t(size_t(max_seq_) * half), sin_t(size_t(max_seq_) * half);
  build_rope_tables(cos_t.data(), sin_t.data(), max_seq_, hd_,
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
    // Decode dispatch (design §15; the S=1 GEMV reads the column-major
    // weights — see kernels.hpp; the fp32 stream is quantized in-register
    // (qkv site), so the f32_to_bf16 launch disappears from the decode
    // path).
    kernels::gemv_cm_x_f32(stream_, dw_qkv_, io, dqkv_, dgemv_, qkvd, d_);
    kernels::rope_append(stream_, dqkv_, dkcache_, dvcache_, drope_cos_, drope_sin_,
                         pos_, S, nh_,
                         nkv_, hd_);
    kernels::attention(stream_, dqkv_, dkcache_, dvcache_, dattn_, dsplit_,
                       prefill_scratch_,
                       pos_, S, nh_, nkv_, hd_, scale_);
    kernels::gemv_cm_x_bf16(stream_, dw_o_, dattn_, io, dgemv_, d_, qd);
    pos_ += S;
    return;
  }

  kernels::f32_to_bf16(stream_, io, dx_, size_t(S) * d_);
  const float alpha = 1.0f, beta = 0.0f;
  // fused QKV, col-major weights: C'[qkvd, S] = [Wq;Wk;Wv]_cm . x^T (op=N)
  RB_CHECK(rocblas_gemm_ex(handle_, rocblas_operation_none, rocblas_operation_none,
                           qkvd, S, d_, &alpha, dw_qkv_, rocblas_datatype_bf16_r, qkvd, dx_,
                           rocblas_datatype_bf16_r, d_, &beta, dqkv_,
                           rocblas_datatype_f32_r, qkvd, dqkv_, rocblas_datatype_f32_r,
                           qkvd, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
  kernels::rope_append(stream_, dqkv_, dkcache_, dvcache_, drope_cos_, drope_sin_, pos_,
                       S, nh_,
                       nkv_, hd_);
  kernels::attention(stream_, dqkv_, dkcache_, dvcache_, dattn_, dsplit_,
                     prefill_scratch_, pos_,
                     S, nh_, nkv_, hd_, scale_);
  // o_proj into io, col-major weights: C'[d, S] = Wo_cm . attn^T (op=N)
  RB_CHECK(rocblas_gemm_ex(handle_, rocblas_operation_none, rocblas_operation_none,
                           d_, S, qd, &alpha, dw_o_, rocblas_datatype_bf16_r, d_, dattn_,
                           rocblas_datatype_bf16_r, qd, &beta, io,
                           rocblas_datatype_f32_r, d_, io, rocblas_datatype_f32_r, d_,
                           rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
  pos_ += S;
}

}  // namespace alice::gpu
