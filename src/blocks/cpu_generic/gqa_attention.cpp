#include "gqa_attention.hpp"

#include <cmath>
#include <cstring>
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
      throw std::runtime_error(std::string("cpu_generic::GqaAttentionBF16: ") + #x +   \
                               " failed: " + hipGetErrorString(e_));                    \
  } while (0)
}  // namespace

GqaAttentionBF16::GqaAttentionBF16(Tensor<bf16> wq, Tensor<bf16> wk,
                                   Tensor<bf16> wv, Tensor<bf16> wo,
                                   const ModelConfig& mc,
                                   const EngineConfig& ec)
    : nh_(mc.num_heads),
      nkv_(mc.num_kv_heads),
      hd_(mc.head_dim),
      d_(mc.hidden_size),
      group_(mc.num_heads / mc.num_kv_heads),
      max_seq_(ec.max_seq_len),
      mpc_(ec.max_prefill_chunk),
      scale_(1.0f / std::sqrt(float(mc.head_dim))) {
  const uint32_t qd = nh_ * hd_, kvd = nkv_ * hd_;
  if (wq.rows() != qd || wq.cols() != d_ || wk.rows() != kvd || wk.cols() != d_ ||
      wv.rows() != kvd || wv.cols() != d_ || wo.rows() != d_ || wo.cols() != qd)
    throw std::invalid_argument("GqaAttentionBF16: weight shape mismatch vs config");
  if (nh_ % nkv_ != 0)
    throw std::invalid_argument("GqaAttentionBF16: num_heads % num_kv_heads != 0");

  wq_ = std::move(wq);
  wk_ = std::move(wk);
  wv_ = std::move(wv);
  wo_ = std::move(wo);

  kcache_ = std::make_unique<float[]>(size_t(max_seq_) * kvd);
  vcache_ = std::make_unique<float[]>(size_t(max_seq_) * kvd);

  q_ = std::make_unique<float[]>(size_t(mpc_) * qd);
  k_ = std::make_unique<float[]>(size_t(mpc_) * kvd);
  v_ = std::make_unique<float[]>(size_t(mpc_) * kvd);
  attn_ = std::make_unique<float[]>(size_t(mpc_) * qd);
  scores_ = std::make_unique<float[]>(max_seq_);

  const uint32_t half = hd_ / 2;
  rope_cos_ = std::make_unique<float[]>(size_t(max_seq_) * half);
  rope_sin_ = std::make_unique<float[]>(size_t(max_seq_) * half);
  build_rope_tables(rope_cos_.get(), rope_sin_.get(), max_seq_, hd_, mc.rope_theta,
                    mc.has_rope_scaling, mc.rope_scaling);
  host_io_.resize(size_t(mpc_) * d_);
}

void GqaAttentionBF16::forward(float* io, uint32_t S) {
  if (S == 0 || S > mpc_)
    throw std::runtime_error("cpu_generic::GqaAttentionBF16::forward: S out of range");
  if (pos_ + S > max_seq_)
    throw std::runtime_error("cpu_generic::GqaAttentionBF16::forward: max_seq_len exceeded");
  const size_t n = size_t(S) * d_;
  HIP_CHECK(hipMemcpy(host_io_.data(), io, n * sizeof(float), hipMemcpyDeviceToHost));

  const uint32_t qd = nh_ * hd_, kvd = nkv_ * hd_, half = hd_ / 2;
  float* q = q_.get();
  float* k = k_.get();
  float* v = v_.get();
  float* attn = attn_.get();

  matmul_wt(host_io_.data(), wq_.data(), q, S, d_, qd);
  matmul_wt(host_io_.data(), wk_.data(), k, S, d_, kvd);
  matmul_wt(host_io_.data(), wv_.data(), v, S, d_, kvd);

  // RoPE at absolute positions, then append to the cache.
  for (uint32_t s = 0; s < S; ++s) {
    const uint32_t pos = pos_ + s;
    const float* cos_row = rope_cos_.get() + size_t(pos) * half;
    const float* sin_row = rope_sin_.get() + size_t(pos) * half;
    for (uint32_t h = 0; h < nh_; ++h)
      rope_rotate(q + size_t(s) * qd + h * hd_, hd_, cos_row, sin_row);
    float* k_row = k + size_t(s) * kvd;
    for (uint32_t g = 0; g < nkv_; ++g) rope_rotate(k_row + g * hd_, hd_, cos_row, sin_row);
    std::memcpy(kcache_.get() + size_t(pos) * kvd, k_row, kvd * sizeof(float));
    std::memcpy(vcache_.get() + size_t(pos) * kvd, v + size_t(s) * kvd, kvd * sizeof(float));
  }

  // Causal attention over positions [0, pos_+s].
  for (uint32_t s = 0; s < S; ++s) {
    const uint32_t T = pos_ + s + 1;
    for (uint32_t h = 0; h < nh_; ++h) {
      const uint32_t g = h / group_;
      const float* q_head = q + size_t(s) * qd + h * hd_;
      for (uint32_t t = 0; t < T; ++t) {
        const float* k_head = kcache_.get() + size_t(t) * kvd + g * hd_;
        float dot = 0.0f;
        for (uint32_t i = 0; i < hd_; ++i) dot += q_head[i] * k_head[i];
        scores_[t] = dot * scale_;
      }
      softmax(scores_.get(), T);
      float* out_head = attn + size_t(s) * qd + h * hd_;
      for (uint32_t i = 0; i < hd_; ++i) out_head[i] = 0.0f;
      for (uint32_t t = 0; t < T; ++t) {
        const float* v_head = vcache_.get() + size_t(t) * kvd + g * hd_;
        const float p = scores_[t];
        for (uint32_t i = 0; i < hd_; ++i) out_head[i] += p * v_head[i];
      }
    }
  }

  matmul_wt(attn, wo_.data(), host_io_.data(), S, qd, d_);
  pos_ += S;

  HIP_CHECK(hipMemcpy(io, host_io_.data(), n * sizeof(float), hipMemcpyHostToDevice));
}

#undef HIP_CHECK

}  // namespace alice::cpu_generic
