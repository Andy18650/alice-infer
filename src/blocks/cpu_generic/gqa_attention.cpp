#include "gqa_attention.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "kernels.hpp"

namespace alice::cpu_generic {

namespace {
template <typename T>
std::unique_ptr<T[]> copy_weight(TView<T> src, TView<T>& view, const char* what) {
  if (src.stride() != src.cols)
    throw std::invalid_argument(std::string(what) + ": strided weights unsupported");
  auto buf = std::make_unique<T[]>(src.numel());
  std::memcpy(buf.get(), src.data, src.nbytes());
  view = TView<T>{buf.get(), src.rows, src.cols, 0};
  return buf;
}
}  // namespace

GqaAttentionBF16CpuGeneric::GqaAttentionBF16CpuGeneric(
    BF16View wq, BF16View wk, BF16View wv, BF16View wo, const ModelConfig& mc,
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
  if (wq.rows != qd || wq.cols != d_ || wk.rows != kvd || wk.cols != d_ ||
      wv.rows != kvd || wv.cols != d_ || wo.rows != d_ || wo.cols != qd)
    throw std::invalid_argument("GqaAttentionBF16CpuGeneric: weight shape mismatch vs config");
  if (nh_ % nkv_ != 0)
    throw std::invalid_argument("GqaAttentionBF16CpuGeneric: num_heads % num_kv_heads != 0");

  wq_ = copy_weight(wq, wqv_, "GqaAttentionBF16CpuGeneric");
  wk_ = copy_weight(wk, wkv_, "GqaAttentionBF16CpuGeneric");
  wv_ = copy_weight(wv, wvv_, "GqaAttentionBF16CpuGeneric");
  wo_ = copy_weight(wo, wov_, "GqaAttentionBF16CpuGeneric");

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
}

void GqaAttentionBF16CpuGeneric::forward(F32View io) {
  const uint32_t S = io.rows;
  if (S > mpc_)
    throw std::runtime_error("GqaAttentionBF16CpuGeneric: prefill chunk exceeded");
  if (io.cols != d_)
    throw std::invalid_argument("GqaAttentionBF16CpuGeneric: io must be [S, d]");
  if (pos_ + S > max_seq_)
    throw std::runtime_error("GqaAttentionBF16CpuGeneric: max_seq_len exceeded");

  const uint32_t qd = nh_ * hd_, kvd = nkv_ * hd_, half = hd_ / 2;
  F32View q{q_.get(), S, qd, 0};
  F32View k{k_.get(), S, kvd, 0};
  F32View v{v_.get(), S, kvd, 0};
  F32View attn{attn_.get(), S, qd, 0};

  matmul_wt(io, wqv_, q);
  matmul_wt(io, wkv_, k);
  matmul_wt(io, wvv_, v);

  // RoPE at absolute positions, then append to the cache.
  for (uint32_t s = 0; s < S; ++s) {
    const uint32_t pos = pos_ + s;
    const float* cos_row = rope_cos_.get() + size_t(pos) * half;
    const float* sin_row = rope_sin_.get() + size_t(pos) * half;
    for (uint32_t h = 0; h < nh_; ++h)
      rope_rotate(q.data + size_t(s) * qd + h * hd_, hd_, cos_row, sin_row);
    float* k_row = k.data + size_t(s) * kvd;
    for (uint32_t g = 0; g < nkv_; ++g) rope_rotate(k_row + g * hd_, hd_, cos_row, sin_row);
    std::memcpy(kcache_.get() + size_t(pos) * kvd, k_row, kvd * sizeof(float));
    std::memcpy(vcache_.get() + size_t(pos) * kvd, v.data + size_t(s) * kvd,
                kvd * sizeof(float));
  }

  // Causal attention over positions [0, pos_+s].
  for (uint32_t s = 0; s < S; ++s) {
    const uint32_t T = pos_ + s + 1;
    for (uint32_t h = 0; h < nh_; ++h) {
      const uint32_t g = h / group_;
      const float* q_head = q.data + size_t(s) * qd + h * hd_;
      for (uint32_t t = 0; t < T; ++t) {
        const float* k_head = kcache_.get() + size_t(t) * kvd + g * hd_;
        float dot = 0.0f;
        for (uint32_t i = 0; i < hd_; ++i) dot += q_head[i] * k_head[i];
        scores_[t] = dot * scale_;
      }
      softmax(scores_.get(), T);
      float* out_head = attn.data + size_t(s) * qd + h * hd_;
      for (uint32_t i = 0; i < hd_; ++i) out_head[i] = 0.0f;
      for (uint32_t t = 0; t < T; ++t) {
        const float* v_head = vcache_.get() + size_t(t) * kvd + g * hd_;
        const float p = scores_[t];
        for (uint32_t i = 0; i < hd_; ++i) out_head[i] += p * v_head[i];
      }
    }
  }

  matmul_wt(attn, wov_, io);  // reads attn_, writes io: no aliasing
  pos_ += S;
}

}  // namespace alice::cpu_generic
