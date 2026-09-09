#pragma once

// CPU_GENERIC (naive oracle) implementation of GqaAttentionBF16.
// Weights bf16; KV cache fp32 (oracle numerics); position = private counter.

#include "alice/gqa_attention_bf16.hpp"

namespace alice::cpu_generic {

class GqaAttentionBF16CpuGeneric final : public GqaAttentionBF16 {
 public:
  GqaAttentionBF16CpuGeneric(TensorView wq, TensorView wk, TensorView wv, TensorView wo,
                             const ModelConfig& mc, const EngineConfig& ec);

  void forward(TensorView io) override;
  void reset_state() override { pos_ = 0; }  // allocation kept, cache invalidated

 private:
  uint32_t nh_ = 0, nkv_ = 0, hd_ = 0, d_ = 0, group_ = 0;
  uint32_t max_seq_ = 0, mpc_ = 0;
  float scale_ = 0.0f;  // 1/sqrt(head_dim)

  std::unique_ptr<std::byte[]> wq_, wk_, wv_, wo_;  // private bf16 copies
  TensorView wqv_, wkv_, wvv_, wov_;

  // State (append-only): [max_seq, nkv*hd] fp32. pos_ is both the private
  // position counter and the cache length — plain attention never evicts
  // (a KVPress-ing implementation would separate the two, design §6).
  std::unique_ptr<float[]> kcache_, vcache_;
  uint32_t pos_ = 0;

  // Transients (sized by max_prefill_chunk).
  std::unique_ptr<float[]> q_, k_, v_, attn_, scores_;
  std::unique_ptr<float[]> rope_cos_, rope_sin_;  // [max_seq, hd/2]
};

}  // namespace alice::cpu_generic
