#pragma once

// CPU reference (naive oracle) GQA attention — development-only reference
// implementation (design §3/§8.1): the test oracle and the compute core of
// gpu::CpuReferenceBlock. Never production compute.
// Weights bf16; KV cache fp32 (oracle numerics); position = private counter.

#include <memory>

#include "alice/engine_config.hpp"
#include "alice/manifest.hpp"
#include "alice/tensor.hpp"

namespace alice::cpu_generic {

class GqaAttentionBF16CpuGeneric final {
 public:
  GqaAttentionBF16CpuGeneric(BF16View wq, BF16View wk, BF16View wv, BF16View wo,
                             const ModelConfig& mc, const EngineConfig& ec);

  // io: fp32 [S, d] host, in-place (block output overwrites input).
  void forward(F32View io);
  void reset_state() { pos_ = 0; }  // allocation kept, cache invalidated
  uint32_t position() const { return pos_; }

 private:
  uint32_t nh_ = 0, nkv_ = 0, hd_ = 0, d_ = 0, group_ = 0;
  uint32_t max_seq_ = 0, mpc_ = 0;
  float scale_ = 0.0f;  // 1/sqrt(head_dim)

  std::unique_ptr<bf16[]> wq_, wk_, wv_, wo_;  // private copies
  BF16View wqv_, wkv_, wvv_, wov_;

  // State (append-only): [max_seq, nkv*hd] fp32. pos_ is both the private
  // position counter and the cache length — plain attention never evicts.
  std::unique_ptr<float[]> kcache_, vcache_;
  uint32_t pos_ = 0;

  // Transients (sized by max_prefill_chunk).
  std::unique_ptr<float[]> q_, k_, v_, attn_, scores_;
  std::unique_ptr<float[]> rope_cos_, rope_sin_;  // [max_seq, hd/2]
};

}  // namespace alice::cpu_generic
