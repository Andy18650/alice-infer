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
  // Takes ownership of the staged weights (moved in, shape-checked).
  GqaAttentionBF16CpuGeneric(Tensor<bf16> wq, Tensor<bf16> wk, Tensor<bf16> wv,
                             Tensor<bf16> wo, const ModelConfig& mc,
                             const EngineConfig& ec);

  // io: fp32 [S, d] host, in-place (block output overwrites input).
  void forward(float* io, uint32_t S);
  void reset_state() { pos_ = 0; }  // allocation kept, cache invalidated
  uint32_t position() const { return pos_; }

 private:
  uint32_t nh_ = 0, nkv_ = 0, hd_ = 0, d_ = 0, group_ = 0;
  uint32_t max_seq_ = 0, mpc_ = 0;
  float scale_ = 0.0f;  // 1/sqrt(head_dim)

  Tensor<bf16> wq_, wk_, wv_, wo_;  // private weights (moved in)

  // State (append-only): [max_seq, nkv*hd] fp32. pos_ is both the private
  // position counter and the cache length — plain attention never evicts.
  std::unique_ptr<float[]> kcache_, vcache_;
  uint32_t pos_ = 0;

  // Transients (sized by max_prefill_chunk).
  std::unique_ptr<float[]> q_, k_, v_, attn_, scores_;
  std::unique_ptr<float[]> rope_cos_, rope_sin_;  // [max_seq, hd/2]
};

}  // namespace alice::cpu_generic
