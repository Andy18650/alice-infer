#pragma once

// GPU-resident GQA attention block (design §8.1; gfx90a implementation):
// fused QKV projection (rocBLAS), SIMT RoPE+KV-append, SIMT online-softmax
// attention over the device-resident fp32 KV cache, o_proj (rocBLAS) into
// io. State: KV cache + private position counter (StatefulBlock).

#include "../block.hpp"

#include "../../gpu/resources.hpp"

#include "alice/engine_config.hpp"
#include "../../model_config.hpp"
#include "alice/tensor.hpp"

namespace alice::gpu_rocm {

class GqaAttentionBF16 final : public block::StatefulBlock {
 public:
  // prefill_scratch comes from res.scratch() (Resources-owned, the §14
  // shared partials buffer). Device VRAM is allocated through `res`
  // (bookkeeping); the block borrows the handle + carved pointers + scratch
  // and does NOT free them (Resources owns them, freed at teardown).
  GqaAttentionBF16(TensorView<bf16> wq, TensorView<bf16> wk, TensorView<bf16> wv, TensorView<bf16> wo,
                   const ModelConfig& mc, const EngineConfig& ec,
                   gpu::Resources& res);
  ~GqaAttentionBF16() override;

  void forward(float* io, uint32_t S) override;
  void reset_state() override { pos_ = 0; }

  uint32_t position() const { return pos_; }  // test introspection

 private:
  uint32_t nh_ = 0, nkv_ = 0, hd_ = 0, d_ = 0, group_ = 0;
  uint32_t max_seq_ = 0, mpc_ = 0, pos_ = 0;
  float scale_ = 0;
  void* dw_qkv_ = nullptr;    // bf16 [(qd + 2*kvd), d] fused
  void* dw_o_ = nullptr;      // bf16 [d, qd]
  float* dkcache_ = nullptr;  // fp32 [max_seq, kvd]
  float* dvcache_ = nullptr;  // fp32 [max_seq, kvd]
  float* drope_cos_ = nullptr;  // fp32 [max_seq, hd/2]
  float* drope_sin_ = nullptr;
  void* dx_ = nullptr;      // bf16 [mpc, d]
  float* dqkv_ = nullptr;   // fp32 [mpc, qd + 2*kvd]
  void* dattn_ = nullptr;   // bf16 [mpc, qd]
  float* dsplit_ = nullptr;  // split-T partials [kAttnMaxSplits*nh*kAttnSplitMaxS*(hd+2)]
  float* dgemv_ = nullptr;   // S=1 GEMV partials (max of the two sites' needs)
  float* prefill_scratch_ = nullptr;  // shared v4 prefill partials (not owned)
  rocblas_handle handle_ = nullptr;   // borrowed from the Model's Resources
};

}  // namespace alice::gpu
