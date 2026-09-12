#pragma once

// Development-only CPU reference block (design §3/§8.1): wraps a naive
// cpu_generic block so it can be swapped INTO the GPU-resident model for
// fault localization (e.g. bring-up of research-grade kernels). The D2H →
// host compute → H2D round trip lives entirely inside this adapter, so the
// gpu::Block interface stays uniform. Never production compute.

#include <memory>
#include <variant>
#include <vector>

#include "../blocks/cpu_generic/dense_ffn.hpp"
#include "../blocks/cpu_generic/gqa_attention.hpp"
#include "block.hpp"

namespace alice::gpu {

class CpuReferenceBlock final : public StatefulBlock {
 public:
  // Takes ownership of a constructed cpu_generic block.
  explicit CpuReferenceBlock(cpu_generic::DenseFfnBF16CpuGeneric b, uint32_t d,
                             uint32_t max_prefill_chunk);
  explicit CpuReferenceBlock(cpu_generic::GqaAttentionBF16CpuGeneric b, uint32_t d,
                             uint32_t max_prefill_chunk);

  void forward(float* io, uint32_t S) override;  // D2H -> host compute -> H2D
  void reset_state() override;

 private:
  std::variant<cpu_generic::DenseFfnBF16CpuGeneric, cpu_generic::GqaAttentionBF16CpuGeneric>
      impl_;
  uint32_t d_ = 0, mpc_ = 0;
  std::vector<float> host_io_;  // [mpc, d] staging
};

}  // namespace alice::gpu
