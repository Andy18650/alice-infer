#pragma once

// Factory for gqa_attention_bf16 v1 (the gpu_rocm GqaAttentionBF16 block).
// Layout:
//   [wq (nh*hd x io)] [wk (nkv*hd x io)] [wv (nkv*hd x io)] [wo (io x nh*hd)]
// bf16 row-major, no padding. v1 requires qk_dim == v_dim (one head dim).
// Lives next to the block implementation it constructs; layout order MUST
// match tools/pack_model.py.

#include "block_factory.hpp"
#include "../manifest.hpp"  // ModelConfig (block-constructor vocab)

namespace alice::block {

class GqaAttentionBF16Factory final : public BlockFactory {
 public:
  // Parses + validates the spec into a populated factory. Takes primitives
  // (version, config), not a BlockSpecDecl, so it never sees the type name.
  // Returns the base type so &from_spec is directly a ResolveFn (no trampoline).
  static std::unique_ptr<BlockFactory> from_spec(uint32_t version,
                                                 const nlohmann::json& config);

  uint64_t derived_bytes() const override;
  size_t scratch_bytes(const EngineConfig& ec) const override;
  std::string interface_sig() const override;
  std::unique_ptr<Block> construct(Chunk chunk, modelfmt::DeviceAllocator& alloc,
                                   const gpu_rocm::Context& ctx,
                                   const EngineConfig& ec) const override;

 private:
  uint32_t io = 0;    // hidden_size
  uint32_t nh = 0;    // num query heads
  uint32_t nkv = 0;   // num kv heads
  uint32_t hd = 0;    // head dim
  ModelConfig mc;     // built during from_spec, consumed by construct
};

}  // namespace alice::block
