#pragma once

// Factory for dense_ffn_bf16 v1 (the gpu_rocm DenseFfnBF16 block). Layout:
//   [w_gate (inter x io)] [w_up (inter x io)] [w_down (io x inter)]
// bf16 row-major, no padding. Lives next to the block implementation it
// constructs; the layout order MUST match tools/pack_model.py.

#include "block_factory.hpp"
#include "dense_ffn_config.hpp"  // DenseFfnConfig (parsed spec state + ctor vocab)

namespace alice::block {

class DenseFfnBF16Factory final : public BlockFactory {
 public:
  // Parses + validates the spec into a populated factory. Takes primitives
  // (version, config), not a BlockSpecDecl, so it never sees the type name.
  // Returns the base type so &from_spec is directly a ResolveFn (no trampoline).
  static std::unique_ptr<BlockFactory> from_spec(uint32_t version,
                                                 const nlohmann::json& config);

  uint64_t derived_bytes() const override;
  uint64_t device_bytes(const EngineConfig& ec) const override;
  size_t scratch_bytes(const EngineConfig& ec) const override;
  std::string interface_sig() const override;
  std::unique_ptr<Block> construct(std::span<const std::byte> data,
                                   gpu::Resources& res,
                                   const EngineConfig& ec) const override;

 private:
  DenseFfnConfig cfg_;
};

}  // namespace alice::block
