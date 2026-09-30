#pragma once

// Factory for dense_ffn_bf16 v1 (the gpu_rocm DenseFfnBF16 block). Layout:
//   [w_gate (inter x io)] [w_up (inter x io)] [w_down (io x inter)]
// bf16 row-major, no padding. Lives next to the block implementation it
// constructs; the layout order MUST match tools/pack_model.py.

#include "block_factory.hpp"

namespace alice::modelfmt {
struct BlockSpecDecl;
}  // namespace alice::modelfmt

namespace alice::block {

class DenseFfnBF16Factory final : public BlockFactory {
 public:
  uint64_t derived_bytes() const override;
  size_t scratch_bytes(const EngineConfig& ec) const override;
  std::string interface_sig() const override;
  std::unique_ptr<Block> construct(Chunk chunk, modelfmt::DeviceAllocator& alloc,
                                   const gpu_rocm::Context& ctx,
                                   const EngineConfig& ec) const override;

  uint32_t io = 0;     // hidden_size
  uint32_t inter = 0;  // intermediate_size
};

std::unique_ptr<BlockFactory> resolve_dense_ffn(const modelfmt::BlockSpecDecl& d);

}  // namespace alice::block
