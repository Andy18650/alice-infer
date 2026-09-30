#include "dense_ffn_factory.hpp"

#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "../manifest.hpp"   // ModelConfig (block-constructor vocab)
#include "alice/tensor.hpp"   // bf16, TensorView
#include "../model_format.hpp"  // BlockSpecDecl, DeviceAllocator
#include "gpu_rocm/dense_ffn.hpp"  // gpu_rocm::DenseFfnBF16

namespace alice::block {

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("block factory: " + what);
}

// dense_ffn_bf16 v1 config (missing keys throw via the macro's from_json).
struct DenseFfnV1Config {
  uint32_t io_dim = 0;
  uint32_t intermediate_dim = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DenseFfnV1Config, io_dim, intermediate_dim)

}  // namespace

uint64_t DenseFfnBF16Factory::derived_bytes() const {
  return 3 * uint64_t(io) * inter * sizeof(bf16);
}

size_t DenseFfnBF16Factory::scratch_bytes(const EngineConfig&) const { return 0; }

std::string DenseFfnBF16Factory::interface_sig() const {
  return "io:" + std::to_string(io);
}

std::unique_ptr<Block> DenseFfnBF16Factory::construct(
    Chunk chunk, modelfmt::DeviceAllocator& alloc, const gpu_rocm::Context& ctx,
    const EngineConfig& ec) const {
  (void)alloc;
  if (chunk.size() != derived_bytes())
    fail("dense_ffn_bf16: decoded chunk is " + std::to_string(chunk.size()) +
         " bytes but the v1 layout derives " + std::to_string(derived_bytes()));
  const bf16* base = reinterpret_cast<const bf16*>(chunk.data());
  size_t off = 0;
  auto take = [&](uint32_t rows, uint32_t cols) {
    TensorView<bf16> v(base + off, rows, cols);
    off += size_t(rows) * cols;
    return v;
  };
  const auto gate = take(inter, io);
  const auto up = take(inter, io);
  const auto down = take(io, inter);
  ModelConfig mc;
  mc.hidden_size = io;
  mc.intermediate_size = inter;
  return std::make_unique<gpu_rocm::DenseFfnBF16>(gate, up, down, mc, ec, ctx);
}

std::unique_ptr<BlockFactory> resolve_dense_ffn(const modelfmt::BlockSpecDecl& d) {
  if (d.version != 1)
    fail("dense_ffn_bf16: unsupported version " + std::to_string(d.version) +
         " (registry provides version 1)");
  const DenseFfnV1Config cfg = d.config.get<DenseFfnV1Config>();
  auto r = std::make_unique<DenseFfnBF16Factory>();
  r->io = cfg.io_dim;
  r->inter = cfg.intermediate_dim;
  if (r->io == 0 || r->inter == 0)
    fail("dense_ffn_bf16: io_dim/intermediate_dim must be non-zero");
  return r;
}

namespace {
// Self-registration (design §15): runs at static init. The TU is retained
// by consumers linking the factory objects with --whole-archive.
const bool dense_ffn_registered = [] {
  register_factory("dense_ffn_bf16", &resolve_dense_ffn);
  return true;
}();
}  // namespace

}  // namespace alice::block
