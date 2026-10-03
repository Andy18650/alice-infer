#include "dense_ffn_factory.hpp"

#include <stdexcept>
#include <string>
#include <algorithm>

#include <nlohmann/json.hpp>

#include "alice/tensor.hpp"   // bf16, TensorView
#include "gpu_rocm/dense_ffn.hpp"  // gpu_rocm::DenseFfnBF16
#include "gpu_rocm/kernels.hpp"    // gemv_partial_floats

namespace alice::block {

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("block factory: " + what);
}

}  // namespace

// JSON binding for the block-type config (design §8): plain POD, the macro
// is the only nlohmann touch and stays in the factory .cpp.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DenseFfnConfig, io_dim, intermediate_dim)

uint64_t DenseFfnBF16Factory::derived_bytes() const {
  return 3 * uint64_t(cfg_.io_dim) * cfg_.intermediate_dim * sizeof(bf16);
}

uint64_t DenseFfnBF16Factory::device_bytes(const EngineConfig& ec) const {
  // Mirrors the constructor's allocations (dense_ffn.cpp), but sized for this
  // rank's tensor-parallel shard. Each padded to gpu::kDeviceAlign so the
  // arena total is exactly predictable.
  const auto pad = [](size_t n) {
    return (n + gpu::kDeviceAlign - 1) & ~(gpu::kDeviceAlign - 1);
  };
  const uint32_t tp = ec.tp_degree ? ec.tp_degree : 1;
  const uint32_t io = cfg_.io_dim, inter = cfg_.intermediate_dim / tp;
  const uint32_t mpc = ec.max_prefill_chunk;
  uint64_t total = 0;
  total += pad(size_t(2) * inter * io * sizeof(bf16));          // dw_gup_ [2I/tp,d] fused
  total += pad(size_t(io) * inter * sizeof(bf16));               // dw_down_ [d,I/tp]
  total += pad(size_t(mpc) * io * sizeof(bf16));                 // dx_ [mpc,d]
  total += pad(size_t(mpc) * 2 * inter * sizeof(float));         // dgu_ [mpc,2I/tp]
  total += pad(size_t(mpc) * inter * sizeof(bf16));              // dact_ [mpc,I/tp]
  const size_t gemv = std::max(gpu_rocm::kernels::gemv_partial_floats(2 * inter, io),
                               gpu_rocm::kernels::gemv_partial_floats(io, inter));
  total += pad(gemv * sizeof(float));                            // dgemv_
  return total;
}

size_t DenseFfnBF16Factory::scratch_bytes(const EngineConfig&) const { return 0; }

std::string DenseFfnBF16Factory::interface_sig() const {
  return "io:" + std::to_string(cfg_.io_dim);
}

std::unique_ptr<Block> DenseFfnBF16Factory::construct(
    std::span<const std::byte> data, gpu::Resources& res, const EngineConfig& ec) const {
  if (data.size() != derived_bytes())
    fail("dense_ffn_bf16: decoded chunk is " + std::to_string(data.size()) +
         " bytes but the v1 layout derives " + std::to_string(derived_bytes()));
  const uint32_t tp = ec.tp_degree ? ec.tp_degree : 1;
  const uint32_t rank = static_cast<uint32_t>(res.rank());
  const uint32_t io = cfg_.io_dim, inter = cfg_.intermediate_dim;
  if (inter % tp != 0)
    fail("dense_ffn_bf16: tp_degree does not divide intermediate_dim");
  const uint32_t inter_local = inter / tp;

  const bf16* base = reinterpret_cast<const bf16*>(data.data());
  // Column-parallel gate/up: contiguous slices along the output dim.
  const bf16* base_gate = base;
  const bf16* base_up = base + size_t(inter) * io;
  const bf16* base_down = base + size_t(2 * inter) * io;
  TensorView<bf16> gate(base_gate + rank * size_t(inter_local) * io, inter_local, io);
  TensorView<bf16> up(base_up + rank * size_t(inter_local) * io, inter_local, io);
  // Row-parallel down: strided slice along the input dim.
  TensorView<bf16> down(base_down + rank * inter_local, io, inter_local, inter);
  return std::make_unique<gpu_rocm::DenseFfnBF16>(gate, up, down, cfg_, ec, res);
}

std::unique_ptr<BlockFactory> DenseFfnBF16Factory::from_spec(uint32_t version,
                                                              const nlohmann::json& config) {
  if (version != 1)
    fail("dense_ffn_bf16: unsupported version " + std::to_string(version) +
         " (registry provides version 1)");
  const DenseFfnConfig cfg = config.get<DenseFfnConfig>();
  if (cfg.io_dim == 0 || cfg.intermediate_dim == 0)
    fail("dense_ffn_bf16: io_dim/intermediate_dim must be non-zero");
  auto r = std::make_unique<DenseFfnBF16Factory>();
  r->cfg_ = cfg;
  return r;
}

namespace {
// Self-registration (design §15): runs at static init. The TU is retained
// by consumers linking the factory objects with --whole-archive.
const bool dense_ffn_registered = [] {
  register_factory("dense_ffn_bf16", &DenseFfnBF16Factory::from_spec);
  return true;
}();
}  // namespace

}  // namespace alice::block
