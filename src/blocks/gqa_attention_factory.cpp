#include "gqa_attention_factory.hpp"

#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "../manifest.hpp"   // ModelConfig
#include "alice/tensor.hpp"   // bf16, TensorView
#include "../model_format.hpp"  // DeviceAllocator
#include "gpu_rocm/gqa_attention.hpp"  // gpu_rocm::GqaAttentionBF16

namespace alice::block {

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("block factory: " + what);
}

// gqa_attention_bf16 v1 config: core fields via struct (missing keys
// throw); the optional llama3 rope-scaling group is read manually because
// it is conditional (present-type => four more required fields).
struct GqaAttentionV1Config {
  uint32_t io_dim = 0;
  uint32_t qk_dim = 0;
  uint32_t v_dim = 0;
  uint32_t num_q_heads = 0;
  uint32_t num_kv_heads = 0;
  double rope_theta = 0.0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GqaAttentionV1Config, io_dim, qk_dim, v_dim,
                                   num_q_heads, num_kv_heads, rope_theta)

}  // namespace

uint64_t GqaAttentionBF16Factory::derived_bytes() const {
  return uint64_t(2 * nh + 2 * nkv) * hd * io * sizeof(bf16);
}

size_t GqaAttentionBF16Factory::scratch_bytes(const EngineConfig& ec) const {
  // v4 prefill partials: nh * max_prefill_chunk * ceil(max_seq/64) * (hd+2)
  const uint64_t max_chunks = (uint64_t(ec.max_seq_len) + 63) / 64;
  return uint64_t(nh) * ec.max_prefill_chunk * max_chunks * (hd + 2) * sizeof(float);
}

std::string GqaAttentionBF16Factory::interface_sig() const {
  return "io:" + std::to_string(io);
}

std::unique_ptr<Block> GqaAttentionBF16Factory::construct(
    Chunk chunk, modelfmt::DeviceAllocator& alloc, const gpu_rocm::Context& ctx,
    const EngineConfig& ec) const {
  if (chunk.size() != derived_bytes())
    fail("gqa_attention_bf16: decoded chunk is " + std::to_string(chunk.size()) +
         " bytes but the v1 layout derives " + std::to_string(derived_bytes()));
  const bf16* base = reinterpret_cast<const bf16*>(chunk.data());
  size_t off = 0;
  auto take = [&](uint32_t rows, uint32_t cols) {
    TensorView<bf16> v(base + off, rows, cols);
    off += size_t(rows) * cols;
    return v;
  };
  const auto wq = take(nh * hd, io);
  const auto wk = take(nkv * hd, io);
  const auto wv = take(nkv * hd, io);
  const auto wo = take(io, nh * hd);
  return std::make_unique<gpu_rocm::GqaAttentionBF16>(wq, wk, wv, wo, mc, ec, ctx,
                                                      alloc.scratch_ptr());
}

std::unique_ptr<BlockFactory> GqaAttentionBF16Factory::from_spec(uint32_t version,
                                                                 const nlohmann::json& config) {
  if (version != 1)
    fail("gqa_attention_bf16: unsupported version " + std::to_string(version) +
         " (registry provides version 1)");
  const GqaAttentionV1Config cfg = config.get<GqaAttentionV1Config>();
  auto r = std::make_unique<GqaAttentionBF16Factory>();
  r->io = cfg.io_dim;
  r->nh = cfg.num_q_heads;
  r->nkv = cfg.num_kv_heads;
  if (cfg.qk_dim != cfg.v_dim)
    fail("gqa_attention_bf16: qk_dim != v_dim (v1 requires equal head dims; "
         "separate value head dim is a future version)");
  r->hd = cfg.qk_dim;
  if (r->io == 0 || r->nh == 0 || r->nkv == 0 || r->hd == 0 || r->nh % r->nkv != 0)
    fail("gqa_attention_bf16: bad head config (io/heads/kv-heads/head-dim, "
         "num_q_heads must divide num_kv_heads evenly)");

  ModelConfig& mc = r->mc;
  mc.hidden_size = r->io;
  mc.num_heads = r->nh;
  mc.num_kv_heads = r->nkv;
  mc.head_dim = r->hd;
  mc.rope_theta = float(cfg.rope_theta);
  if (config.contains("rope_scaling_type")) {
    const std::string type = config.at("rope_scaling_type").get<std::string>();
    if (type != "llama3")
      fail("gqa_attention_bf16: unsupported rope_scaling_type '" + type + "'");
    mc.has_rope_scaling = true;
    mc.rope_scaling.factor = config.at("rope_scaling_factor").get<float>();
    mc.rope_scaling.low_freq_factor =
        config.at("rope_scaling_low_freq_factor").get<float>();
    mc.rope_scaling.high_freq_factor =
        config.at("rope_scaling_high_freq_factor").get<float>();
    mc.rope_scaling.original_max_position_embeddings =
        config.at("original_max_position_embeddings").get<uint32_t>();
  }
  return r;
}

namespace {
// Self-registration (design §15): runs at static init. The TU is retained
// by consumers linking the factory objects with --whole-archive.
const bool gqa_attention_registered = [] {
  register_factory("gqa_attention_bf16", &GqaAttentionBF16Factory::from_spec);
  return true;
}();
}  // namespace

}  // namespace alice::block
