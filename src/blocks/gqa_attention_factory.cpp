#include "gqa_attention_factory.hpp"

#include <stdexcept>
#include <string>
#include <algorithm>

#include <nlohmann/json.hpp>

#include "alice/tensor.hpp"   // bf16, TensorView
#include "gpu_rocm/gqa_attention.hpp"  // gpu_rocm::GqaAttentionBF16
#include "gpu_rocm/kernels.hpp"        // gemv_partial_floats, kAttn* constants

namespace alice::block {

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("block factory: " + what);
}

}  // namespace

// JSON binding for the block-type config (design §8): plain POD, the macro
// is the only nlohmann touch and stays in the factory .cpp. The optional
// rope_scaling group is conditional (present-type => four more required
// fields) so it is NOT in the macro — read manually in from_spec.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GqaAttentionConfig, io_dim, qk_dim, v_dim,
                                   num_q_heads, num_kv_heads, rope_theta)

uint64_t GqaAttentionBF16Factory::derived_bytes() const {
  const uint32_t nh = cfg_.num_q_heads, nkv = cfg_.num_kv_heads,
                 hd = cfg_.qk_dim, io = cfg_.io_dim;
  return uint64_t(2 * nh + 2 * nkv) * hd * io * sizeof(bf16);
}

uint64_t GqaAttentionBF16Factory::device_bytes(const ModelConfig& mc) const {
  // Mirrors the constructor's allocations (gqa_attention.cpp), but sized for
  // this rank's tensor-parallel shard. Each padded to gpu::kDeviceAlign so
  // the arena total is exactly predictable.
  const auto pad = [](size_t n) {
    return (n + gpu::kDeviceAlign - 1) & ~(gpu::kDeviceAlign - 1);
  };
  const uint32_t tp = mc.tp_degree ? mc.tp_degree : 1;
  const uint32_t nh = cfg_.num_q_heads, nkv = cfg_.num_kv_heads,
                 hd = cfg_.qk_dim, io = cfg_.io_dim;
  const uint32_t qd = nh * hd / tp, kvd = nkv * hd / tp, qkvd = qd + 2 * kvd;
  const uint32_t mpc = mc.max_prefill_chunk, max_seq = mc.max_seq_len;
  const uint32_t half = hd / 2;
  uint64_t total = 0;
  total += pad(size_t(qkvd) * io * sizeof(bf16));                     // dw_qkv_
  total += pad(size_t(io) * qd * sizeof(bf16));                      // dw_o_
  total += pad(size_t(max_seq) * kvd * sizeof(float));               // dkcache_
  total += pad(size_t(max_seq) * kvd * sizeof(float));               // dvcache_
  total += pad(size_t(max_seq) * half * sizeof(float));              // drope_cos_
  total += pad(size_t(max_seq) * half * sizeof(float));              // drope_sin_
  total += pad(size_t(mpc) * io * sizeof(bf16));                     // dx_
  total += pad(size_t(mpc) * qkvd * sizeof(float));                  // dqkv_
  total += pad(size_t(mpc) * qd * sizeof(bf16));                     // dattn_
  total += pad(size_t(gpu_rocm::kernels::kAttnMaxSplits) * (nh / tp) *
               gpu_rocm::kernels::kAttnSplitMaxS * (hd + 2) * sizeof(float));  // dsplit_
  const size_t gemv = std::max(gpu_rocm::kernels::gemv_partial_floats(qkvd, io),
                               gpu_rocm::kernels::gemv_partial_floats(io, qd));
  total += pad(gemv * sizeof(float));                                // dgemv_
  return total;
}

size_t GqaAttentionBF16Factory::scratch_bytes(const ModelConfig& mc) const {
  // v4 prefill partials: (nh/tp) * max_prefill_chunk * ceil(max_seq/64) * (hd+2)
  const uint32_t tp = mc.tp_degree ? mc.tp_degree : 1;
  const uint64_t max_chunks = (uint64_t(mc.max_seq_len) + 63) / 64;
  return uint64_t(cfg_.num_q_heads / tp) * mc.max_prefill_chunk * max_chunks *
         (cfg_.qk_dim + 2) * sizeof(float);
}

std::string GqaAttentionBF16Factory::interface_sig() const {
  return "io:" + std::to_string(cfg_.io_dim);
}

std::unique_ptr<Block> GqaAttentionBF16Factory::construct(
    std::span<const std::byte> data, gpu::Resources& res, const ModelConfig& mc) const {
  if (data.size() != derived_bytes())
    fail("gqa_attention_bf16: decoded chunk is " + std::to_string(data.size()) +
         " bytes but the v1 layout derives " + std::to_string(derived_bytes()));
  const uint32_t tp = mc.tp_degree ? mc.tp_degree : 1;
  const uint32_t rank = static_cast<uint32_t>(res.rank());
  const uint32_t nh = cfg_.num_q_heads, nkv = cfg_.num_kv_heads,
                 hd = cfg_.qk_dim, io = cfg_.io_dim;
  if (nh % tp != 0 || nkv % tp != 0)
    fail("gqa_attention_bf16: tp_degree does not divide num_q_heads/num_kv_heads");
  const uint32_t qd = nh * hd, kvd = nkv * hd;
  const uint32_t qd_local = qd / tp, kvd_local = kvd / tp;

  const bf16* base = reinterpret_cast<const bf16*>(data.data());
  // Column-parallel Q/K/V: contiguous slices along the output-feature dim.
  const bf16* base_q = base;
  const bf16* base_k = base + size_t(qd) * io;
  const bf16* base_v = base + size_t(qd + kvd) * io;
  TensorView<bf16> wq(base_q + rank * size_t(qd_local) * io, qd_local, io);
  TensorView<bf16> wk(base_k + rank * size_t(kvd_local) * io, kvd_local, io);
  TensorView<bf16> wv(base_v + rank * size_t(kvd_local) * io, kvd_local, io);
  // Row-parallel O: strided slice along the input-feature (head) dim.
  TensorView<bf16> wo(base + size_t(qd + 2 * kvd) * io + rank * qd_local,
                      io, qd_local, qd);
  return std::make_unique<gpu_rocm::GqaAttentionBF16>(wq, wk, wv, wo, cfg_, mc, res);
}

std::unique_ptr<BlockFactory> GqaAttentionBF16Factory::from_spec(uint32_t version,
                                                                 const nlohmann::json& config) {
  if (version != 1)
    fail("gqa_attention_bf16: unsupported version " + std::to_string(version) +
         " (registry provides version 1)");
  GqaAttentionConfig cfg = config.get<GqaAttentionConfig>();
  if (cfg.qk_dim != cfg.v_dim)
    fail("gqa_attention_bf16: qk_dim != v_dim (v1 requires equal head dims; "
         "separate value head dim is a future version)");
  if (cfg.io_dim == 0 || cfg.num_q_heads == 0 || cfg.num_kv_heads == 0 ||
      cfg.qk_dim == 0 || cfg.num_q_heads % cfg.num_kv_heads != 0)
    fail("gqa_attention_bf16: bad head config (io/heads/kv-heads/head-dim, "
         "num_q_heads must divide num_kv_heads evenly)");
  auto r = std::make_unique<GqaAttentionBF16Factory>();
  if (config.contains("rope_scaling_type")) {
    const std::string type = config.at("rope_scaling_type").get<std::string>();
    if (type != "llama3")
      fail("gqa_attention_bf16: unsupported rope_scaling_type '" + type + "'");
    RopeScaling rs;
    rs.factor = config.at("rope_scaling_factor").get<float>();
    rs.low_freq_factor = config.at("rope_scaling_low_freq_factor").get<float>();
    rs.high_freq_factor = config.at("rope_scaling_high_freq_factor").get<float>();
    rs.original_max_position_embeddings =
        config.at("original_max_position_embeddings").get<uint32_t>();
    cfg.rope_scaling = rs;
  }
  r->cfg_ = std::move(cfg);
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
