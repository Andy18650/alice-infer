// The alice model format loader — first implementation round
// (docs/model-format.md RC1, docs/model-loader.md §8 scope). Blocks only;
// members parsed structurally and set aside. Signature verification is
// deferred with the chain of trust: the envelope's signature state is
// DECLARED loudly, not checked.

#include "alice/model_format.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <hip/hip_runtime_api.h>
#include <nlohmann/json.hpp>

#include "gpu/block.hpp"
#include "blocks/gpu_rocm/platform.hpp"
#include "blocks/gpu_rocm/dense_ffn.hpp"
#include "blocks/gpu_rocm/gqa_attention.hpp"
#include "alice/blake3.hpp"
#include "alice/manifest.hpp"  // ModelConfig: the block constructors' vocabulary
#include "alice/tensor.hpp"

namespace alice::modelfmt {

using nlohmann::json;
using gpu::Block;
using gpu_rocm::Context;

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("model loader: " + what);
}

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
  }
};

void expect_keys(const json& j, const std::string& where,
                 std::initializer_list<const char*> allowed) {
  for (auto it = j.begin(); it != j.end(); ++it) {
    bool ok = false;
    for (const char* a : allowed)
      if (it.key() == a) ok = true;
    if (!ok) fail(where + ": unknown field '" + it.key() + "'");
  }
}

uint64_t get_u(const json& j, const char* key, const std::string& where) {
  if (!j.contains(key) || !j.at(key).is_number_integer())
    fail(where + ": field '" + key + "' must be an integer");
  const int64_t v = j.at(key).get<int64_t>();
  if (v < 0) fail(where + ": field '" + key + "' must be non-negative");
  return uint64_t(v);
}

std::string get_s(const json& j, const char* key, const std::string& where) {
  if (!j.contains(key) || !j.at(key).is_string())
    fail(where + ": field '" + key + "' must be a string");
  return j.at(key).get<std::string>();
}

// --- config: stored raw; types parse their own (struct +
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE) -------------------------------------------------

// --- record (fields inline in the owning object, per the v0.2 example;
// the OWNING object's expect_keys enforces the full key set, so this only
// reads the five record fields) ----------------------------------------------
Record parse_record(const json& j, const std::string& where) {
  Record r;
  r.file = get_s(j, "file", where);
  if (r.file.empty()) fail(where + ": 'file' must be non-empty");
  r.offset = get_u(j, "offset", where);
  r.size = get_u(j, "size", where);
  r.codec = get_s(j, "codec", where);
  r.hash = blake3::from_hex(get_s(j, "hash", where));
  return r;
}

MemberDecl parse_member(const json& j, const std::string& where) {
  expect_keys(j, where,
              {"type", "version", "config", "file", "offset", "size", "codec", "hash"});
  MemberDecl m;
  m.type = get_s(j, "type", where);
  m.version = uint32_t(get_u(j, "version", where));
  if (!j.contains("config")) fail(where + ": missing 'config'");
  m.config = j.at("config");
  m.record = parse_record(j, where);
  return m;
}

BlockSpecDecl parse_spec(const json& j, const std::string& where) {
  expect_keys(j, where, {"type", "version", "config"});
  BlockSpecDecl s;
  s.type = get_s(j, "type", where);
  s.version = uint32_t(get_u(j, "version", where));
  if (!j.contains("config")) fail(where + ": missing 'config'");
  s.config = j.at("config");
  return s;
}

Descriptor parse_descriptor(const json& m) {
  const std::string w = "descriptor";
  expect_keys(m, w,
              {"name", "version", "producer", "packager", "license",
               "license_url", "license_text", "max_context",
               "chat_template", "tokenizer", "embedder", "language_head",
               "trunk", "block_specs", "blocks"});
  Descriptor d;
  d.name = get_s(m, "name", w);
  d.version = uint32_t(get_u(m, "version", w));
  if (d.version != 1)
    fail(w + ": unsupported descriptor version " + std::to_string(d.version) +
         " (this loader reads version 1)");
  d.max_context = get_u(m, "max_context", w);
  if (m.contains("producer")) d.producer = get_s(m, "producer", w);
  if (m.contains("packager")) d.packager = get_s(m, "packager", w);
  if (m.contains("license")) d.license = get_s(m, "license", w);
  if (m.contains("license_url")) d.license_url = get_s(m, "license_url", w);
  if (m.contains("license_text")) d.license_text = get_s(m, "license_text", w);

  // Members: parsed structurally, set aside this round (docs/model-loader.md).
  for (const char* name :
       {"chat_template", "tokenizer", "embedder", "language_head", "trunk"}) {
    if (m.contains(name))
      d.members[name] = parse_member(m.at(name), std::string(w) + "." + name);
  }

  if (!m.contains("block_specs")) fail(w + ": missing 'block_specs'");
  if (!m.at("block_specs").is_object()) fail(w + ": block_specs must be an object");
  for (auto it = m.at("block_specs").begin(); it != m.at("block_specs").end(); ++it)
    d.specs[it.key()] = parse_spec(it.value(), "block_specs." + it.key());

  if (!m.contains("blocks")) fail(w + ": missing 'blocks'");
  if (!m.at("blocks").is_array() || m.at("blocks").empty())
    fail(w + ": blocks must be a non-empty array");
  size_t i = 0;
  for (const auto& b : m.at("blocks")) {
    const std::string where = "blocks[" + std::to_string(i++) + "]";
    expect_keys(b, where, {"spec", "file", "offset", "size", "codec", "hash"});
    BlockEntry e;
    e.spec = get_s(b, "spec", where);
    e.record = parse_record(b, where);
    d.blocks.push_back(std::move(e));
  }
  return d;
}

}  // namespace

// ---------------------------------------------------------------------------
// BlockConfig accessors — REMOVED (2026-09): configs cross the handoff as raw
// json; each type parses its own via struct + NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE
// in its resolve function. Missing/wrong keys throw with spec context;
// unknown keys are ignored (type-key drift is review-time business).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Registry: the two v1 block types
// ---------------------------------------------------------------------------

namespace {

// dense_ffn_bf16 v1 config (missing keys throw via the macro's from_json).
struct DenseFfnV1Config {
  uint32_t io_dim = 0;
  uint32_t intermediate_dim = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DenseFfnV1Config, io_dim, intermediate_dim)

// dense_ffn_bf16 v1 layout: [w_gate (inter x io)] [w_up (inter x io)]
// [w_down (io x inter)], bf16 row-major, no padding.
struct DenseFfnResolved : ResolvedSpec {
  uint32_t io = 0, inter = 0;

  uint64_t derived_bytes() const override {
    return 3 * uint64_t(io) * inter * sizeof(bf16);
  }
  size_t scratch_bytes(const EngineConfig&) const override { return 0; }
  std::string interface_sig() const override { return "io:" + std::to_string(io); }

  std::unique_ptr<Block> construct(Chunk chunk, DeviceAllocator& alloc,
                                   const Context& ctx,
                                   const EngineConfig& ec) const override {
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
};

std::unique_ptr<ResolvedSpec> resolve_dense_ffn(const BlockSpecDecl& d) {
  if (d.version != 1)
    fail("dense_ffn_bf16: unsupported version " + std::to_string(d.version) +
         " (registry provides version 1)");
  const DenseFfnV1Config cfg = d.config.get<DenseFfnV1Config>();
  auto r = std::make_unique<DenseFfnResolved>();
  r->io = cfg.io_dim;
  r->inter = cfg.intermediate_dim;
  if (r->io == 0 || r->inter == 0) fail("dense_ffn_bf16: io_dim/intermediate_dim must be non-zero");
  return r;
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

// gqa_attention_bf16 v1 layout: [wq (nh*hd x io)] [wk (nkv*hd x io)]
// [wv (nkv*hd x io)] [wo (io x nh*hd)], bf16 row-major, no padding.
// v1 requires qk_dim == v_dim (the ModelConfig vocabulary has one head dim).
struct GqaAttentionResolved : ResolvedSpec {
  uint32_t io = 0, nh = 0, nkv = 0, hd = 0;
  ModelConfig mc;

  uint64_t derived_bytes() const override {
    return uint64_t(2 * nh + 2 * nkv) * hd * io * sizeof(bf16);
  }
  size_t scratch_bytes(const EngineConfig& ec) const override {
    // v4 prefill partials: nh * max_prefill_chunk * ceil(max_seq/64) * (hd+2)
    const uint64_t max_chunks = (uint64_t(ec.max_seq_len) + 63) / 64;
    return uint64_t(nh) * ec.max_prefill_chunk * max_chunks * (hd + 2) * sizeof(float);
  }
  std::string interface_sig() const override { return "io:" + std::to_string(io); }

  std::unique_ptr<Block> construct(Chunk chunk, DeviceAllocator& alloc,
                                   const Context& ctx,
                                   const EngineConfig& ec) const override {
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
    return std::make_unique<gpu_rocm::GqaAttentionBF16>(wq, wk, wv, wo, mc, ec,
                                                        ctx, alloc.scratch_ptr());
  }
};

std::unique_ptr<ResolvedSpec> resolve_gqa_attention(const BlockSpecDecl& d) {
  if (d.version != 1)
    fail("gqa_attention_bf16: unsupported version " + std::to_string(d.version) +
         " (registry provides version 1)");
  const GqaAttentionV1Config cfg = d.config.get<GqaAttentionV1Config>();
  auto r = std::make_unique<GqaAttentionResolved>();
  r->io = cfg.io_dim;
  r->nh = cfg.num_q_heads;
  r->nkv = cfg.num_kv_heads;
  if (cfg.qk_dim != cfg.v_dim)
    fail("gqa_attention_bf16: qk_dim != v_dim (v1 requires equal head dims; "
         "separate value head dim is a future version)");
  r->hd = cfg.qk_dim;
  if (r->io == 0 || r->nh == 0 || r->nkv == 0 || r->hd == 0 ||
      r->nh % r->nkv != 0)
    fail("gqa_attention_bf16: bad head config (io/heads/kv-heads/head-dim, "
         "num_q_heads must divide num_kv_heads evenly)");

  ModelConfig& mc = r->mc;
  mc.hidden_size = r->io;
  mc.num_heads = r->nh;
  mc.num_kv_heads = r->nkv;
  mc.head_dim = r->hd;
  mc.rope_theta = float(cfg.rope_theta);
  if (d.config.contains("rope_scaling_type")) {
    const std::string type = d.config.at("rope_scaling_type").get<std::string>();
    if (type != "llama3")
      fail("gqa_attention_bf16: unsupported rope_scaling_type '" + type + "'");
    mc.has_rope_scaling = true;
    mc.rope_scaling.factor = d.config.at("rope_scaling_factor").get<float>();
    mc.rope_scaling.low_freq_factor =
        d.config.at("rope_scaling_low_freq_factor").get<float>();
    mc.rope_scaling.high_freq_factor =
        d.config.at("rope_scaling_high_freq_factor").get<float>();
    mc.rope_scaling.original_max_position_embeddings =
        d.config.at("original_max_position_embeddings").get<uint32_t>();
  }
  return r;
}

}  // namespace

const std::map<std::string, ResolveFn*>& registry() {
  static const std::map<std::string, ResolveFn*> r = {
      {"dense_ffn_bf16", &resolve_dense_ffn},
      {"gqa_attention_bf16", &resolve_gqa_attention},
  };
  return r;
}

// ---------------------------------------------------------------------------
// Chunk + DeviceAllocator
// ---------------------------------------------------------------------------

Chunk::Chunk(size_t bytes) : size_(bytes) {
  if (bytes) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
    (void)madvise(p, bytes, MADV_HUGEPAGE);  // THP staging (design §9)
    data_ = static_cast<uint8_t*>(p);
  }
}

Chunk::~Chunk() {
  if (data_) (void)munmap(data_, size_);
}

Chunk::Chunk(Chunk&& o) noexcept : data_(o.data_), size_(o.size_) {
  o.data_ = nullptr;
  o.size_ = 0;
}

Chunk& Chunk::operator=(Chunk&& o) noexcept {
  if (this != &o) {
    if (data_) (void)munmap(data_, size_);
    data_ = o.data_;
    size_ = o.size_;
    o.data_ = nullptr;
    o.size_ = 0;
  }
  return *this;
}

DeviceAllocator::~DeviceAllocator() {
  for (auto& [name, np] : named_)
    if (np.first) (void)hipFree(np.first);
  if (scratch_) (void)hipFree(scratch_);
}

void* DeviceAllocator::named(const std::string& name, size_t bytes) {
  auto it = named_.find(name);
  if (it != named_.end()) {
    if (it->second.second != bytes)
      fail("allocator: named buffer '" + name + "' requested at " +
           std::to_string(bytes) + " bytes but exists at " +
           std::to_string(it->second.second) + " bytes");
    return it->second.first;
  }
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess)
    fail("allocator: hipMalloc(" + std::to_string(bytes) + ") failed for '" +
         name + "'");
  named_[name] = {p, bytes};
  return p;
}

void DeviceAllocator::plan_scratch(size_t bytes) {
  scratch_planned_ = std::max(scratch_planned_, bytes);
}

void DeviceAllocator::init_scratch() {
  if (scratch_initialized_) return;
  scratch_initialized_ = true;
  scratch_bytes_ = scratch_planned_;
  if (scratch_planned_ == 0) return;
  if (hipMalloc(reinterpret_cast<void**>(&scratch_), scratch_planned_) != hipSuccess)
    fail("allocator: scratch hipMalloc(" + std::to_string(scratch_planned_) +
         ") failed");
  // Poison fill (quiet NaN): read-before-first-write becomes visible.
  if (hipMemsetD32(scratch_, 0x7fc00000, uint32_t(scratch_planned_ / 4)) !=
      hipSuccess)
    fail("allocator: scratch poison fill failed");
}

// ---------------------------------------------------------------------------
// load_blocks
// ---------------------------------------------------------------------------

namespace {

// A file descriptor. RAII: closed at scope exit, including on throw.
// Deliberately non-movable and non-copyable — the load loop opens a fresh
// one per block, so at most one fd is open at any moment and there is
// nothing to move. (Replaces the earlier FdCache: per-file open caching
// saved ~0.1 ms per load — measured — while carrying a path-keyed map and
// load-scoped fd lifetimes; the guard is stateless. pread, not ifstream:
// offset-addressed and stateless, matching the §9 weight path — streams
// are for text, syscalls for bulk bytes.)
class Fd {
 public:
  Fd(const std::filesystem::path& p) {
    fd_ = ::open(p.c_str(), O_RDONLY);
    if (fd_ < 0)
      fail("cannot open data file '" + p.string() + "': " + std::strerror(errno));
  }
  ~Fd() {
    if (fd_ >= 0) (void)::close(fd_);
  }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;

  int get() const { return fd_; }

 private:
  int fd_ = -1;
};

// Reads a .model descriptor file and returns the parsed envelope: text
// region (everything before the first NUL, which must be \n- or \r\n-
// terminated; a binary tail may follow in single-file models, so no
// whole-file read), {"Alice" magic, then the json parse. The raw text is
// local — only the parsed envelope crosses out (the string is freed once
// the json is constructed). Descriptor budget: real ones measure <100 KiB
// (the 70B-with-MoE projects to ~32 KiB); the 1 MiB read cap exists to
// catch "wrong file pointed at the loader", not as a designed limit.
json read_envelope(const std::filesystem::path& file) {
  constexpr size_t kMaxDescriptor = 1 << 20;

  std::ifstream in(file, std::ios::binary);
  if (!in) fail("cannot open '" + file.string() + "'");
  std::string text(kMaxDescriptor, '\0');
  in.read(text.data(), kMaxDescriptor);  // partial read at EOF is fine
  const size_t got = size_t(in.gcount());

  // Magic first: a wrong file should fail HERE, not on some
  // incidental terminator detail deeper in.
  if (got < 8 || text.compare(0, 8, "{\"Alice\"") != 0)
    fail("bad magic: a .model file must begin with {\"Alice\"");

  const size_t nul = text.find('\0');
  if (nul == std::string::npos || nul >= got)
    fail("'" + file.string() + "' has no NUL within the first 1 MiB — "
         "not a .model file?");
  if (nul == 0 || text[nul - 1] != '\n')
    fail("text region must end with \\n before the NUL gate");
  size_t end = nul - 1;  // drop the \n
  if (end > 0 && text[end - 1] == '\r') --end;  // tolerate \r\n
  text.resize(end);

  json envelope;
  try {
    envelope = json::parse(text);
  } catch (const json::exception& e) {
    fail(std::string("envelope JSON: ") + e.what());
  }
  return envelope;
}

}  // namespace

LoadedBlocks load_blocks(const std::filesystem::path& model_file,
                         const Context& ctx, const EngineConfig& ec) {
  LoadedBlocks out;
  out.allocator = std::make_unique<DeviceAllocator>();

  // --- framing + envelope + descriptor --------------------------------------
  Timer t;
  const json envelope = read_envelope(model_file);
  if (!envelope.is_object()) fail("envelope must be an object");
  expect_keys(envelope, "envelope", {"Alice", "model"});
  const json& alice = envelope.at("Alice");
  if (!alice.is_object()) fail("envelope.Alice must be an object");
  expect_keys(alice, "envelope.Alice", {"format_version", "signature"});
  if (get_u(alice, "format_version", "envelope.Alice") != 1)
    fail("envelope.Alice: unsupported format_version (this loader reads 1)");
  if (alice.contains("signature")) {
    std::string key = "?", alg = "?";
    const json& s = alice.at("signature");
    if (s.is_object()) {
      if (s.contains("key")) key = s.at("key").get<std::string>();
      if (s.contains("alg")) alg = s.at("alg").get<std::string>();
    }
    std::fprintf(stderr,
                 "[model loader] descriptor carries a signature (alg=%s key=%s) — "
                 "verification not implemented (chain-of-trust round); proceeding\n",
                 alg.c_str(), key.c_str());
  } else {
    std::fprintf(stderr,
                 "[model loader] unsigned descriptor — proceeding (dev artifacts only)\n");
  }
  const json& model = envelope.at("model");
  if (!model.is_object()) fail("envelope.model must be an object");
  out.descriptor = parse_descriptor(model);
  out.stats.parse_ms = t.ms();
  out.stats.specs = out.descriptor.specs.size();

  // --- spec pre-resolution (registry lookup; config parse errors get spec
  // context; no consumption check — type-key drift is review-time business)
  t = Timer{};
  std::map<std::string, std::unique_ptr<ResolvedSpec>> resolved;
  for (const auto& [name, decl] : out.descriptor.specs) {
    const std::string where = "block_specs." + name;
    const auto it = registry().find(decl.type);
    if (it == registry().end())
      fail(where + ": unknown block type '" + decl.type + "'");
    std::unique_ptr<ResolvedSpec> rs;
    try {
      rs = it->second(decl);
    } catch (const json::exception& e) {
      fail(where + " (" + decl.type + " v" + std::to_string(decl.version) +
           "): config: " + e.what());
    }
    resolved[name] = std::move(rs);
  }

  // --- block plan: resolve refs, codec/size pre-read checks, interface bind --
  std::vector<std::pair<const ResolvedSpec*, const Record*>> plan;
  std::string iface;
  for (size_t i = 0; i < out.descriptor.blocks.size(); ++i) {
    const BlockEntry& e = out.descriptor.blocks[i];
    const std::string where = "blocks[" + std::to_string(i) + "]";
    const auto rit = resolved.find(e.spec);
    if (rit == resolved.end())
      fail(where + ": references unknown spec '" + e.spec + "'");
    const ResolvedSpec* rs = rit->second.get();
    if (e.record.codec != "raw")
      fail(where + ": codec '" + e.record.codec +
           "' is not supported (format v1 provides 'raw' only)");
    if (rs->derived_bytes() != e.record.size)
      fail(where + " (spec '" + e.spec + "'): record size " +
           std::to_string(e.record.size) + " != layout-derived " +
           std::to_string(rs->derived_bytes()) + " bytes");
    const std::string isig = rs->interface_sig();
    if (!isig.empty()) {
      if (iface.empty())
        iface = isig;
      else if (iface != isig)
        fail(where + ": interface " + isig + " disagrees with earlier blocks (" +
             iface + ") — composition does not bind");
    }
    plan.emplace_back(rs, &e.record);
  }

  // --- scratch: max over specs, allocated once, before construction ----------
  for (const auto& [_, rs] : resolved) out.allocator->plan_scratch(rs->scratch_bytes(ec));
  out.allocator->init_scratch();
  out.stats.scratch_bytes = out.allocator->scratch_bytes();

  // --- the block loop: read -> hash -> construct ------------------------------
  const std::filesystem::path base = model_file.parent_path();
  for (size_t i = 0; i < plan.size(); ++i) {
    const auto& [rs, r] = plan[i];
    const std::string where = "blocks[" + std::to_string(i) + "] (" + r->file +
                              ":" + std::to_string(r->offset) + ")";

    t = Timer{};
    Chunk chunk(size_t(r->size));
    const Fd fd(base / r->file);
    size_t got = 0;
    while (got < r->size) {
      const ssize_t n =
          ::pread(fd.get(), chunk.data() + got, r->size - got, off_t(r->offset + got));
      if (n <= 0)
        fail(where + ": short read (" + std::to_string(got) + "/" +
             std::to_string(r->size) + " bytes): " + std::strerror(errno));
      got += size_t(n);
    }
    out.stats.read_ms += t.ms();
    out.stats.bytes_read += r->size;

    t = Timer{};
    if (blake3::hash(chunk.data(), chunk.size()) != r->hash)
      fail(where + ": BLAKE3 mismatch — record is corrupt or tampered");
    out.stats.hash_ms += t.ms();
    ++out.stats.hash_verifications;

    t = Timer{};
    out.blocks.push_back(rs->construct(std::move(chunk), *out.allocator, ctx, ec));
    out.stats.construct_ms += t.ms();
  }
  out.stats.blocks = out.blocks.size();
  return out;
}

}  // namespace alice::modelfmt
