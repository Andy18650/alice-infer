// The alice model format loader — first implementation round
// (docs/model-format.md RC1, docs/model-loader.md §8 scope). Blocks only;
// members parsed structurally and set aside. Signature verification is
// deferred with the chain of trust: the envelope's signature state is
// DECLARED loudly, not checked.

#include "model_format.hpp"

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
#include <blake3.h>
#include <nlohmann/json.hpp>

#include "gpu/block.hpp"
#include "blocks/gpu_rocm/platform.hpp"
#include "blocks/gpu_rocm/dense_ffn.hpp"
#include "blocks/gpu_rocm/gqa_attention.hpp"
#include "manifest.hpp"  // ModelConfig: the block constructors' vocabulary
#include "alice/tensor.hpp"

namespace alice::modelfmt {

using nlohmann::json;
using gpu::Block;
using gpu_rocm::Context;

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("model loader: " + what);
}

// BLAKE3 of a byte range as lowercase hex (the record-hash comparison
// vocabulary — the descriptor stores hashes as text). Malformed stored
// hashes funnel into the single mismatch error at the comparison site.
std::string blake3_hex(const void* data, size_t len) {
  blake3_hasher h;
  blake3_hasher_init(&h);
  blake3_hasher_update(&h, data, len);
  uint8_t d[32];
  blake3_hasher_finalize(&h, d, sizeof(d));
  static const char* hex = "0123456789abcdef";
  std::string s(64, '0');
  for (size_t i = 0; i < 32; ++i) {
    s[2 * i] = hex[d[i] >> 4];
    s[2 * i + 1] = hex[d[i] & 0xF];
  }
  return s;
}

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
  }
};

// --- config: stored raw; types parse their own (struct +
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE) -------------------------------------------------

// Records parse via get<Record>() (NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE in
// the header); json exceptions carry the key name. Unknown fields are
// ignored: schema evolution is version-governed (a field addition is a
// version bump, and an old engine rejects the file on the version —
// catching a forgotten bump is review/CI's job, not the loader's).

// MemberDecl, BlockSpecDecl, and BlockEntry parse via their
// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE from_json (header) — json members and
// nested macro-defined types compose correctly, and faults throw with the
// full JSON path (JSON_DIAGNOSTICS), converted once at the load_blocks
// boundary. Only SEMANTIC checks hand-build context (unknown
// type/version/codec, hash mismatches): things the json library cannot
// know.
Descriptor parse_descriptor(const json& m) {
  Descriptor d;
  d.name = m.at("name").get<std::string>();
  d.version = m.at("version").get<uint32_t>();
  if (d.version != 1)
    fail("descriptor: unsupported version " + std::to_string(d.version) +
         " (this loader reads version 1)");
  d.max_context = m.at("max_context").get<uint64_t>();
  if (m.contains("producer")) d.producer = m.at("producer").get<std::string>();
  if (m.contains("packager")) d.packager = m.at("packager").get<std::string>();
  if (m.contains("license")) d.license = m.at("license").get<std::string>();
  if (m.contains("license_url")) d.license_url = m.at("license_url").get<std::string>();
  if (m.contains("license_text")) d.license_text = m.at("license_text").get<std::string>();

  // Members: parsed structurally, set aside this round (docs/model-loader.md).
  for (const char* name :
       {"chat_template", "tokenizer", "embedder", "language_head", "trunk"}) {
    if (m.contains(name)) d.members[name] = m.at(name).get<MemberDecl>();
  }

  if (!m.at("block_specs").is_object()) fail("descriptor: block_specs must be an object");
  for (auto it = m.at("block_specs").begin(); it != m.at("block_specs").end(); ++it)
    d.specs[it.key()] = it.value().get<BlockSpecDecl>();

  const json& blocks = m.at("blocks");
  if (!blocks.is_array() || blocks.empty())
    fail("descriptor: blocks must be a non-empty array");
  for (const auto& b : blocks)
    d.blocks.push_back(b.get<BlockEntry>());
  return d;
}

}  // namespace

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
// Non-movable and non-copyable — the load loop opens a fresh one per
// block, so at most one fd is open at any moment. pread, not ifstream:
// offset-addressed and stateless (prefetch-pipeline safe), matching the
// §9 weight path — streams for text, syscalls for bulk bytes.
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

namespace {

// All json faults (missing keys, wrong types) convert here, ONCE: with
// JSON_DIAGNOSTICS the exception message carries the full JSON pointer
// path (e.g. "/model/block_specs/ffn/config"), replacing the hand-rolled
// where-string plumbing. json::exception is not a runtime_error, so this
// also keeps load_blocks' exception contract uniform.
LoadedBlocks load_blocks_impl(const std::filesystem::path& model_file,
                              const Context& ctx, const EngineConfig& ec,
                              LoadStats& st) {
  LoadedBlocks out;
  out.allocator = std::make_unique<DeviceAllocator>();

  // --- framing + envelope + descriptor --------------------------------------
  Timer t;
  const json envelope = read_envelope(model_file);
  if (!envelope.is_object()) fail("envelope must be an object");
  const json& alice = envelope.at("Alice");
  if (!alice.is_object()) fail("envelope.Alice must be an object");
  if (alice.at("format_version").get<uint64_t>() != 1)
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
  st.parse_ms = t.ms();
  st.specs = out.descriptor.specs.size();

  // --- spec pre-resolution + composition check: every spec resolves
  // against the registry (config faults carry spec context), and all
  // specs must present the same interface.
  t = Timer{};
  std::map<std::string, std::unique_ptr<ResolvedSpec>> resolved;
  std::string iface;
  for (const auto& [name, decl] : out.descriptor.specs) {
    const std::string where = "block_specs." + name;
    const auto it = registry().find(decl.type);
    if (it == registry().end())
      fail(where + ": unknown block type '" + decl.type + "'");
    std::unique_ptr<ResolvedSpec> rs;
    try {
      rs = it->second(decl);
    } catch (const json::exception& e) {
      // BlockSpecDecl::config is a COPY of the document subtree, so the
      // diagnostics parent chain (and JSON path) is severed — this one
      // site re-adds its context by hand. Everything parsed in place
      // carries its path natively.
      fail(where + " (" + decl.type + " v" + std::to_string(decl.version) +
           "): " + e.what());
    }
    const std::string isig = rs->interface_sig();
    if (!isig.empty()) {
      if (iface.empty())
        iface = isig;
      else if (iface != isig)
        fail(where + ": interface " + isig + " disagrees with " + iface +
             " — composition does not bind");
    }
    resolved[name] = std::move(rs);
  }

  // --- scratch: max over specs, allocated once, before construction ----------
  for (const auto& [_, rs] : resolved) out.allocator->plan_scratch(rs->scratch_bytes(ec));
  out.allocator->init_scratch();
  st.scratch_bytes = out.allocator->scratch_bytes();

  // --- the block loop: single pass over the composition — per block,
  // checks (spec ref, codec) then read -> hash -> construct. A bad
  // record at block i fails after blocks < i are loaded: accepted
  // (files are machine-packed; RAII teardown handles the partial state).
  const std::filesystem::path base = model_file.parent_path();
  for (size_t i = 0; i < out.descriptor.blocks.size(); ++i) {
    const BlockEntry& e = out.descriptor.blocks[i];
    const std::string where = "blocks[" + std::to_string(i) + "]";
    const auto rit = resolved.find(e.spec);
    if (rit == resolved.end())
      fail(where + ": references unknown spec '" + e.spec + "'");
    const ResolvedSpec* rs = rit->second.get();
    if (e.codec != "raw")
      fail(where + ": codec '" + e.codec +
           "' is not supported (format v1 provides 'raw' only)");
    // NOTE: no derived==e.size check here. Stored size equals the layout-
    // derived size only as a raw-codec coincidence; the contract is the
    // factory-side reconciliation (decoded length == derived, exactly) —
    // which for raw checks the same mismatch, since decoded == stored.

    t = Timer{};
    Chunk chunk(size_t(e.size));
    const Fd fd(base / e.file);
    size_t got = 0;
    while (got < e.size) {
      const ssize_t n =
          ::pread(fd.get(), chunk.data() + got, e.size - got, off_t(e.offset + got));
      if (n <= 0)
        fail(where + " (" + e.file + ":" + std::to_string(e.offset) +
             "): short read (" + std::to_string(got) + "/" +
             std::to_string(e.size) + " bytes): " + std::strerror(errno));
      got += size_t(n);
    }
    st.read_ms += t.ms();
    st.bytes_read += e.size;

    t = Timer{};
    if (blake3_hex(chunk.data(), chunk.size()) != e.hash)
      fail(where + " (" + e.file + ":" + std::to_string(e.offset) +
           "): BLAKE3 mismatch — record is corrupt or tampered");
    st.hash_ms += t.ms();
    ++st.hash_verifications;

    t = Timer{};
    out.blocks.push_back(rs->construct(std::move(chunk), *out.allocator, ctx, ec));
    st.construct_ms += t.ms();
  }
  st.blocks = out.blocks.size();
  return out;
}

}  // namespace

LoadedBlocks load_blocks(const std::filesystem::path& model_file,
                         const Context& ctx, const EngineConfig& ec,
                         LoadStats* out_stats) {
  LoadStats dummy;
  try {
    return load_blocks_impl(model_file, ctx, ec, out_stats ? *out_stats : dummy);
  } catch (const json::exception& e) {
    fail(std::string("descriptor: ") + e.what());
  }
}

}  // namespace alice::modelfmt
