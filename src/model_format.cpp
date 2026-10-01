// The alice model format loader — first implementation round
// (docs/model-format.md RC1, docs/model-loader.md §8 scope). Blocks only;
// members parsed structurally and set aside. Signature verification is
// deferred with the chain of trust: the envelope's signature state is
// DECLARED loudly, not checked.

#include "model_format.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <span>
#include <stdexcept>

#include <hip/hip_runtime_api.h>
#include <nlohmann/json.hpp>

#include "blocks/block_factory.hpp"  // BlockFactory + registry() (block layer)
#include "gpu/resources.hpp"
#include "data_loader.hpp"  // LocalFileDataLoader (content-addressed fetch)
#include "manifest.hpp"  // ModelConfig: the block constructors' vocabulary
#include "alice/tensor.hpp"

namespace alice::modelfmt {

using nlohmann::json;
using gpu::Resources;

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
// load_blocks
// ---------------------------------------------------------------------------

namespace {

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
                              gpu::Resources& res, const EngineConfig& ec,
                              LoadStats& st) {
  LoadedBlocks out;

  // --- framing + envelope + descriptor --------------------------------------
  Timer t;
  const json envelope = read_envelope(model_file);
  if (!envelope.is_object()) fail("envelope must be an object");
  const json& alice = envelope.at("Alice");
  if (!alice.is_object()) fail("envelope.Alice must be an object");
  if (alice.at("format_version").get<uint64_t>() != 1)
    fail("envelope.Alice: unsupported format_version (this loader reads 1)");
  if (envelope.contains("signature") && !envelope.at("signature").is_null()) {
    std::string key = "?", alg = "?";
    const json& s = envelope.at("signature");
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
  std::map<std::string, std::unique_ptr<block::BlockFactory>> resolved;
  std::string iface;
  for (const auto& [name, decl] : out.descriptor.specs) {
    const std::string where = "block_specs." + name;
    const auto it = block::registry().find(decl.type);
    if (it == block::registry().end())
      fail(where + ": unknown block type '" + decl.type + "'");
    std::unique_ptr<block::BlockFactory> rs;
    try {
      rs = it->second(decl.version, decl.config);
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
  size_t scratch_bytes = 0;
  for (const auto& [_, rs] : resolved)
    scratch_bytes = std::max(scratch_bytes, rs->scratch_bytes(ec));
  res.init_scratch(scratch_bytes);
  st.scratch_bytes = scratch_bytes;

  // --- the block loop: single pass over the composition — per block,
  // checks (spec ref, codec) then read -> hash -> construct. A bad
  // record at block i fails after blocks < i are loaded: accepted
  // (files are machine-packed; RAII teardown handles the partial state).
  const std::filesystem::path base = model_file.parent_path();
  // The fused .model file's unsigned `files` array is the default location
  // source; the DataLoader resolves content-IDs to verified bytes (fetch
  // verifies BLAKE3==id). "Where bytes come from" is overridable at runtime
  // in principle; for now the loader uses this in-file default.
  const loader::LocalFileDataLoader data_loader(envelope.at("files"),
                                                 base.string());
  for (size_t i = 0; i < out.descriptor.blocks.size(); ++i) {
    const BlockEntry& e = out.descriptor.blocks[i];
    const std::string where = "blocks[" + std::to_string(i) + "]";
    const auto rit = resolved.find(e.spec);
    if (rit == resolved.end())
      fail(where + ": references unknown spec '" + e.spec + "'");
    const block::BlockFactory* rs = rit->second.get();
    // Location (file/offset/size/codec) lives in the `files` table, validated
    // by the DataLoader at construction; the factory-side reconciliation
    // (decoded length == derived) still runs in construct.
    data_loader.fetch(e.id, [&](std::span<const std::byte> data) {
      Timer tc;
      out.blocks.push_back(rs->construct(data, res, ec));
      st.construct_ms += tc.ms();
    });
  }
  st.read_ms = data_loader.read_ms();
  st.hash_ms = data_loader.hash_ms();
  st.bytes_read = data_loader.bytes_read();
  st.hash_verifications = data_loader.hash_verifications();
  st.blocks = out.blocks.size();
  return out;
}

}  // namespace

LoadedBlocks load_blocks(const std::filesystem::path& model_file,
                         gpu::Resources& res, const EngineConfig& ec,
                         LoadStats* out_stats) {
  LoadStats dummy;
  try {
    return load_blocks_impl(model_file, res, ec, out_stats ? *out_stats : dummy);
  } catch (const json::exception& e) {
    fail(std::string("descriptor: ") + e.what());
  }
}

}  // namespace alice::modelfmt
