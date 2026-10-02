// The alice model format loader (docs/model-format.md RC1,
// docs/model-loader.md §8 scope). init_engine is the public entry: it
// reads the .model envelope (magic/framing/signature-state), extracts the
// signed metadata + non-device members (tokenizer, chat_template), and
// loads the device model via gpu::Model::load_from_format — unifying the
// model-format path and the manifest path on one runnable gpu::Model type.
// Signature verification is deferred with the chain of trust: the
// envelope's signature state is DECLARED loudly, not checked.

#include "model_format.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <span>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "blocks/block_factory.hpp"  // BlockFactory + registry() (block layer)
#include "data_loader.hpp"  // LocalFileDataLoader (content-addressed fetch)
#include "gpu/embedder.hpp"  // gpu::Embedder (member)
#include "gpu/gpu_model.hpp"  // gpu::Model (load_from_format + ~Engine)
#include "gpu/lm_head.hpp"  // gpu::LmHead (member)
#include "gpu/resources.hpp"  // gpu::Resources (load_from_format + handle for LmHead)
#include "gpu/sampler.hpp"  // gpu::Sampler (member)
#include "gpu/trunk.hpp"  // gpu::RmsNormAddTrunk (member)

namespace alice {

using nlohmann::json;

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

// Descriptor parses via from_json (header); json exceptions carry the key
// name (JSON_DIAGNOSTICS), converted once at the init_engine boundary. Only
// SEMANTIC checks hand-build context here (version, unknown type/codec,
// hash mismatches): things the json library cannot know.

// Reads a .model descriptor file and returns the parsed envelope: text
// region (everything before the first NUL, which must be \n- or \r\n-
// terminated; a binary tail may follow in single-file models, so no
// whole-file read), {"ALICE" magic, then the json parse. The raw text is
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
  if (got < 8 || text.compare(0, 8, "{\"ALICE\"") != 0)
    fail("bad magic: a .model file must begin with {\"ALICE\"");

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

// Verified bytes (BLAKE3==id) -> an owning Tensor<bf16> of the given shape;
// a size mismatch is a descriptor/packer contract violation (the ID already
// certified the bytes, so this only catches a config/id pairing bug).
Tensor<bf16> blob_to_tensor(std::span<const std::byte> blob, uint32_t rows,
                            uint32_t cols, const char* what) {
  const size_t want = size_t(rows) * cols * sizeof(bf16);
  if (blob.size() != want)
    fail(std::string(what) + ": blob size mismatch (expected " +
         std::to_string(want) + " = " + std::to_string(rows) + "x" +
         std::to_string(cols) + " bf16, got " + std::to_string(blob.size()) +
         ")");
  return Tensor<bf16>::copy_of(reinterpret_cast<const bf16*>(blob.data()),
                                rows, cols);
}

const modelfmt::MemberDecl& require_member(const modelfmt::Descriptor& d,
                                           const char* key) {
  const auto it = d.members.find(key);
  if (it == d.members.end())
    fail(std::string("descriptor missing required member '") + key + "'");
  return it->second;
}

}  // namespace

namespace gpu {

// Model::load_from_format: the model-format device load. Given a parsed
// Descriptor + the content-addressed bytes source, construct a Model IN
// PLACE (new Model() -> fill members), the same shape as the manifest
// path's Model::load but sourcing weights from the DataLoader. In-place
// construction avoids any copy/move of Model (which is non-copyable and,
// with a user-declared destructor, non-movable). json faults convert at
// the init_engine boundary.
std::unique_ptr<Model> Model::load_from_format(
    const modelfmt::Descriptor& descriptor,
    const loader::LocalFileDataLoader& data_loader, const EngineConfig& ec,
    modelfmt::LoadStats& st) {
  auto m = std::unique_ptr<Model>(new Model());  // private ctor; static member
  m->cfg_ = ec;  // forward() reads cfg_.max_prefill_chunk / max_seq_len
  m->resources_ = std::make_unique<Resources>(0);  // device 0 (single-GCD bring-up)
  Resources& res = *m->resources_;

  // --- spec pre-resolution + composition check: every spec resolves
  // against the registry (config faults carry spec context), and all
  // specs must present the same interface.
  std::map<std::string, std::unique_ptr<block::BlockFactory>> resolved;
  std::string iface;
  for (const auto& [name, decl] : descriptor.specs) {
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

  // --- arena + scratch: pre-allocate device VRAM before construction -----
  // Arena (one hipMalloc, bump-carved): sum of each block's device_bytes()
  // (weights + transients + state, padded to kDeviceAlign). Fit-checked
  // against hipMemGetInfo inside init_arena. Scratch (shared v4 prefill
  // partials) stays on its own alloc, per the clean buffer-type separation.
  uint64_t arena_bytes = 0;
  for (const auto& entry : descriptor.blocks)
    arena_bytes += resolved.at(entry.spec)->device_bytes(ec);
  res.init_arena(arena_bytes);

  size_t scratch_bytes = 0;
  for (const auto& [_, rs] : resolved)
    scratch_bytes = std::max(scratch_bytes, rs->scratch_bytes(ec));
  res.init_scratch(scratch_bytes);
  st.scratch_bytes = scratch_bytes;

  // --- the block loop: single pass over the composition — per block,
  // checks (spec ref, codec) then read -> hash -> construct. A bad
  // record at block i fails after blocks < i are loaded: accepted
  // (files are machine-packed; RAII teardown handles the partial state).
  // Location (file/offset/size/codec) lives in the `files` table, validated
  // by the DataLoader at construction; the factory-side reconciliation
  // (decoded length == derived) still runs in construct.
  for (size_t i = 0; i < descriptor.blocks.size(); ++i) {
    const modelfmt::BlockEntry& e = descriptor.blocks[i];
    const std::string where = "blocks[" + std::to_string(i) + "]";
    const auto rit = resolved.find(e.spec);
    if (rit == resolved.end())
      fail(where + ": references unknown spec '" + e.spec + "'");
    const block::BlockFactory* rs = rit->second.get();
    data_loader.fetch(e.id, [&](std::span<const std::byte> data) {
      Timer tc;
      m->blocks_.push_back(rs->construct(data, res, ec));
      st.construct_ms += tc.ms();
    });
  }

  // --- members: embedder, trunk, language_head, sampler ------------------
  // Weights come from the same content-addressed DataLoader (one fetch per
  // member; the ID is BLAKE3-verified on fetch). Each member is built with
  // its EXISTING constructor (own hipMalloc, unchanged) — the arena is the
  // block loop's only; members stay on direct hipMalloc per the bring-up
  // decision (small footprint, no churn to the soon-to-be-phased-out manifest
  // path). Tied lm_head shares the embedder's content-ID; the DataLoader
  // returns the same verified bytes for both (the ID is the single source of
  // truth), so no special case. LmHead borrows res.handle() (same contract
  // as the manifest path).
  {  // embedder: [vocab, d] bf16
    const modelfmt::MemberDecl& mm = require_member(descriptor, "embedder");
    const uint32_t vocab = mm.config.at("vocab_size").get<uint32_t>();
    const uint32_t d = mm.config.at("embedding_dim").get<uint32_t>();
    data_loader.fetch(mm.id, [&](std::span<const std::byte> blob) {
      Timer tc;
      m->embedder_ = std::make_unique<Embedder>(
          blob_to_tensor(blob, vocab, d, "embedder"), ec.max_prefill_chunk);
      st.construct_ms += tc.ms();
    });
  }
  {  // trunk: 2L pre_norms [1,d] (one per block, exec order) + final [1,d]
    const modelfmt::MemberDecl& mm = require_member(descriptor, "trunk");
    const uint32_t d = mm.config.at("hidden_dim").get<uint32_t>();
    const float eps = mm.config.at("epsilon").get<float>();
    const size_t npre = m->blocks_.size();
    data_loader.fetch(mm.id, [&](std::span<const std::byte> blob) {
      Timer tc;
      const size_t per = size_t(d) * sizeof(bf16);
      const size_t want = (npre + 1) * per;
      if (blob.size() != want)
        fail("trunk: blob size mismatch (expected " + std::to_string(want) +
             " = " + std::to_string(npre + 1) + "x" + std::to_string(d) +
             " bf16, got " + std::to_string(blob.size()) + ")");
      std::vector<Tensor<bf16>> pre_norms;
      pre_norms.reserve(npre);
      const auto* p = reinterpret_cast<const bf16*>(blob.data());
      for (size_t i = 0; i < npre; ++i)
        pre_norms.push_back(Tensor<bf16>::copy_of(p + i * d, 1, d));
      auto final_norm = Tensor<bf16>::copy_of(p + npre * d, 1, d);
      m->trunk_ = std::make_unique<RmsNormAddTrunk>(
          std::move(pre_norms), std::move(final_norm), eps, d,
          ec.max_prefill_chunk);
      st.construct_ms += tc.ms();
    });
  }
  {  // language_head: [vocab, d] bf16 (tied shares embedder's content-ID)
    const modelfmt::MemberDecl& mm = require_member(descriptor, "language_head");
    const uint32_t vocab = mm.config.at("vocab_size").get<uint32_t>();
    const uint32_t d = mm.config.at("hidden_dim").get<uint32_t>();
    data_loader.fetch(mm.id, [&](std::span<const std::byte> blob) {
      Timer tc;
      m->lm_head_ = std::make_unique<LmHead>(
          blob_to_tensor(blob, vocab, d, "language_head"),
          ec.max_prefill_chunk, res.handle());
      st.construct_ms += tc.ms();
    });
  }
  m->sampler_ = std::make_unique<Sampler>(ec.sampling);

  st.read_ms = data_loader.read_ms();
  st.hash_ms = data_loader.hash_ms();
  st.bytes_read = data_loader.bytes_read();
  st.hash_verifications = data_loader.hash_verifications();
  st.blocks = m->blocks_.size();
  return m;
}

}  // namespace gpu

namespace modelfmt {

// init_engine: the envelope half. Reads the .model file (magic/framing),
// validates + parses the descriptor, declares the signature state, builds
// the content-addressed DataLoader from the unsigned `files` table, fetches
// the non-device members (tokenizer, chat_template) + signed metadata, then
// hands the descriptor + DataLoader to Model::load_from_format for the
// device half.
Engine init_engine_impl(const std::filesystem::path& model_file,
                        const EngineConfig& ec, LoadStats& st) {
  Engine out;

  Timer t;
  const json envelope = read_envelope(model_file);
  if (!envelope.is_object()) fail("envelope must be an object");
  const json& alice = envelope.at("ALICE");
  if (!alice.is_object()) fail("envelope.ALICE must be an object");
  if (alice.at("format_version").get<uint64_t>() != 1)
    fail("envelope.ALICE: unsupported format_version (this loader reads 1)");
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
  Descriptor descriptor = model.get<Descriptor>();
  if (descriptor.version != 1)
    fail("descriptor: unsupported version " + std::to_string(descriptor.version) +
         " (this loader reads version 1)");
  st.parse_ms = t.ms();
  st.specs = descriptor.specs.size();

  // The bytes source: content-addressed DataLoader from the unsigned `files`
  // table (location separated from identity, per the format).
  const std::filesystem::path base = model_file.parent_path();
  const loader::LocalFileDataLoader data_loader(envelope.at("files"),
                                                 base.string());

  // Non-device members + signed metadata, retained for the session/REPL
  // layer (not part of the device forward path). Optional: absent -> empty.
  auto fetch_string = [&](const char* key) -> std::string {
    const auto it = descriptor.members.find(key);
    if (it == descriptor.members.end()) return {};
    std::string s;
    data_loader.fetch(it->second.id, [&](std::span<const std::byte> blob) {
      s.assign(reinterpret_cast<const char*>(blob.data()), blob.size());
    });
    return s;
  };
  out.tokenizer = fetch_string("tokenizer");
  out.chat_template = fetch_string("chat_template");

  // Token metadata from the tokenizer member config (the packer emits
  // bos_token_id + eos_token_id there). The REPL uses these for BOS/EOS.
  if (const auto it = descriptor.members.find("tokenizer");
      it != descriptor.members.end()) {
    const json& tc = it->second.config;
    if (tc.contains("bos_token_id"))
      out.bos_token_id = tc.at("bos_token_id").get<uint32_t>();
    if (tc.contains("eos_token_id")) {
      const json& e = tc.at("eos_token_id");
      if (e.is_array())
        for (const auto& x : e) out.eos_token_ids.push_back(x.get<uint32_t>());
      else
        out.eos_token_ids.push_back(e.get<uint32_t>());
    }
  }

  out.name = descriptor.name;
  out.producer = descriptor.producer;
  out.packager = descriptor.packager;
  out.license = descriptor.license;
  out.license_url = descriptor.license_url;
  out.license_text = descriptor.license_text;
  out.max_context = descriptor.max_context;

  out.model = gpu::Model::load_from_format(descriptor, data_loader, ec, st);
  return out;
}

Engine init_engine(const std::filesystem::path& model_file,
                   const EngineConfig& ec, LoadStats* out_stats) {
  LoadStats dummy;
  try {
    return init_engine_impl(model_file, ec, out_stats ? *out_stats : dummy);
  } catch (const json::exception& e) {
    fail(std::string("descriptor: ") + e.what());
  }
}

// Out-of-line: the unique_ptr<Model> deleter needs Model complete
// (gpu_model.hpp, included above). Defined here so model_format.hpp stays
// free of the vendor/internal includes Model drags in.
Engine::~Engine() = default;

}  // namespace modelfmt

}  // namespace alice
