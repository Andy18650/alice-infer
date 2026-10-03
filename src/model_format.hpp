#pragma once

// The alice model format loader (docs/model-format.md RC1,
// docs/model-loader.md). Loads a full model: the block list AND the
// component members (embedder/trunk/language_head/sampler) from the
// descriptor's content-addressed member records. The chat_template is
// carried as text; the tokenizer is constructed so the engine owns the
// generation-side special-token metadata.
// Signature verification is deferred with the chain of trust: the
// envelope's signature state is DECLARED loudly and the load proceeds.

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>  // config handoff is raw json (dependency
                              // policy, AGENTS 2026-09: industry-standard
                              // deps when appropriate; the CMake script is
                              // the source of truth for links)

#include "alice/engine_config.hpp"
#include "frontend/tokenizer.hpp"

namespace alice {
class Sampler;  // host-side sampler (src/sampler.hpp)
}

namespace alice::block {
class Block;
}
namespace alice::frontend {
class Tokenizer;  // src/frontend/tokenizer.hpp
}
namespace alice::gpu {
class Resources;
class Embedder;
class RmsNormAddTrunk;
class LmHead;
class Model;  // the device-resident model (src/gpu/gpu_model.hpp)
}

namespace alice::modelfmt {

// ---------------------------------------------------------------------------
// Descriptor structures (the "model" map of the envelope)
// ---------------------------------------------------------------------------

// PROTOTYPE (fused JSON, 2026-09 — NOT frozen): location is separated
// from identity. A block/member record carries only a content-ID (the
// BLAKE3 of its bytes); the unsigned `files` array in the same JSON maps
// IDs to {file, offset, size, codec}. The ID doubles as the integrity
// check (verified on fetch).

struct BlockSpecDecl {
  std::string type;     // registry name, e.g. "dense_ffn_bf16"
  uint32_t version = 0; // registry-owned; pins the layout function
  nlohmann::json config;  // raw; each type parses its own (struct +
                          // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE in its
                          // resolve function). Unknown keys are ignored —
                          // type-key drift is a code-review concern, not
                          // a runtime checker's.
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BlockSpecDecl, type, version, config)

// Members are parsed structurally this round (type/version/config/record),
// not interpreted.
struct MemberDecl {
  std::string type;
  uint32_t version = 0;
  nlohmann::json config;
  std::string id;  // content-ID (BLAKE3 hex); location in the `files` table
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MemberDecl, type, version, config, id)

struct BlockEntry {
  std::string spec;  // key into block_specs
  std::string id;    // content-ID (BLAKE3 hex); location in the `files` table
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BlockEntry, spec, id)

struct Descriptor {
  std::string name;
  uint32_t version = 0;  // descriptor schema version; must be 1
  std::string producer, packager, license;  // informational, signed
  std::string license_url, license_text;    // optional
  uint64_t max_context = 0;

  std::map<std::string, MemberDecl> members;  // chat_template, tokenizer,
                                              // embedder, language_head, trunk
  std::map<std::string, BlockSpecDecl> specs;
  std::vector<BlockEntry> blocks;
};

// from_json for Descriptor: the field-extraction half of the former
// parse_descriptor, in nlohmann's from_json infrastructure (uniform with
// the macro-defined BlockSpecDecl/MemberDecl/BlockEntry). Optional fields
// use value(key, default); the members map collects whichever of the five
// member keys are present. The version!=1 semantic check stays in the
// loader (from_json is pure extraction, no validation).
inline void from_json(const nlohmann::json& m, Descriptor& d) {
  d.name = m.at("name").get<std::string>();
  d.version = m.at("version").get<uint32_t>();
  d.max_context = m.at("max_context").get<uint64_t>();
  d.producer = m.value("producer", "");
  d.packager = m.value("packager", "");
  d.license = m.value("license", "");
  d.license_url = m.value("license_url", "");
  d.license_text = m.value("license_text", "");
  for (const char* name :
       {"chat_template", "tokenizer", "embedder", "language_head", "trunk"}) {
    if (m.contains(name)) d.members[name] = m.at(name).get<MemberDecl>();
  }
  d.specs = m.at("block_specs").get<std::map<std::string, BlockSpecDecl>>();
  d.blocks = m.at("blocks").get<std::vector<BlockEntry>>();
}

// BlockFactory (the loader's block-construction seam, the former
// "ResolvedSpec") lives in src/blocks/block_factory.hpp; the THP-backed
// staging buffer (Chunk) is a loader-private RAII helper in model_format.cpp,
// not a boundary type — construct() takes a non-owning std::span<const std::byte>.

// ---------------------------------------------------------------------------
// The load entry point
// ---------------------------------------------------------------------------

struct LoadStats {
  size_t blocks = 0;
  size_t specs = 0;
  uint64_t bytes_read = 0;
  double parse_ms = 0, read_ms = 0, hash_ms = 0, construct_ms = 0;
  size_t scratch_bytes = 0;
  size_t hash_verifications = 0;
};

// A full loaded engine: signed metadata + the non-device members
// (tokenizer, chat_template) + the device-resident model. init_engine is
// the one public entry: it reads the .model envelope, extracts the
// metadata + non-device members, and loads the device model (blocks +
// embedder/trunk/lm_head/sampler) via Model::load_from_format. The
// Descriptor is an internal parse structure — consumed by the loader, not
// returned. Throws std::runtime_error loudly on any framing, parse,
// validation, hash, or length failure.
//
// Member construction uses each component's existing constructor (own
// hipMalloc, unchanged); the arena covers the block loop only. LmHead
// borrows the model's resources->handle() (same contract as the manifest
// path).
//
// out_stats, if non-null, receives timing/throughput breakdown for each
// phase (parse, read, hash, construct). Null Object pattern: the loader
// writes to a local dummy if null, so internal code has no per-site
// branches.
struct Engine {
  // Signed metadata (informational; retained for "what is this model?"
  // queries — no consumer yet, stored so they survive the load).
  std::string name, producer, packager, license, license_url, license_text;
  uint64_t max_context = 0;
  // Non-device members, fetched + retained for the session/REPL layer.
  frontend::Tokenizer tokenizer;  // constructed from the tokenizer.json member
  std::string chat_template;      // jinja template (content-addressed)
  std::unique_ptr<gpu::Model> model;  // device-resident; built by Model::load_from_format
  std::unique_ptr<Sampler> sampler;   // host-side sampling (moved here from gpu::Model)

  // Model is forward-declared above (exploded pimpl: this header stays free
  // of vendor/internal includes). The destructor is out-of-line so the
  // unique_ptr deleter sees Model complete without forcing every includer
  // to pull gpu_model.hpp. Move-only (RVO covers the return).
  ~Engine();
  Engine() = default;
  Engine(Engine&&) noexcept = default;
  Engine& operator=(Engine&&) noexcept = default;
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // --- generation API (moved here from repl.cpp) ---------------------------
  struct GenerateOptions {
    int max_new_tokens = 64;
    uint32_t max_seq_len = 8192;
    uint32_t max_prefill_chunk = 512;
  };

  struct GenerateResult {
    std::string text;          // decoded generated text (not including the prompt)
    size_t prompt_tokens = 0;  // prompt ids including BOS
    size_t generated_tokens = 0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
  };

  // Encode + add BOS, chunked prefill, then greedy decode up to max_new_tokens
  // or EOS. on_token is called with each newly available complete UTF-8 chunk
  // as decoding proceeds (may be empty if no callback is desired).
  GenerateResult generate(const std::string& prompt, const GenerateOptions& opts,
                          std::function<void(const std::string&)> on_token = {});

  // Low-level conveniences; the generation loop uses these internally.
  std::vector<int32_t> encode(const std::string& text) const;
  std::string decode(const std::vector<int32_t>& ids) const;
  std::vector<float> forward_logits(std::span<const int32_t> tokens);
  std::vector<float> forward_logits_all(std::span<const int32_t> tokens);
  int32_t forward(std::span<const int32_t> tokens);
  void reset();
};
Engine init_engine(const std::filesystem::path& model_file,
                   const EngineConfig& ec, LoadStats* out_stats = nullptr);

}  // namespace alice::modelfmt
