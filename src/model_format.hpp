#pragma once

// The alice model format loader (docs/model-format.md RC1,
// docs/model-loader.md). This round: framing + envelope + descriptor
// parse, spec pre-resolution, the allocator, and the BLOCK loading loop.
// Non-block members (embedder/trunk/language_head/tokenizer) are parsed
// structurally and set aside. Signature verification is deferred with the
// chain of trust: the envelope's signature state is DECLARED loudly and
// the load proceeds.

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>  // config handoff is raw json (dependency
                              // policy, AGENTS 2026-09: industry-standard
                              // deps when appropriate; the CMake script is
                              // the source of truth for links)

#include "alice/engine_config.hpp"

namespace alice::block {
class Block;
}
namespace alice::gpu_rocm {
struct Context;
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

// BlockFactory (the loader's block-construction seam, the former
// "ResolvedSpec") lives in src/blocks/block_factory.hpp; the THP-backed
// staging buffer (Chunk) is a loader-private RAII helper in model_format.cpp,
// not a boundary type — construct() takes a non-owning std::span<const std::byte>.

// ---------------------------------------------------------------------------
// Allocator (docs/model-loader.md §5)
// ---------------------------------------------------------------------------

class DeviceAllocator {
 public:
  DeviceAllocator() = default;
  ~DeviceAllocator();
  DeviceAllocator(const DeviceAllocator&) = delete;
  DeviceAllocator& operator=(const DeviceAllocator&) = delete;

  // Named-buffer interner: the name IS the sharing contract. Same name +
  // same size -> same pointer; same name + different size -> loud error.
  void* named(const std::string& name, size_t bytes);

  // Scratch: requirement recorded per spec during pre-resolution; the max
  // is allocated ONCE, before the construction loop, and poison-filled
  // (NaN) so read-before-first-write is observable in tests.
  void plan_scratch(size_t bytes);
  void init_scratch();
  float* scratch_ptr() const { return scratch_; }
  size_t scratch_bytes() const { return scratch_bytes_; }

 private:
  std::map<std::string, std::pair<void*, size_t>> named_;
  size_t scratch_planned_ = 0;
  size_t scratch_bytes_ = 0;
  float* scratch_ = nullptr;
  bool scratch_initialized_ = false;
};

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

struct LoadedBlocks {
  Descriptor descriptor;  // members set aside, available to later rounds
  std::unique_ptr<DeviceAllocator> allocator;
  std::vector<std::unique_ptr<block::Block>> blocks;  // destroyed first
};

// Loads the block list of a .model descriptor file. Throws
// std::runtime_error loudly on any framing, parse, validation, hash, or
// length failure (message names the file/spec/block involved).
//
// out_stats, if non-null, receives timing/throughput breakdown for each
// phase (parse, read, hash, construct). Null Object pattern: the loader
// writes to a local dummy if null, so internal code has no per-site
// branches.
LoadedBlocks load_blocks(const std::filesystem::path& model_file,
                         const gpu_rocm::Context& ctx, const EngineConfig& ec,
                         LoadStats* out_stats = nullptr);

}  // namespace alice::modelfmt
