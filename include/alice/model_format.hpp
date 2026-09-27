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
#include <variant>
#include <vector>

#include "alice/engine_config.hpp"

namespace alice::gpu {
class Block;
}
namespace alice::gpu_rocm {
struct Context;
}

namespace alice::modelfmt {

// ---------------------------------------------------------------------------
// Config: flat map, type-checked accessors, read-tracking. "Fully consumed"
// is mechanical: after a resolve function runs, read set == key set or the
// load fails (docs/model-loader.md §3).
// ---------------------------------------------------------------------------

class BlockConfig {
 public:
  // Value vocabulary: scalars + integer lists (tokenizer eos ids etc.).
  // Block-type configs are consumed via the typed accessors; the
  // full-consumption check is the enforcement that a type read everything.
  using Value =
      std::variant<bool, int64_t, double, std::string, std::vector<uint64_t>>;
  void put(const std::string& key, Value v) {
    vals_[key] = std::move(v);
    keys_.insert(key);
  }

  // Required accessors: missing key or wrong type = loud error naming it.
  uint64_t get_u(const std::string& key) const;
  double get_f(const std::string& key) const;
  std::string get_s(const std::string& key) const;
  bool get_b(const std::string& key) const;
  std::vector<uint64_t> get_u_list(const std::string& key) const;
  // Optional variants: absent key -> default; wrong type still errors.
  uint64_t get_u_or(const std::string& key, uint64_t def) const;
  double get_f_or(const std::string& key, double def) const;
  std::string get_s_or(const std::string& key, std::string def) const;
  bool get_b_or(const std::string& key, bool def) const;

  bool has(const std::string& key) const { return vals_.count(key) != 0; }
  const std::set<std::string>& keys() const { return keys_; }
  const std::set<std::string>& read() const { return read_; }

 private:
  const Value& at(const std::string& key) const;

  std::map<std::string, Value> vals_;
  std::set<std::string> keys_;
  mutable std::set<std::string> read_;  // keys handed out by accessors
};

// ---------------------------------------------------------------------------
// Descriptor structures (the "model" map of the envelope)
// ---------------------------------------------------------------------------

struct Record {
  std::string file;   // relative to the descriptor file's directory
  uint64_t offset = 0;
  uint64_t size = 0;  // stored (post-codec) bytes
  std::string codec;  // "raw" is the only legal value in format v1
  std::array<uint8_t, 32> hash{};  // BLAKE3 over [offset, offset + size)
};

struct BlockSpecDecl {
  std::string type;     // registry name, e.g. "dense_ffn_bf16"
  uint32_t version = 0; // registry-owned; pins the layout function
  BlockConfig config;
};

// Members are parsed structurally this round (type/version/config/record),
// not interpreted.
struct MemberDecl {
  std::string type;
  uint32_t version = 0;
  BlockConfig config;
  Record record;
};

struct BlockEntry {
  std::string spec;  // key into block_specs
  Record record;
};

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

// ---------------------------------------------------------------------------
// The chunk: one decoded block extent, THP-backed, moved into the factory
// (docs/model-loader.md §2: the loader owns bytes; the factory borrows
// them out as tensor views and the chunk dies at factory return).
// ---------------------------------------------------------------------------

class Chunk {
 public:
  Chunk() = default;
  explicit Chunk(size_t bytes);  // mmap + MADV_HUGEPAGE (design §9)
  ~Chunk();
  Chunk(Chunk&& o) noexcept;
  Chunk& operator=(Chunk&& o) noexcept;
  Chunk(const Chunk&) = delete;
  Chunk& operator=(const Chunk&) = delete;

  uint8_t* data() { return data_; }
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// Registry: (type, version) -> resolved spec. The resolved spec IS the
// layout function's output: derived byte size, scratch requirement (pure
// function of config + engine knobs), and the construct step.
// ---------------------------------------------------------------------------

class DeviceAllocator;

struct ResolvedSpec {
  virtual ~ResolvedSpec() = default;
  virtual uint64_t derived_bytes() const = 0;
  virtual size_t scratch_bytes(const EngineConfig& ec) const = 0;
  // Composition invariant (docs/model-loader.md §1): specs may state the
  // interface they present to the stream (e.g. "io:2048"); empty = no
  // opinion. The loader checks all present signatures agree.
  virtual std::string interface_sig() const { return {}; }
  // Reconciles the chunk length against the derived length (exact), slices
  // tensor views, runs the block constructor (private H2D), returns the
  // block. The chunk is consumed; the caller frees it after return.
  virtual std::unique_ptr<gpu::Block> construct(Chunk chunk, DeviceAllocator& alloc,
                                                const gpu_rocm::Context& ctx,
                                                const EngineConfig& ec) const = 0;
};

using ResolveFn = std::unique_ptr<ResolvedSpec>(const BlockSpecDecl& decl);

// The built-in type registry. Unknown type in a file = loud load error.
const std::map<std::string, ResolveFn*>& registry();

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
  std::vector<std::unique_ptr<gpu::Block>> blocks;  // destroyed first
  LoadStats stats;
};

// Loads the block list of a .model descriptor file. Throws
// std::runtime_error loudly on any framing, parse, validation, hash, or
// length failure (message names the file/spec/block involved).
LoadedBlocks load_blocks(const std::filesystem::path& model_file,
                         const gpu_rocm::Context& ctx, const EngineConfig& ec);

}  // namespace alice::modelfmt
