#pragma once

// PROTOTYPE (2026-09 design sessions — NOT frozen). A content-addressed data
// loader. The signed model descriptor references chunks by content-ID (the
// BLAKE3 of their bytes); a DataLoader resolves IDs to verified bytes. "Where
// bytes come from" is a runtime/EngineConfig concern, not a packaging one — a
// local file, a P2P swarm, or an HTTP mirror are all transports behind one
// fetch(id) contract.
//
// The contract: fetch(id) yields bytes whose BLAKE3 == id, or fails loudly.
// Verification lives in the loader (the interface's invariant), so no caller
// or transport impl can hand back unverified bytes — the ID doubles as the
// lookup key and the integrity check (single source of truth).
//
// This first cut is a concrete LocalFileDataLoader — no abstract transport
// base yet, per the project's "no abstract until a second implementation
// exists" rule (design §8.3); the interface gets extracted when a second
// transport (P2P/HTTP) is built. The location table is a separate JSON file
// today; merging it into the .model file as a gated unsigned section is a
// later format decision (the table is untrusted plumbing, verified by the
// content hash either way).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>

namespace alice::loader {

// BLAKE3 of a byte range as a 64-char lowercase hex string. This is the
// content-ID vocabulary and the location-table key form.
std::string blake3_hex(const std::byte* data, size_t len);
std::string blake3_hex(std::span<const std::byte> data);

// Local-file content-addressed loader. Built from a location-table file
// (provisional JSON: a top-level array of {id, file, offset, size, codec};
// `file` resolves relative to the table's directory). Owns one THP-backed
// staging buffer (sized to the max entry, reused across fetches) — the
// Chunk-as-boundary-type question dissolves: it is a private detail of this
// one transport impl.
class LocalFileDataLoader {
 public:
  explicit LocalFileDataLoader(const std::string& table_path);
  ~LocalFileDataLoader();
  LocalFileDataLoader(const LocalFileDataLoader&) = delete;
  LocalFileDataLoader& operator=(const LocalFileDataLoader&) = delete;

  // Fetch the bytes for `id`, verify BLAKE3(bytes)==id, then invoke `consume`
  // with a non-owning view while the staging buffer is alive. Throws
  // std::runtime_error on unknown id, IO error, or hash mismatch — so
  // `consume` only ever sees verified bytes (the self-certifying guarantee).
  // Not reentrant: the pooled buffer is reused across fetches.
  void fetch(const std::string& id,
             std::function<void(std::span<const std::byte>)> consume) const;

 private:
  struct Entry {
    std::string file;     // absolute (resolved from the table's directory)
    uint64_t offset = 0;
    uint64_t size = 0;
    std::string codec;    // "raw" (the only legal value in v1)
  };

  std::unordered_map<std::string, Entry> entries_;
  mutable std::byte* buf_ = nullptr;   // pooled THP staging (max-entry-sized)
  mutable size_t buf_cap_ = 0;
};

}  // namespace alice::loader
