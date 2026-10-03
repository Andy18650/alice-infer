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
// transport (P2P/HTTP) is built. The location table is the fused .model
// file's `files` array (unsigned, verified by the content hash); a separate
// table file is also accepted (used by tests).
//
// The loader keeps a single shared THP-backed staging chunk. One thread loads
// a record at a time; once verified, other threads needing the same id share
// the buffer and consume concurrently. The chunk is evicted once the last
// consumer finishes.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>  // location-table ctor carries json; internal header

namespace alice::loader {

// BLAKE3 of a byte range as a 64-char lowercase hex string. This is the
// content-ID vocabulary and the location-table key form.
std::string blake3_hex(const std::byte* data, size_t len);
std::string blake3_hex(std::span<const std::byte> data);

// Local-file content-addressed loader. A location table is a JSON array of
// records {id, file, offset, size, codec}; `file` resolves relative to the
// table's base directory.
class LocalFileDataLoader {
 public:
  // From a separate location-table file.
  explicit LocalFileDataLoader(const std::string& table_path);
  // From an already-parsed `files` array (e.g. the fused .model file's
  // `files` element), with `file` paths resolved relative to `base_dir`.
  LocalFileDataLoader(const nlohmann::json& files, const std::string& base_dir);
  ~LocalFileDataLoader();
  LocalFileDataLoader(const LocalFileDataLoader&) = delete;
  LocalFileDataLoader& operator=(const LocalFileDataLoader&) = delete;

  // Fetch the bytes for `id`, verify BLAKE3(bytes)==id, then invoke `consume`
  // with a non-owning view while the staging buffer is alive. Throws
  // std::runtime_error on unknown id, IO error, or hash mismatch — so
  // `consume` only ever sees verified bytes (the self-certifying guarantee).
  // Thread-safe: multiple threads may fetch concurrently; a record is read and
  // verified at most once, then shared among concurrent consumers.
  void fetch(const std::string& id,
             std::function<void(std::span<const std::byte>)> consume) const;

  // Transport-reported metrics, accumulated across fetches (provisional: a
  // P2P transport would report different quantities).
  double read_ms() const { return read_ms_; }
  double hash_ms() const { return hash_ms_; }
  uint64_t bytes_read() const { return bytes_read_; }
  size_t hash_verifications() const { return hash_verifications_; }

 private:
  struct Entry {
    std::string file;     // absolute (resolved from the table's base dir)
    uint64_t offset = 0;
    uint64_t size = 0;
    std::string codec;    // "raw" (the only legal value in v1)
  };

  void build(const nlohmann::json& files, const std::string& base_dir);
  void ensure_buffer_(size_t size) const;

  std::unordered_map<std::string, Entry> entries_;

  // Metrics; protected by metrics_mutex_.
  mutable std::mutex metrics_mutex_;
  mutable double read_ms_ = 0.0, hash_ms_ = 0.0;
  mutable std::atomic<uint64_t> bytes_read_{0};
  mutable std::atomic<size_t> hash_verifications_{0};

  // Shared single-chunk cache; protected by cache_mutex_.
  mutable std::mutex cache_mutex_;
  mutable std::condition_variable cache_cv_;
  mutable std::string cached_id_;
  mutable std::byte* buffer_ = nullptr;
  mutable size_t buffer_size_ = 0;
  mutable int consumers_ = 0;
};

}  // namespace alice::loader
