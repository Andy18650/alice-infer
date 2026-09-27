#pragma once

// BLAKE3 (unkeyed, 32-byte digest) — the alice model format's record hash
// (docs/model-format.md §4): BLAKE3 over exactly [offset, offset+size).
// Portable reference implementation, no SIMD; verified against the official
// test vectors (empty input) and cross-checked against the reference
// Python implementation by the model-format smoke harness. If hashing ever
// shows up in a load profile, the SIMD versions drop in behind this header.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace alice::blake3 {

using Digest = std::array<uint8_t, 32>;

class Hasher {
 public:
  Hasher();
  void update(const void* data, size_t len);
  Digest finalize() const;  // hasher state is unchanged; reusable

 private:
  size_t chunk_len() const;  // bytes accounted in the current chunk
  void push_cv(const uint32_t* new_cv, uint64_t total_chunks);

  uint32_t key_[8];         // chunk chaining value (IV at chunk start)
  uint8_t block_[64] = {};  // current block buffer
  uint32_t block_len_ = 0;
  uint32_t blocks_compressed_ = 0;  // whole blocks folded into key_ so far
  uint64_t chunk_counter_ = 0;      // id of the current chunk
  uint32_t cv_stack_[54][8] = {};
  size_t cv_stack_len_ = 0;

  friend Digest hash_parallel(const void* data, size_t len, unsigned nthreads);
};

Digest hash(const void* data, size_t len);
// Bit-identical to hash(); full 1024-byte chunks are hashed on independent
// threads (the scalar compressor alone runs ~56 MB/s, which would dominate
// multi-GB model loads). nthreads 0 = hardware_concurrency.
Digest hash_parallel(const void* data, size_t len, unsigned nthreads = 0);
std::string to_hex(const Digest& d);

// Parses exactly 64 hex chars; throws std::runtime_error loudly otherwise.
Digest from_hex(std::string_view hex);

}  // namespace alice::blake3
