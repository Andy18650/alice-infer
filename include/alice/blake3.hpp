#pragma once

// BLAKE3 (unkeyed, 32-byte digest) — the alice model format's record hash
// (docs/model-format.md §4): BLAKE3 over exactly [offset, offset+size).
// Thin wrapper over the system libblake3 (SIMD implementations included).

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <blake3.h>

namespace alice::blake3 {

using Digest = std::array<uint8_t, 32>;

class Hasher {
 public:
  Hasher() { blake3_hasher_init(&h_); }
  void update(const void* data, size_t len) {
    blake3_hasher_update(&h_, data, len);
  }
  Digest finalize() const {  // hasher state is unchanged; reusable
    Digest d;
    blake3_hasher_finalize(&h_, d.data(), d.size());
    return d;
  }

 private:
  blake3_hasher h_;
};

Digest hash(const void* data, size_t len);
std::string to_hex(const Digest& d);

}  // namespace alice::blake3
