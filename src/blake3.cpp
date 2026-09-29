// BLAKE3 wrapper (system libblake3). See include/alice/blake3.hpp.

#include "alice/blake3.hpp"

#include <stdexcept>

namespace alice::blake3 {

Digest hash(const void* data, size_t len) {
  Hasher h;
  h.update(data, len);
  return h.finalize();
}

std::string to_hex(const Digest& d) {
  static const char* hex = "0123456789abcdef";
  std::string s(64, '0');
  for (size_t i = 0; i < 32; ++i) {
    s[2 * i] = hex[d[i] >> 4];
    s[2 * i + 1] = hex[d[i] & 0xF];
  }
  return s;
}

}  // namespace alice::blake3
