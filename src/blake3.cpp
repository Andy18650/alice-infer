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

Digest from_hex(std::string_view hex) {
  Digest d{};
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  if (hex.size() != 64)
    throw std::runtime_error("blake3: hash field must be 64 hex chars, got " +
                             std::to_string(hex.size()));
  for (size_t i = 0; i < 32; ++i) {
    const int hi = nib(hex[2 * i]), lo = nib(hex[2 * i + 1]);
    if (hi < 0 || lo < 0)
      throw std::runtime_error("blake3: invalid hex in hash field");
    d[i] = uint8_t((hi << 4) | lo);
  }
  return d;
}

}  // namespace alice::blake3
