// BLAKE3 portable reference implementation (unkeyed, 32-byte output).
// Algorithm: the BLAKE3 spec / official reference implementation. In-tree
// because the format needs it at load and no system package provides it;
// self-contained so the loader has zero crypto dependencies beyond this
// file. Structure follows the reference: a 1024-byte chunk state (64-byte
// block buffer, CV, block-compression counter), an incremental CV stack
// for the chunk tree, parent nodes merged pairwise on push and finalize.

#include "alice/blake3.hpp"

#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace alice::blake3 {
namespace {

constexpr uint32_t kIV[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                             0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};

constexpr uint32_t kChunkStart = 1u << 0;
constexpr uint32_t kChunkEnd = 1u << 1;
constexpr uint32_t kParent = 1u << 2;
constexpr uint32_t kRoot = 1u << 3;

constexpr size_t kChunkLen = 1024;
constexpr size_t kBlockLen = 64;

constexpr uint32_t kPerm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};

inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

inline uint32_t load_le32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

inline void g(uint32_t* s, size_t a, size_t b, size_t c, size_t d, uint32_t mx,
              uint32_t my) {
  s[a] = s[a] + s[b] + mx;
  s[d] = rotr(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];
  s[b] = rotr(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + my;
  s[d] = rotr(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];
  s[b] = rotr(s[b] ^ s[c], 7);
}

inline void words_from_le(const uint8_t* bytes, uint32_t out[16]) {
  for (int i = 0; i < 16; ++i) out[i] = load_le32(bytes + 4 * i);
}

// The compression function. The chaining value is the first 8 output
// words; the full 16 are the XOF/root output.
inline void compress(const uint32_t* cv, const uint32_t* m, uint64_t counter,
                     uint32_t block_len, uint32_t flags, uint32_t out[16]) {
  uint32_t s[16] = {
      cv[0],  cv[1],  cv[2],  cv[3],  cv[4],  cv[5],  cv[6],  cv[7],
      kIV[0], kIV[1], kIV[2], kIV[3],
      uint32_t(counter), uint32_t(counter >> 32), block_len, flags,
  };
  uint32_t msg[16];
  std::memcpy(msg, m, sizeof(msg));
  for (int round = 0; round < 7; ++round) {
    g(s, 0, 4, 8, 12, msg[0], msg[1]);    // column step
    g(s, 1, 5, 9, 13, msg[2], msg[3]);
    g(s, 2, 6, 10, 14, msg[4], msg[5]);
    g(s, 3, 7, 11, 15, msg[6], msg[7]);
    g(s, 0, 5, 10, 15, msg[8], msg[9]);   // diagonal step
    g(s, 1, 6, 11, 12, msg[10], msg[11]);
    g(s, 2, 7, 8, 13, msg[12], msg[13]);
    g(s, 3, 4, 9, 14, msg[14], msg[15]);
    uint32_t permuted[16];
    for (int i = 0; i < 16; ++i) permuted[i] = msg[kPerm[i]];
    std::memcpy(msg, permuted, sizeof(msg));
  }
  for (int i = 0; i < 8; ++i) {
    out[i] = s[i] ^ s[i + 8];
    out[i + 8] = s[i + 8] ^ cv[i];
  }
}

}  // namespace

Hasher::Hasher() { std::memcpy(key_, kIV, sizeof(key_)); }

size_t Hasher::chunk_len() const {
  return size_t(blocks_compressed_) * kBlockLen + block_len_;
}

void Hasher::update(const void* data, size_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  while (len > 0) {
    if (chunk_len() == kChunkLen) {
      // Close out the completed chunk: its (full) last block gets the
      // CHUNK_END compression; its CV enters the tree.
      uint32_t m[16];
      words_from_le(block_, m);
      const uint32_t flags =
          kChunkEnd | (blocks_compressed_ == 0 ? kChunkStart : 0u);
      uint32_t out[16];
      compress(key_, m, chunk_counter_, kBlockLen, flags, out);
      ++chunk_counter_;
      push_cv(out, chunk_counter_);
      std::memcpy(key_, kIV, sizeof(key_));  // fresh chunk
      blocks_compressed_ = 0;
      block_len_ = 0;
    } else if (block_len_ == kBlockLen) {
      // A full non-final block: compress lazily, exactly when more input
      // arrives (a block completed by the END of input stays in the
      // buffer and is compressed by finalize — reference semantics).
      uint32_t m[16];
      words_from_le(block_, m);
      const uint32_t flags = blocks_compressed_ == 0 ? kChunkStart : 0u;
      uint32_t out[16];
      compress(key_, m, chunk_counter_, kBlockLen, flags, out);
      std::memcpy(key_, out, sizeof(key_));
      ++blocks_compressed_;
      block_len_ = 0;
    }
    const size_t take = len < kBlockLen - block_len_ ? len : kBlockLen - block_len_;
    std::memcpy(block_ + block_len_, p, take);
    block_len_ += uint32_t(take);
    p += take;
    len -= take;
  }
}

void Hasher::push_cv(const uint32_t* new_cv, uint64_t total_chunks) {
  uint32_t cv[8];
  std::memcpy(cv, new_cv, sizeof(cv));
  while ((total_chunks & 1) == 0) {
    // parent(left = stack top, right = cv), keyed by the IV.
    uint32_t m[16];
    std::memcpy(m, cv_stack_[--cv_stack_len_], 32);
    std::memcpy(m + 8, cv, 32);
    uint32_t out[16];
    compress(kIV, m, 0, kBlockLen, kParent, out);
    std::memcpy(cv, out, sizeof(cv));
    total_chunks >>= 1;
  }
  std::memcpy(cv_stack_[cv_stack_len_++], cv, 32);
}

Digest Hasher::finalize() const {
  // The current chunk's output: CV (post compressed blocks), the
  // zero-padded last block as words, its length, and the chunk flags.
  uint32_t cv[8];
  std::memcpy(cv, key_, sizeof(cv));
  uint32_t block_words[16] = {};
  {
    uint8_t block[kBlockLen] = {};
    std::memcpy(block, block_, block_len_);
    words_from_le(block, block_words);
  }
  uint32_t block_len = block_len_;
  uint32_t flags = kChunkEnd | (blocks_compressed_ == 0 ? kChunkStart : 0u);
  uint64_t counter = chunk_counter_;  // chunk outputs carry their chunk id

  // Merge the CV stack right-to-left, reference Output chain style: each
  // step's right input is the chaining value of the current output, and
  // the result becomes a PARENT output (input CV = IV, counter = 0, block
  // = left||right).
  for (size_t i = cv_stack_len_; i-- > 0;) {
    uint32_t out[16];
    compress(cv, block_words, counter, block_len, flags, out);
    std::memcpy(cv, kIV, sizeof(cv));
    std::memcpy(block_words, cv_stack_[i], 32);
    std::memcpy(block_words + 8, out, 32);
    block_len = kBlockLen;
    flags = kParent;
    counter = 0;
  }

  uint32_t out[16];
  compress(cv, block_words, counter, block_len, flags | kRoot, out);
  Digest d;
  for (int i = 0; i < 8; ++i) {
    d[4 * i + 0] = uint8_t(out[i]);
    d[4 * i + 1] = uint8_t(out[i] >> 8);
    d[4 * i + 2] = uint8_t(out[i] >> 16);
    d[4 * i + 3] = uint8_t(out[i] >> 24);
  }
  return d;
}

Digest hash_parallel(const void* data, size_t len, unsigned nthreads_req) {
  const size_t nfull = len / kChunkLen;
  unsigned nthreads =
      nthreads_req ? nthreads_req : std::thread::hardware_concurrency();
  if (nthreads <= 1 || nfull < 2) return hash(data, len);

  // Sequential semantics: the last 1024-byte window is the "current
  // chunk" — never pushed; finalize compresses its final block with
  // CHUNK_END. Everything before it contributes chunk CVs.
  const size_t pushed = nfull - 1;

  std::vector<std::array<uint32_t, 8>> cvs(pushed);
  const uint8_t* p = static_cast<const uint8_t*>(data);
  {
    const size_t nth = std::min<size_t>(nthreads, pushed);
    std::vector<std::thread> pool;
    pool.reserve(nth);
    const size_t per = (pushed + nth - 1) / nth;
    for (size_t t = 0; t < nth; ++t) {
      const size_t c0 = t * per;
      const size_t c1 = std::min(pushed, c0 + per);
      if (c0 >= c1) break;
      pool.emplace_back([&, c0, c1] {
        for (size_t c = c0; c < c1; ++c) {
          uint32_t key[8];
          std::memcpy(key, kIV, sizeof(key));
          const uint8_t* chunk = p + c * kChunkLen;
          for (int b = 0; b < 16; ++b) {
            uint32_t m[16], out[16];
            words_from_le(chunk + b * kBlockLen, m);
            const uint32_t flags = (b == 0 ? kChunkStart : 0u) |
                                   (b == 15 ? kChunkEnd : 0u);
            compress(key, m, c, kBlockLen, flags, out);
            std::memcpy(key, out, sizeof(key));
          }
          std::memcpy(cvs[c].data(), key, 32);
        }
      });
    }
    for (auto& t : pool) t.join();
  }

  // Replay the CV-stack merge (identical to the sequential push history),
  // then run the tail through a normal Hasher with its correct chunk id.
  // A full 1024-byte tail stays "current" per sequential semantics.
  Hasher h;
  for (size_t c = 0; c < pushed; ++c)
    h.push_cv(cvs[c].data(), c + 1);
  h.chunk_counter_ = pushed;
  const size_t tail_len = len - pushed * kChunkLen;
  h.update(p + pushed * kChunkLen, tail_len);
  return h.finalize();
}

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
