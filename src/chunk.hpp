#pragma once

// The chunk: one decoded block extent, THP-backed, moved into the block
// factory (docs/model-loader.md §2: the loader owns bytes; the factory
// borrows them out as tensor views and the chunk dies at factory return).
// Lives at the `alice` top level — it is a generic owning byte buffer
// shared by the loader and the block-factory seam, not a format concept.

#include <cstddef>
#include <cstdint>

namespace alice {

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

}  // namespace alice
