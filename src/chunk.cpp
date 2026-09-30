#include "chunk.hpp"

#include <sys/mman.h>
#include <cstring>
#include <new>

namespace alice {

Chunk::Chunk(size_t bytes) : size_(bytes) {
  if (bytes) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
    (void)madvise(p, bytes, MADV_HUGEPAGE);  // THP staging (design §9)
    data_ = static_cast<uint8_t*>(p);
  }
}

Chunk::~Chunk() {
  if (data_) (void)munmap(data_, size_);
}

Chunk::Chunk(Chunk&& o) noexcept : data_(o.data_), size_(o.size_) {
  o.data_ = nullptr;
  o.size_ = 0;
}

Chunk& Chunk::operator=(Chunk&& o) noexcept {
  if (this != &o) {
    if (data_) (void)munmap(data_, size_);
    data_ = o.data_;
    size_ = o.size_;
    o.data_ = nullptr;
    o.size_ = 0;
  }
  return *this;
}

}  // namespace alice
