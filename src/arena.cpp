#include "alice/arena.hpp"

#include <new>

namespace alice {

ActivationArena::ActivationArena(size_t capacity_bytes)
    : buffer_(std::make_unique<std::byte[]>(capacity_bytes + kAlign)),
      capacity_(capacity_bytes) {
  auto p = reinterpret_cast<uintptr_t>(buffer_.get());
  base_ = reinterpret_cast<std::byte*>((p + kAlign - 1) & ~uintptr_t(kAlign - 1));
}

TensorView ActivationArena::alloc(DType dtype, uint32_t rows, uint32_t cols) {
  const size_t bytes = size_t(rows) * cols * dtype_size(dtype);
  const size_t aligned = (offset_ + kAlign - 1) / kAlign * kAlign;
  if (aligned + bytes > capacity_) throw std::bad_alloc();
  offset_ = aligned + bytes;
  return TensorView{base_ + aligned, dtype, rows, cols, 0};
}

}  // namespace alice
