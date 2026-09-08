#pragma once

#include "alice/tensor.hpp"
#include <memory>

namespace alice {

// Bump allocator for per-forward activations. Everything allocated here
// dies at the end of Model::forward; persistent state (KV cache) lives
// elsewhere. Single-threaded by design.
class ActivationArena {
public:
  explicit ActivationArena(size_t capacity_bytes);
  ~ActivationArena() = default;
  ActivationArena(const ActivationArena&) = delete;
  ActivationArena& operator=(const ActivationArena&) = delete;

  TensorView alloc(DType dtype, uint32_t rows, uint32_t cols);
  TensorView alloc_like(TensorView v) { return alloc(v.dtype, v.rows, v.cols); }

  // Reuse the whole arena (called at the start of each forward pass).
  void reset() { offset_ = 0; }

  size_t used() const { return offset_; }
  size_t capacity() const { return capacity_; }

private:
  static constexpr size_t kAlign = 64;
  std::unique_ptr<std::byte[]> buffer_;
  std::byte* base_ = nullptr;  // 64-byte aligned start inside buffer_
  size_t capacity_;
  size_t offset_ = 0;
};

}  // namespace alice
