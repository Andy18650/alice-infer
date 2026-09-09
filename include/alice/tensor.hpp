#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace alice {

// Storage dtypes supported at load time. Compute dtype (fp32 accumulation)
// is a contract of the primitives, not of the storage. New storage dtypes
// (quantization) extend this enum; the manifest declares dtypes per weight.
enum class DType : uint8_t { BF16, F32 };

constexpr size_t dtype_size(DType t) {
  switch (t) {
    case DType::BF16: return 2;
    case DType::F32:  return 4;
  }
  throw std::invalid_argument("unknown DType");
}

// Non-owning, rank <= 2 view over row-major storage. Ownership lives in
// private Block buffers, load-staging buffers, and Trunk buffers — never
// in this type. Used at trust boundaries only (Block io contract,
// constructor weight hand-off, load-time validation); block internals are
// dtype-static and layout-private (see docs/engine-design.md §4, §7.1).
//
// Deliberately NOT an N-D tensor: weights are matrices or vectors, and
// per-head structure in attention is a kernel parameter, not a layout.
// Vectors are represented as 1xN (safetensors rank-1 maps to rows==1).
struct TensorView {
  void*    data = nullptr;
  DType    dtype = DType::F32;
  uint32_t rows = 0;
  uint32_t cols = 0;
  size_t   ld   = 0;  // row stride in elements; 0 means packed (ld == cols)

  constexpr size_t stride() const { return ld ? ld : cols; }
  constexpr size_t numel() const { return size_t(rows) * cols; }
  // Bytes spanned in memory, accounting for stride padding.
  constexpr size_t nbytes() const {
    return rows == 0 ? 0 : (size_t(rows - 1) * stride() + cols) * dtype_size(dtype);
  }
  constexpr bool is_vector() const { return rows == 1 || cols == 1; }

  template <typename T> T* as() { return static_cast<T*>(data); }
  template <typename T> const T* as() const { return static_cast<const T*>(data); }

  // Row i as a 1 x cols view.
  TensorView row(uint32_t i) const {
    if (i >= rows) throw std::out_of_range("TensorView::row");
    return TensorView{static_cast<std::byte*>(data) + size_t(i) * stride() * dtype_size(dtype),
                      dtype, 1, cols, 0};
  }
};

}  // namespace alice
