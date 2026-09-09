#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace alice {

// --- dtype vocabulary -------------------------------------------------------
//
// DType is the RUNTIME dtype vocabulary of FILE FORMATS (safetensors headers,
// manifest weight specs) — it describes bytes on disk. Inside the engine,
// dtype is static at type level (see TView below); the single checked
// conversion from runtime to static happens at the load boundary
// (safetensors::File::view_as<T>).

enum class DType : uint8_t { BF16, F32 };

constexpr size_t dtype_size(DType t) {
  switch (t) {
    case DType::BF16: return 2;
    case DType::F32:  return 4;
  }
  throw std::invalid_argument("unknown DType");
}

// bfloat16 storage element. Top 16 bits of fp32; widening conversion is
// exact and free (a shift). Explicit everywhere — no silent narrowing.
struct bf16 {
  uint16_t bits = 0;

  constexpr bf16() = default;
  explicit constexpr bf16(uint16_t raw) : bits(raw) {}

  // Truncating conversion (round-toward-zero mantissa); matches safetensors
  // producers for exactly-representable values and the test helpers.
  static bf16 from_f32(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return bf16(uint16_t(b >> 16));
  }

  float to_f32() const {
    const uint32_t b = uint32_t(bits) << 16;
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
  }
  explicit operator float() const { return to_f32(); }
};

// Maps a C++ element type to its file-format dtype (the boundary check).
template <typename T> constexpr DType dtype_of();
template <> constexpr DType dtype_of<bf16>() { return DType::BF16; }
template <> constexpr DType dtype_of<float>() { return DType::F32; }

// --- typed view -------------------------------------------------------------

// Non-owning, rank <= 2 view over row-major storage, with the element type
// in the TYPE (no void*, no runtime dtype field, no as<T>()).
//
// Deliberately NOT an N-D tensor: weights are matrices or vectors, and
// per-head structure in attention is a kernel parameter, not a layout.
// Vectors are represented as 1xN (safetensors rank-1 maps to rows==1).
//
// Used at the Block io contract (F32View), constructor weight hand-off
// (BF16View etc.), and component buffers. Quantized formats get concrete
// COMPOSITE types (e.g. struct { TView<uint8_t> data; TView<float> scales; })
// named by their abstract block class's constructor contract — not a
// polymorphic view hierarchy (design §11).
template <typename T>
struct TView {
  T* data = nullptr;
  uint32_t rows = 0;
  uint32_t cols = 0;
  size_t ld = 0;  // row stride in elements; 0 means packed (ld == cols)

  constexpr size_t stride() const { return ld ? ld : cols; }
  constexpr size_t numel() const { return size_t(rows) * cols; }
  // Bytes spanned in memory, accounting for stride padding.
  constexpr size_t nbytes() const {
    return rows == 0 ? 0 : (size_t(rows - 1) * stride() + cols) * sizeof(T);
  }
  constexpr bool is_vector() const { return rows == 1 || cols == 1; }

  // Row i as a 1 x cols view.
  TView row(uint32_t i) const {
    if (i >= rows) throw std::out_of_range("TView::row");
    return TView{data + size_t(i) * stride(), 1, cols, 0};
  }
};

using F32View = TView<float>;
using BF16View = TView<bf16>;

}  // namespace alice
