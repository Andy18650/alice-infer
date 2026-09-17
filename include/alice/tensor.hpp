#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace alice {

// --- dtype vocabulary -------------------------------------------------------
//
// DType is the RUNTIME dtype vocabulary of FILE FORMATS (safetensors headers,
// manifest weight specs) — it describes bytes on disk. Inside the engine,
// dtype is static at type level (see Tensor below); the single checked
// conversion from runtime to static happens at the load boundary
// (safetensors::File::check_as<T>).

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

// --- typed tensor -----------------------------------------------------------

// Owning, rank <= 2, row-major, PACKED host tensor with the element type in
// the TYPE (no void*, no runtime dtype field, no as<T>()). Move-only:
// exactly one owner per buffer — the former TView/Staging pair merged
// (2026-09). Staged weights move into constructors and die there, so
// "staged views are freed on return" is mechanical, not a convention.
//
// Deliberately NOT strided: transposes are explicit decisions at GEMM call
// sites (rocBLAS op flags), never metadata — a stride field would only ever
// hide one. Deliberately NOT an N-D tensor: weights are matrices or
// vectors, and per-head structure in attention is a kernel parameter, not a
// layout. Vectors are 1xN (safetensors rank-1 maps to rows==1).
//
// This is the HOST-side vocabulary (loading, oracle compute, host glue).
// Device buffers are plain pointers, not Tensors; borrowed host access at
// compute boundaries is ptr + dims (mirroring the gpu::Block io contract).
// Quantized formats get concrete COMPOSITE types (data + scales) named by
// their abstract block class's constructor contract — not a polymorphic
// hierarchy (design §11).
template <typename T>
class Tensor {
 public:
  Tensor() = default;

  // Uninitialized [rows, cols] storage.
  explicit Tensor(uint32_t rows, uint32_t cols)
      : buf_(std::make_unique<T[]>(size_t(rows) * cols)), rows_(rows), cols_(cols) {}

  // Copying factory from existing packed host memory (tests, host fusion).
  static Tensor copy_of(const T* data, uint32_t rows, uint32_t cols) {
    Tensor t(rows, cols);
    if (t.numel()) std::memcpy(t.buf_.get(), data, t.nbytes());
    return t;
  }

  Tensor(Tensor&&) = default;
  Tensor& operator=(Tensor&&) = default;
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  T* data() { return buf_.get(); }
  const T* data() const { return buf_.get(); }
  uint32_t rows() const { return rows_; }
  uint32_t cols() const { return cols_; }
  size_t numel() const { return size_t(rows_) * cols_; }
  size_t nbytes() const { return numel() * sizeof(T); }

 private:
  std::unique_ptr<T[]> buf_;
  uint32_t rows_ = 0, cols_ = 0;
};

}  // namespace alice
