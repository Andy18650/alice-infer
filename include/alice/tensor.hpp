#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>

#include <sys/mman.h>  // anonymous mmap + THP-backed storage (design §9)

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
// Storage is anonymous mmap with MADV_HUGEPAGE (§9, revised 2026-09 after
// measurement): 2 MiB transparent huge pages make first-touch faults
// effectively free (256 faults per 500 MB vs 128k at 4 KiB), so pread
// faults pages in at ~8.6 GB/s. Measured cycle vs alternatives (500 MB,
// warm cache): THP fault-in-pread 58 ms; hipHostMalloc pin+H2D 455 ms (the
// pin/unpin itself costs ~1.3 GB/s); plain new[] 466 ms. Eager population
// (MAP_POPULATE/memset) loses — it only adds a zero-fill pass over pages
// the pread was going to write anyway. Not pinned: pageable H2D on this
// stack measures ~20 GB/s, so pinning buys ~8 ms per 500 MB at ~380 ms of
// pin/unpin cost.
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

  // Uninitialized [rows, cols] THP-backed storage; pages fault in on first
  // touch (the load path's pread).
  explicit Tensor(uint32_t rows, uint32_t cols) : rows_(rows), cols_(cols) {
    bytes_ = size_t(rows) * cols * sizeof(T);
    if (bytes_) {
      void* p = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (p == MAP_FAILED) throw std::bad_alloc();
      (void)madvise(p, bytes_, MADV_HUGEPAGE);  // system is in madvise mode
      data_ = static_cast<T*>(p);
    }
  }

  // Copying factory from existing packed host memory (tests, host fusion).
  static Tensor copy_of(const T* data, uint32_t rows, uint32_t cols) {
    Tensor t(rows, cols);
    if (t.numel()) std::memcpy(t.data_, data, t.nbytes());
    return t;
  }

  Tensor(Tensor&& o) noexcept
      : data_(o.data_), bytes_(o.bytes_), rows_(o.rows_), cols_(o.cols_) {
    o.data_ = nullptr;
    o.bytes_ = 0;
    o.rows_ = 0;
    o.cols_ = 0;
  }
  Tensor& operator=(Tensor&& o) noexcept {
    if (this != &o) {
      destroy();
      data_ = o.data_;
      bytes_ = o.bytes_;
      rows_ = o.rows_;
      cols_ = o.cols_;
      o.data_ = nullptr;
      o.bytes_ = 0;
      o.rows_ = 0;
      o.cols_ = 0;
    }
    return *this;
  }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  ~Tensor() { destroy(); }

  T* data() { return data_; }
  const T* data() const { return data_; }
  uint32_t rows() const { return rows_; }
  uint32_t cols() const { return cols_; }
  size_t numel() const { return size_t(rows_) * cols_; }
  size_t nbytes() const { return numel() * sizeof(T); }

 private:
  void destroy() noexcept {
    if (data_) (void)munmap(data_, bytes_);
  }

  T* data_ = nullptr;
  size_t bytes_ = 0;
  uint32_t rows_ = 0, cols_ = 0;
};

}  // namespace alice
