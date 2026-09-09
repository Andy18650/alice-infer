#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "alice/tensor.hpp"

namespace alice::safetensors {

struct TensorInfo {
  std::string name;
  DType dtype = DType::F32;
  uint32_t rows = 0;
  uint32_t cols = 0;
  size_t offset = 0;  // absolute file offset of the tensor's data
  size_t nbytes = 0;

  size_t numel() const { return size_t(rows) * cols; }
};

// A single safetensors file, read with pread (design §9: no mmap anywhere;
// the loader stages tensors in RAM and constructors copy into private
// buffers). Multi-shard directory assembly is the ModelLoader's business.
//
// Enforced at open: header parses, dtype ∈ {BF16, F32}, rank ≤ 2 (rank-1
// loads as 1×N), data_offsets consistent with shape × dtype_size.
class File {
 public:
  File() = default;
  ~File();
  File(File&& other) noexcept;
  File& operator=(File&& other) noexcept;
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  // Throws std::runtime_error on any IO or format problem.
  static File open(const std::string& path);

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  bool contains(const std::string& name) const;
  const TensorInfo& at(const std::string& name) const;  // throws std::out_of_range

  // Reads the whole tensor into dst (must have room for info.nbytes).
  // Throws std::runtime_error on IO error.
  void read(const TensorInfo& info, void* dst) const;

  // The single checked runtime->static dtype conversion at the load
  // boundary: describes a staging buffer holding `info` as a TView<T>,
  // verifying the file dtype matches T. Throws std::invalid_argument.
  template <typename T>
  static TView<T> view_as(const TensorInfo& info, void* staging) {
    if (info.dtype != dtype_of<T>())
      throw std::invalid_argument("safetensors: view_as type mismatch on '" + info.name + "'");
    return TView<T>{static_cast<T*>(staging), info.rows, info.cols, 0};
  }

 private:
  int fd_ = -1;
  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string, size_t> index_;
};

}  // namespace alice::safetensors
