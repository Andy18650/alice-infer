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
// buffers). Multi-shard assembly is the Files class below.
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
  // boundary: verifies the file dtype matches T (the caller owns the
  // destination storage). Throws std::invalid_argument.
  template <typename T>
  static void check_as(const TensorInfo& info) {
    if (info.dtype != dtype_of<T>())
      throw std::invalid_argument("safetensors: check_as type mismatch on '" + info.name + "'");
  }

 private:
  int fd_ = -1;
  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string, size_t> index_;
};

// A multi-shard weight source: an ordered set of opened Files. Lookups probe
// the shards' RESIDENT header maps linearly — no centralized catalog is
// built or retained (design §8: each tensor is resolved exactly once at
// load, so a probe of <= N hash maps is negligible; HF's
// model.safetensors.index.json is deliberately ignored — the manifest is the
// contract and the two-way audit validates the source directly).
// Duplicate names ACROSS shards are rejected at open.
class Files {
 public:
  Files() = default;
  Files(Files&&) = default;
  Files& operator=(Files&&) = default;
  Files(const Files&) = delete;
  Files& operator=(const Files&) = delete;

  // Opens every shard; throws std::runtime_error on any IO/format problem
  // or on a tensor name claimed by two shards.
  static Files open(const std::vector<std::string>& paths);

  size_t num_shards() const { return files_.size(); }
  const std::vector<File>& shards() const { return files_; }

  bool contains(const std::string& name) const;
  const TensorInfo& at(const std::string& name) const;  // throws std::out_of_range

  // Reads the whole tensor into dst from whichever shard holds it.
  void read(const TensorInfo& info, void* dst) const;

  // The single checked runtime->static dtype conversion happens on the
  // per-shard TensorInfo — reuse File::check_as<T> unchanged.

 private:
  const File& owner_of(const std::string& name) const;  // throws out_of_range

  std::vector<File> files_;
  std::vector<std::string> paths_;  // for error messages
};

}  // namespace alice::safetensors
