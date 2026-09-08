#pragma once

#include "alice/tensor.hpp"
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace alice {

class SafetensorsError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Read-only access to tensors stored in safetensors files.
// Tensors are zero-copy views into memory-mapped shards (bf16 stays bf16).
class TensorSource {
public:
  virtual ~TensorSource() = default;
  virtual bool contains(std::string_view name) const = 0;
  virtual TensorView get(std::string_view name) const = 0;  // throws SafetensorsError
  virtual std::vector<std::string> names() const = 0;       // sorted, metadata excluded
};

// One memory-mapped .safetensors file. POSIX mmap for now; the mapping
// lives as long as the object.
class SafetensorsFile : public TensorSource {
public:
  static std::unique_ptr<SafetensorsFile> open(const std::filesystem::path& path);

  bool contains(std::string_view name) const override;
  TensorView get(std::string_view name) const override;
  std::vector<std::string> names() const override;

  ~SafetensorsFile();
  SafetensorsFile(SafetensorsFile&&) noexcept;
  SafetensorsFile& operator=(SafetensorsFile&&) noexcept;
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;

private:
  struct Impl;
  SafetensorsFile();
  std::unique_ptr<Impl> impl_;
};

// All .safetensors shards in a directory, merged into one index.
// Duplicate tensor names across shards are a load error.
class ShardedTensorSource : public TensorSource {
public:
  static std::unique_ptr<ShardedTensorSource> open_dir(const std::filesystem::path& dir);

  bool contains(std::string_view name) const override;
  TensorView get(std::string_view name) const override;
  std::vector<std::string> names() const override;

private:
  std::vector<std::unique_ptr<SafetensorsFile>> shards_;
  std::map<std::string, size_t, std::less<>> route_;  // tensor name -> shard index
};

}  // namespace alice
