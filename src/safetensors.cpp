#include "alice/safetensors.hpp"

#include "dtype_string.hpp"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace alice {

namespace {

// RAII for a read-only file mapping.
struct MappedFile {
  const std::byte* data = nullptr;
  size_t size = 0;

  ~MappedFile() {
    if (data) munmap(const_cast<std::byte*>(data), size);
  }
  MappedFile() = default;
  MappedFile(MappedFile&& o) noexcept : data(o.data), size(o.size) { o.data = nullptr; }
  MappedFile& operator=(MappedFile&& o) noexcept {
    if (this != &o) { this->~MappedFile(); data = o.data; size = o.size; o.data = nullptr; }
    return *this;
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
};

MappedFile map_file(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw SafetensorsError("cannot open " + path.string() + ": " + std::strerror(errno));
  struct stat st;
  if (fstat(fd, &st) != 0) {
    const int e = errno;
    close(fd);
    throw SafetensorsError("cannot stat " + path.string() + ": " + std::strerror(e));
  }
  const size_t size = size_t(st.st_size);
  void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  const int e = errno;
  close(fd);  // mapping survives
  if (p == MAP_FAILED)
    throw SafetensorsError("cannot mmap " + path.string() + ": " + std::strerror(e));
  MappedFile f;
  f.data = static_cast<const std::byte*>(p);
  f.size = size;
  return f;
}

constexpr uint64_t kMaxHeaderBytes = 256ull << 20;  // sanity cap; real headers are ~100 KB

uint64_t read_le64(const std::byte* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= uint64_t(uint8_t(p[i])) << (8 * i);
  return v;
}

}  // namespace

struct SafetensorsFile::Impl {
  MappedFile file;
  size_t data_begin = 0;  // absolute offset of the tensor data region
  struct Entry {
    DType dtype;
    uint32_t rows, cols;
    uint64_t offset;  // relative to data_begin
  };
  std::map<std::string, Entry, std::less<>> index;
};

SafetensorsFile::SafetensorsFile() = default;
SafetensorsFile::~SafetensorsFile() = default;
SafetensorsFile::SafetensorsFile(SafetensorsFile&&) noexcept = default;
SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&&) noexcept = default;

std::unique_ptr<SafetensorsFile> SafetensorsFile::open(const std::filesystem::path& path) {
  auto impl = std::make_unique<Impl>();
  impl->file = map_file(path);
  const std::string pname = path.filename().string();

  if (impl->file.size < 8)
    throw SafetensorsError(pname + ": file too small for a safetensors header");
  const uint64_t header_len = read_le64(impl->file.data);
  if (header_len > kMaxHeaderBytes)
    throw SafetensorsError(pname + ": implausible header length " + std::to_string(header_len));
  if (8 + header_len > impl->file.size)
    throw SafetensorsError(pname + ": truncated header");
  impl->data_begin = 8 + size_t(header_len);
  const uint64_t data_size = impl->file.size - impl->data_begin;

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(
        reinterpret_cast<const char*>(impl->file.data + 8),
        reinterpret_cast<const char*>(impl->file.data + 8 + header_len));
  } catch (const nlohmann::json::exception& e) {
    throw SafetensorsError(pname + ": bad header JSON: " + e.what());
  }
  if (!header.is_object())
    throw SafetensorsError(pname + ": header must be a JSON object");

  for (const auto& [name, meta] : header.items()) {
    if (name == "__metadata__") continue;
    try {
      const std::string dtype_str = meta.at("dtype").get<std::string>();
      DType dtype;
      if (!detail::dtype_from_string(dtype_str, dtype))
        throw SafetensorsError("unsupported dtype '" + dtype_str + "' (supported: BF16, F32)");

      const auto& shape = meta.at("shape");
      if (!shape.is_array() || shape.empty() || shape.size() > 2)
        throw SafetensorsError("rank must be 1 or 2 (engine is rank-2 capped by design)");
      const uint64_t d0 = shape.at(0).get<uint64_t>();
      const uint64_t d1 = shape.size() == 2 ? shape.at(1).get<uint64_t>() : 1;
      if (d0 > UINT32_MAX || d1 > UINT32_MAX)
        throw SafetensorsError("dimension too large");

      const auto& offs = meta.at("data_offsets");
      if (!offs.is_array() || offs.size() != 2)
        throw SafetensorsError("data_offsets must be [begin, end]");
      const uint64_t begin = offs.at(0).get<uint64_t>();
      const uint64_t end = offs.at(1).get<uint64_t>();
      if (end < begin || end > data_size)
        throw SafetensorsError("data_offsets out of file bounds");
      if (end - begin != d0 * d1 * dtype_size(dtype))
        throw SafetensorsError("data_offsets size does not match shape * dtype");

      Impl::Entry e;
      e.dtype = dtype;
      e.rows = shape.size() == 2 ? uint32_t(d0) : 1;       // rank-1 -> 1 x N vector
      e.cols = shape.size() == 2 ? uint32_t(d1) : uint32_t(d0);
      e.offset = begin;
      if (!impl->index.emplace(name, e).second)
        throw SafetensorsError("duplicate tensor name");
    } catch (const nlohmann::json::exception& e) {
      throw SafetensorsError(pname + ": tensor '" + name + "': " + e.what());
    } catch (const SafetensorsError& e) {
      throw SafetensorsError(pname + ": tensor '" + name + "': " + e.what());
    }
  }

  auto f = std::unique_ptr<SafetensorsFile>(new SafetensorsFile());
  f->impl_ = std::move(impl);
  return f;
}

bool SafetensorsFile::contains(std::string_view name) const {
  return impl_->index.find(name) != impl_->index.end();
}

TensorView SafetensorsFile::get(std::string_view name) const {
  auto it = impl_->index.find(name);
  if (it == impl_->index.end())
    throw SafetensorsError("tensor '" + std::string(name) + "' not found");
  const Impl::Entry& e = it->second;
  void* p = const_cast<std::byte*>(impl_->file.data) + impl_->data_begin + e.offset;
  return TensorView{p, e.dtype, e.rows, e.cols, 0};
}

std::vector<std::string> SafetensorsFile::names() const {
  std::vector<std::string> out;
  out.reserve(impl_->index.size());
  for (const auto& [name, _] : impl_->index) out.push_back(name);
  return out;
}

// --- ShardedTensorSource ---

std::unique_ptr<ShardedTensorSource> ShardedTensorSource::open_dir(
    const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> paths;
  for (const auto& de : std::filesystem::directory_iterator(dir))
    if (de.is_regular_file() && de.path().extension() == ".safetensors")
      paths.push_back(de.path());
  std::sort(paths.begin(), paths.end());
  if (paths.empty())
    throw SafetensorsError("no .safetensors files in " + dir.string());

  auto src = std::unique_ptr<ShardedTensorSource>(new ShardedTensorSource());
  for (const auto& p : paths) {
    auto shard = SafetensorsFile::open(p);
    for (const auto& name : shard->names()) {
      auto [_, inserted] =
          src->route_.emplace(name, src->shards_.size());
      if (!inserted)
        throw SafetensorsError("tensor '" + name + "' appears in more than one shard (" +
                               p.filename().string() + ")");
    }
    src->shards_.push_back(std::move(shard));
  }
  return src;
}

bool ShardedTensorSource::contains(std::string_view name) const {
  return route_.find(name) != route_.end();
}

TensorView ShardedTensorSource::get(std::string_view name) const {
  auto it = route_.find(name);
  if (it == route_.end())
    throw SafetensorsError("tensor '" + std::string(name) + "' not found in any shard");
  return shards_[it->second]->get(name);
}

std::vector<std::string> ShardedTensorSource::names() const {
  std::vector<std::string> out;
  out.reserve(route_.size());
  for (const auto& [name, _] : route_) out.push_back(name);
  return out;
}

}  // namespace alice
