#include "alice/safetensors.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace alice::safetensors {

namespace {

constexpr size_t kMaxHeaderBytes = 256ull << 20;  // sanity cap on the JSON header

void read_exact(int fd, void* buf, size_t n, off_t off, const std::string& what) {
  auto* p = static_cast<std::byte*>(buf);
  size_t got = 0;
  while (got < n) {
    ssize_t r = ::pread(fd, p + got, n - got, off + static_cast<off_t>(got));
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("safetensors: pread failed on " + what + ": " +
                               std::strerror(errno));
    }
    if (r == 0) throw std::runtime_error("safetensors: unexpected EOF reading " + what);
    got += static_cast<size_t>(r);
  }
}

DType parse_dtype(const std::string& s, const std::string& name) {
  if (s == "BF16") return DType::BF16;
  if (s == "F32") return DType::F32;
  throw std::runtime_error("safetensors: tensor '" + name + "' has unsupported dtype '" +
                           s + "' (supported: BF16, F32)");
}

}  // namespace

File::~File() {
  if (fd_ >= 0) ::close(fd_);
}

File::File(File&& other) noexcept
    : fd_(other.fd_), tensors_(std::move(other.tensors_)), index_(std::move(other.index_)) {
  other.fd_ = -1;
}

File& File::operator=(File&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = other.fd_;
    tensors_ = std::move(other.tensors_);
    index_ = std::move(other.index_);
    other.fd_ = -1;
  }
  return *this;
}

File File::open(const std::string& path) {
  File f;
  f.fd_ = ::open(path.c_str(), O_RDONLY);
  if (f.fd_ < 0)
    throw std::runtime_error("safetensors: cannot open " + path + ": " + std::strerror(errno));

  try {
    uint8_t lenbuf[8];
    read_exact(f.fd_, lenbuf, 8, 0, path + " (header length)");
    uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) header_len |= uint64_t(lenbuf[i]) << (8 * i);
    if (header_len == 0 || header_len > kMaxHeaderBytes)
      throw std::runtime_error("safetensors: " + path + ": implausible header length " +
                               std::to_string(header_len));

    std::string header(header_len, '\0');
    read_exact(f.fd_, header.data(), header_len, 8, path + " (header)");
    const size_t data_base = 8 + size_t(header_len);

    auto j = nlohmann::json::parse(header);  // throws on malformed JSON
    if (!j.is_object()) throw std::runtime_error("safetensors: " + path + ": header not an object");

    for (auto& [name, v] : j.items()) {
      if (name == "__metadata__") continue;
      if (!v.is_object() || !v.contains("dtype"))
        throw std::runtime_error("safetensors: " + path + ": malformed entry for '" + name + "'");

      TensorInfo t;
      t.name = name;
      t.dtype = parse_dtype(v.at("dtype").get<std::string>(), name);

      const auto& shape = v.at("shape");
      if (!shape.is_array() || shape.size() < 1 || shape.size() > 2)
        throw std::runtime_error("safetensors: tensor '" + name + "' has rank " +
                                 std::to_string(shape.size()) + " (rank 1-2 only)");
      if (shape.size() == 1) {
        t.rows = 1;
        t.cols = shape[0].get<uint32_t>();
      } else {
        t.rows = shape[0].get<uint32_t>();
        t.cols = shape[1].get<uint32_t>();
      }

      const auto& offs = v.at("data_offsets");
      size_t begin = offs.at(0).get<size_t>(), end = offs.at(1).get<size_t>();
      if (end < begin)
        throw std::runtime_error("safetensors: tensor '" + name + "': bad data_offsets");
      t.nbytes = end - begin;
      if (t.nbytes != t.numel() * dtype_size(t.dtype))
        throw std::runtime_error("safetensors: tensor '" + name + "': data size " +
                                 std::to_string(t.nbytes) + " != shape*dtype " +
                                 std::to_string(t.numel() * dtype_size(t.dtype)));
      t.offset = data_base + begin;

      f.index_[t.name] = f.tensors_.size();
      f.tensors_.push_back(std::move(t));
    }
  } catch (const nlohmann::json::exception& e) {
    ::close(f.fd_);
    f.fd_ = -1;
    throw std::runtime_error("safetensors: " + path + ": " + e.what());
  } catch (...) {
    ::close(f.fd_);
    f.fd_ = -1;
    throw;
  }
  return f;
}

bool File::contains(const std::string& name) const { return index_.count(name) != 0; }

const TensorInfo& File::at(const std::string& name) const {
  auto it = index_.find(name);
  if (it == index_.end())
    throw std::out_of_range("safetensors: no tensor named '" + name + "'");
  return tensors_[it->second];
}

void File::read(const TensorInfo& info, void* dst) const {
  read_exact(fd_, dst, info.nbytes, static_cast<off_t>(info.offset), info.name);
}

}  // namespace alice::safetensors
