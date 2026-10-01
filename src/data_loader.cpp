// PROTOTYPE (2026-09 design sessions — NOT frozen). See data_loader.hpp.

#include "data_loader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <blake3.h>
#include <nlohmann/json.hpp>

namespace alice::loader {

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("data_loader: " + what);
}

// RAII file descriptor (mirrors the loader-private Fd in model_format.cpp;
// non-movable, one per fetch — at most one open at a time on this path).
class Fd {
 public:
  explicit Fd(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) fail("cannot open '" + path + "': " + std::strerror(errno));
  }
  ~Fd() {
    if (fd_ >= 0) (void)::close(fd_);
  }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  int get() const { return fd_; }

 private:
  int fd_ = -1;
};

}  // namespace

std::string blake3_hex(const std::byte* data, size_t len) {
  blake3_hasher h;
  blake3_hasher_init(&h);
  blake3_hasher_update(&h, data, len);
  uint8_t d[32];
  blake3_hasher_finalize(&h, d, sizeof(d));
  static const char* hex = "0123456789abcdef";
  std::string s(64, '0');
  for (size_t i = 0; i < 32; ++i) {
    s[2 * i] = hex[d[i] >> 4];
    s[2 * i + 1] = hex[d[i] & 0xF];
  }
  return s;
}

std::string blake3_hex(std::span<const std::byte> data) {
  return blake3_hex(data.data(), data.size());
}

LocalFileDataLoader::LocalFileDataLoader(const std::string& table_path) {
  namespace fs = std::filesystem;
  const fs::path p(table_path);
  const std::string table_dir = p.parent_path().string();

  std::ifstream in(p, std::ios::binary);
  if (!in) fail("cannot open table '" + table_path + "'");
  std::string text((std::istreambuf_iterator<char>(in)), {});

  nlohmann::json tbl;
  try {
    tbl = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& e) {
    fail(std::string("table JSON: ") + e.what());
  }
  if (!tbl.is_array()) fail("table must be a JSON array of records");

  uint64_t max_size = 0;
  for (const auto& e : tbl) {
    Entry ent;
    ent.file =
        (fs::path(table_dir) / e.at("file").get<std::string>()).string();
    ent.offset = e.at("offset").get<uint64_t>();
    ent.size = e.at("size").get<uint64_t>();
    ent.codec = e.value("codec", "raw");
    if (ent.codec != "raw")
      fail("codec '" + ent.codec + "' not supported (v1: raw only)");
    const std::string id = e.at("id").get<std::string>();
    if (!entries_.emplace(id, std::move(ent)).second)
      fail("duplicate id '" + id + "' in table");
    max_size = std::max(max_size, entries_.at(id).size);
  }
  if (entries_.empty()) fail("table has no entries");

  // One THP-backed staging buffer, max-entry-sized, reused across fetches.
  if (max_size) {
    void* m = mmap(nullptr, max_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) fail("staging mmap failed");
    (void)madvise(m, max_size, MADV_HUGEPAGE);  // design §9 THP staging
    buf_ = static_cast<std::byte*>(m);
    buf_cap_ = max_size;
  }
}

LocalFileDataLoader::~LocalFileDataLoader() {
  if (buf_) (void)munmap(buf_, buf_cap_);
}

void LocalFileDataLoader::fetch(
    const std::string& id,
    std::function<void(std::span<const std::byte>)> consume) const {
  const auto it = entries_.find(id);
  if (it == entries_.end()) fail("unknown id '" + id + "'");
  const Entry& e = it->second;
  if (e.size > buf_cap_)  // can't happen: buf_cap_ is the table max
    fail("entry size " + std::to_string(e.size) + " exceeds staging " +
         std::to_string(buf_cap_));

  Fd fd(e.file);
  size_t got = 0;
  while (got < e.size) {
    const ssize_t n =
        ::pread(fd.get(), buf_ + got, e.size - got, off_t(e.offset + got));
    if (n <= 0)
      fail("short read on '" + e.file + "': " + std::strerror(errno));
    got += size_t(n);
  }

  // The self-certifying guarantee: the bytes are verified against the ID
  // (which is their hash) before the caller ever sees them.
  if (blake3_hex(buf_, e.size) != id)
    fail("BLAKE3 mismatch for id '" + id + "' — content corrupt or tampered");

  consume(std::span<const std::byte>(buf_, e.size));
}

}  // namespace alice::loader
