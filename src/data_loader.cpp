// PROTOTYPE (2026-09 design sessions — NOT frozen). See data_loader.hpp.

#include "data_loader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <blake3.h>

namespace alice::loader {

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("data_loader: " + what);
}

// RAII file descriptor (non-movable; one per fetch — at most one open at a
// time on this path).
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

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
  }
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

void LocalFileDataLoader::build(const nlohmann::json& files,
                                const std::string& base_dir) {
  namespace fs = std::filesystem;
  if (!files.is_array()) fail("location table must be a JSON array of records");
  for (const auto& e : files) {
    Entry ent;
    ent.file = (fs::path(base_dir) / e.at("file").get<std::string>()).string();
    ent.offset = e.at("offset").get<uint64_t>();
    ent.size = e.at("size").get<uint64_t>();
    ent.codec = e.value("codec", "raw");
    if (ent.codec != "raw")
      fail("codec '" + ent.codec + "' not supported (v1: raw only)");
    const std::string id = e.at("id").get<std::string>();
    if (!entries_.emplace(id, std::move(ent)).second)
      fail("duplicate id '" + id + "' in location table");
  }
  if (entries_.empty()) fail("location table has no entries");
}

LocalFileDataLoader::LocalFileDataLoader(const nlohmann::json& files,
                                         const std::string& base_dir) {
  build(files, base_dir);
}

LocalFileDataLoader::LocalFileDataLoader(const std::string& table_path) {
  namespace fs = std::filesystem;
  const fs::path p(table_path);
  std::ifstream in(p, std::ios::binary);
  if (!in) fail("cannot open table '" + table_path + "'");
  std::string text((std::istreambuf_iterator<char>(in)), {});
  nlohmann::json tbl;
  try {
    tbl = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& e) {
    fail(std::string("table JSON: ") + e.what());
  }
  build(tbl, p.parent_path().string());
}

LocalFileDataLoader::~LocalFileDataLoader() {
  if (buffer_ && buffer_size_) (void)::munmap(buffer_, buffer_size_);
}

void LocalFileDataLoader::ensure_buffer_(size_t size) const {
  if (buffer_size_ == size) return;
  if (buffer_) {
    (void)::munmap(buffer_, buffer_size_);
    buffer_ = nullptr;
    buffer_size_ = 0;
  }
  if (size) {
    void* m = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED)
      fail("staging mmap failed (size " + std::to_string(size) + ")");
    (void)::madvise(m, size, MADV_HUGEPAGE);  // design §9 THP staging
    buffer_ = static_cast<std::byte*>(m);
    buffer_size_ = size;
  }
}

void LocalFileDataLoader::fetch(
    const std::string& id,
    std::function<void(std::span<const std::byte>)> consume) const {
  const auto it = entries_.find(id);
  if (it == entries_.end()) fail("unknown id '" + id + "'");
  const Entry& e = it->second;

  std::unique_lock<std::mutex> lock(cache_mutex_);
  cache_cv_.wait(lock,
                 [&] { return cached_id_ == id || consumers_ == 0; });

  if (cached_id_ != id) {
    // We are the loader. Evict the previous id so no waiter matches while we
    // are overwriting the buffer.
    cached_id_.clear();
    ensure_buffer_(static_cast<size_t>(e.size));

    Timer total;
    try {
      Fd fd(e.file);
      Timer tr;
      size_t got = 0;
      while (got < e.size) {
        const ssize_t n = ::pread(fd.get(), buffer_ + got, e.size - got,
                                  off_t(e.offset + got));
        if (n <= 0)
          fail("short read on '" + e.file + "': " + std::strerror(errno));
        got += size_t(n);
      }
      {
        std::lock_guard<std::mutex> mlock(metrics_mutex_);
        read_ms_ += tr.ms();
      }
      bytes_read_.fetch_add(e.size);

      // The self-certifying guarantee: bytes are verified against the ID
      // before the caller ever sees them.
      Timer th;
      if (blake3_hex(buffer_, e.size) != id)
        fail("BLAKE3 mismatch for id '" + id +
             "' — content corrupt or tampered");
      {
        std::lock_guard<std::mutex> mlock(metrics_mutex_);
        hash_ms_ += th.ms();
      }
      hash_verifications_.fetch_add(1);

      cached_id_ = id;
      consumers_ = 1;

      std::fprintf(stderr,
                   "[data_loader] loaded %s: %.2f ms (read %.2f ms + hash "
                   "%.2f ms), %.2f MiB\n",
                   id.c_str(), total.ms(), tr.ms(), th.ms(),
                   double(e.size) / (1024.0 * 1024.0));

      cache_cv_.notify_all();
    } catch (...) {
      // Load/verify failed. Reset cache state so waiters can retry.
      cached_id_.clear();
      consumers_ = 0;
      cache_cv_.notify_all();
      throw;
    }
  } else {
    ++consumers_;
  }

  lock.unlock();

  consume(std::span<const std::byte>(buffer_, e.size));

  lock.lock();
  --consumers_;
  if (consumers_ == 0) {
    cached_id_.clear();
    cache_cv_.notify_all();
  }
  lock.unlock();
}

}  // namespace alice::loader
