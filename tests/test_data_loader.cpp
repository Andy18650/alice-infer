// PROTOTYPE tests for the content-addressed DataLoader (host-only; no GPU).
// Exercises the self-certifying contract: a correct fetch yields verified
// bytes, an unknown id is rejected, and a tampered location (id pointing at
// the wrong bytes) is rejected by the hash check — not by trusting the table.

#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "../src/data_loader.hpp"

using alice::loader::blake3_hex;
using alice::loader::LocalFileDataLoader;

namespace {

std::vector<std::byte> make_blob(size_t n, uint32_t seed) {
  std::vector<std::byte> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = std::byte((i * 2654435761u + seed) >> 0);
  return v;
}

void write_file(const std::filesystem::path& p, const std::vector<std::byte>& bytes) {
  std::ofstream(p, std::ios::binary)
      .write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

// RAII temp directory; cleaned up on destruction (including on throw).
struct TempDir {
  std::filesystem::path p;
  TempDir() : p(std::filesystem::temp_directory_path() / "alice_dl_test") {
    std::filesystem::create_directories(p);
  }
  ~TempDir() { std::error_code ec; std::filesystem::remove_all(p, ec); }
  std::string table_path() const { return (p / "table.json").string(); }
  std::string data_path() const { return (p / "data.bin").string(); }
};

}  // namespace

TEST(LocalFileDataLoader, FetchYieldsVerifiedBytes) {
  TempDir tmp;
  const auto blob_a = make_blob(4096, 7);   // multi-page
  const auto blob_b = make_blob(1000, 99);  // sub-page, exercises offset
  // One data file with A then B laid out contiguously.
  {
    std::ofstream out(tmp.data_path(), std::ios::binary);
    out.write(reinterpret_cast<const char*>(blob_a.data()), std::streamsize(blob_a.size()));
    out.write(reinterpret_cast<const char*>(blob_b.data()), std::streamsize(blob_b.size()));
  }
  const std::string id_a = blake3_hex(blob_a);
  const std::string id_b = blake3_hex(blob_b);
  {
    nlohmann::json arr = nlohmann::json::array();
    arr.push_back({{"id", id_a}, {"file", "data.bin"}, {"offset", uint64_t(0)},
                   {"size", uint64_t(blob_a.size())}});
    arr.push_back({{"id", id_b}, {"file", "data.bin"}, {"offset", uint64_t(blob_a.size())},
                   {"size", uint64_t(blob_b.size())}});
    std::ofstream(tmp.table_path()) << arr.dump();
  }

  const LocalFileDataLoader loader(tmp.table_path());

  std::vector<std::byte> got_a, got_b;
  loader.fetch(id_a, [&](std::span<const std::byte> d) {
    got_a.assign(d.begin(), d.end());
  });
  loader.fetch(id_b, [&](std::span<const std::byte> d) {
    got_b.assign(d.begin(), d.end());
  });
  EXPECT_EQ(got_a, blob_a);
  EXPECT_EQ(got_b, blob_b);
}

TEST(LocalFileDataLoader, RejectsUnknownId) {
  TempDir tmp;
  const auto blob = make_blob(512, 1);
  write_file(tmp.data_path(), blob);
  nlohmann::json arr = nlohmann::json::array();
  arr.push_back({{"id", blake3_hex(blob)}, {"file", "data.bin"},
                 {"offset", uint64_t(0)}, {"size", uint64_t(blob.size())}});
  std::ofstream(tmp.table_path()) << arr.dump();

  const LocalFileDataLoader loader(tmp.table_path());
  const std::string bogus(64, '0');  // well-formed but not in the table
  EXPECT_THROW(loader.fetch(bogus, [](std::span<const std::byte>) {}), std::runtime_error);
}

TEST(LocalFileDataLoader, RejectsTamperedContent) {
  // The table claims id_b for bytes that are actually blob_a. The location is
  // "wrong", but integrity is enforced by the hash, not by trusting the
  // table — so fetch must reject (BLAKE3(blob_a) != id_b).
  TempDir tmp;
  const auto blob_a = make_blob(2048, 5);
  const auto blob_b = make_blob(2048, 9);  // same size, different bytes
  write_file(tmp.data_path(), blob_a);
  const std::string id_b = blake3_hex(blob_b);
  nlohmann::json arr = nlohmann::json::array();
  arr.push_back({{"id", id_b}, {"file", "data.bin"}, {"offset", uint64_t(0)},
                 {"size", uint64_t(blob_a.size())}});
  std::ofstream(tmp.table_path()) << arr.dump();

  const LocalFileDataLoader loader(tmp.table_path());
  EXPECT_THROW(loader.fetch(id_b, [](std::span<const std::byte>) {}), std::runtime_error);
}
