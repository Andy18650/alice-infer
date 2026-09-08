#include <gtest/gtest.h>

#include "alice/safetensors.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

using namespace alice;
namespace fs = std::filesystem;

namespace {

void write_shard(const fs::path& p, const std::string& header_json,
                 const std::vector<std::byte>& data) {
  std::ofstream f(p, std::ios::binary);
  const uint64_t n = header_json.size();
  for (int i = 0; i < 8; ++i) f.put(char((n >> (8 * i)) & 0xFF));
  f.write(header_json.data(), std::streamsize(header_json.size()));
  f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
}

template <typename T>
void append(std::vector<std::byte>& v, const T* ptr, size_t count) {
  const std::byte* b = reinterpret_cast<const std::byte*>(ptr);
  v.insert(v.end(), b, b + count * sizeof(T));
}

class SafetensorsTest : public ::testing::Test {
 protected:
  fs::path dir;
  void SetUp() override {
    dir = fs::temp_directory_path() /
          ("alice_st_" + std::to_string(::getpid()) + "_" +
           ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::create_directories(dir);
  }
  void TearDown() override { fs::remove_all(dir); }
};

}  // namespace

TEST_F(SafetensorsTest, LoadsTensorsZeroCopy) {
  const float w[6] = {1, 2, 3, 4, 5, 6};
  const uint16_t b[4] = {0x3F80, 0x4000, 0x4040, 0x4080};  // 1,2,3,4 in bf16
  std::vector<std::byte> data;
  append(data, w, 6);
  append(data, b, 4);
  const std::string hdr =
      R"({"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
      R"("bias":{"dtype":"BF16","shape":[4],"data_offsets":[24,32]}})";
  const auto path = dir / "a.safetensors";
  write_shard(path, hdr, data);

  auto f = SafetensorsFile::open(path);
  EXPECT_TRUE(f->contains("w"));
  EXPECT_TRUE(f->contains("bias"));
  EXPECT_FALSE(f->contains("zzz"));

  const TensorView wv = f->get("w");
  EXPECT_EQ(wv.dtype, DType::F32);
  EXPECT_EQ(wv.rows, 2u);
  EXPECT_EQ(wv.cols, 3u);
  EXPECT_FLOAT_EQ(wv.as<const float>()[5], 6.0f);

  const TensorView bv = f->get("bias");  // rank-1 -> 1 x N vector
  EXPECT_EQ(bv.dtype, DType::BF16);
  EXPECT_EQ(bv.rows, 1u);
  EXPECT_EQ(bv.cols, 4u);
  EXPECT_EQ(bv.as<const uint16_t>()[2], 0x4040);

  EXPECT_THROW(f->get("zzz"), SafetensorsError);
  const auto ns = f->names();
  EXPECT_EQ(ns, (std::vector<std::string>{"bias", "w"}));
}

TEST_F(SafetensorsTest, RejectsCorruptFiles) {
  const float w[6] = {};
  std::vector<std::byte> data;
  append(data, w, 6);

  // Truncated: header claims more bytes than the file holds.
  write_shard(dir / "t.safetensors", std::string(1000, ' '), {});
  EXPECT_THROW(SafetensorsFile::open(dir / "t.safetensors"), SafetensorsError);

  // Unsupported dtype.
  write_shard(dir / "d.safetensors",
              R"({"w":{"dtype":"F16","shape":[2,3],"data_offsets":[0,24]}})", data);
  EXPECT_THROW(SafetensorsFile::open(dir / "d.safetensors"), SafetensorsError);

  // Rank 3: outside the engine's rank-2 cap.
  write_shard(dir / "r.safetensors",
              R"({"w":{"dtype":"F32","shape":[1,2,3],"data_offsets":[0,24]}})", data);
  EXPECT_THROW(SafetensorsFile::open(dir / "r.safetensors"), SafetensorsError);

  // data_offsets past end of file.
  write_shard(dir / "o.safetensors",
              R"({"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,1024]}})", data);
  EXPECT_THROW(SafetensorsFile::open(dir / "o.safetensors"), SafetensorsError);

  // data_offsets inconsistent with shape * dtype.
  write_shard(dir / "s.safetensors",
              R"({"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,12]}})", data);
  EXPECT_THROW(SafetensorsFile::open(dir / "s.safetensors"), SafetensorsError);
}

TEST_F(SafetensorsTest, ShardedMerge) {
  const float a[2] = {1, 2};
  const float b[2] = {3, 4};
  std::vector<std::byte> da, db;
  append(da, a, 2);
  append(db, b, 2);
  write_shard(dir / "m-00001-of-00002.safetensors",
              R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", da);
  write_shard(dir / "m-00002-of-00002.safetensors",
              R"({"b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", db);

  auto s = ShardedTensorSource::open_dir(dir);
  EXPECT_TRUE(s->contains("a"));
  EXPECT_TRUE(s->contains("b"));
  EXPECT_EQ(s->get("b").as<const float>()[1], 4.0f);
  EXPECT_EQ(s->names(), (std::vector<std::string>{"a", "b"}));
}

TEST_F(SafetensorsTest, ShardedDuplicateIsAnError) {
  const float a[2] = {1, 2};
  std::vector<std::byte> da;
  append(da, a, 2);
  const std::string hdr = R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
  write_shard(dir / "m-00001-of-00002.safetensors", hdr, da);
  write_shard(dir / "m-00002-of-00002.safetensors", hdr, da);
  EXPECT_THROW(ShardedTensorSource::open_dir(dir), SafetensorsError);
}
