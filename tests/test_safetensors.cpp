#include "alice/safetensors.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alice::DType;
using alice::safetensors::File;
using alice::safetensors::TensorInfo;

struct ToyTensor {
  std::string name;
  std::string dtype;            // safetensors spelling: "BF16" | "F32" | ...
  std::vector<uint32_t> shape;  // file-native rank
  std::vector<std::byte> data;
};

// Writes a well-formed (or deliberately malformed) safetensors file.
std::string write_file(const std::vector<ToyTensor>& ts, const std::string& tag) {
  std::string path = std::string(::testing::TempDir()) + "/alice_st_" + tag + ".safetensors";
  std::string json = "{";
  size_t off = 0;
  for (size_t i = 0; i < ts.size(); ++i) {
    const auto& t = ts[i];
    if (i) json += ",";
    json += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t d = 0; d < t.shape.size(); ++d) {
      if (d) json += ",";
      json += std::to_string(t.shape[d]);
    }
    json += "],\"data_offsets\":[" + std::to_string(off) + "," +
            std::to_string(off + t.data.size()) + "]}";
    off += t.data.size();
  }
  json += "}";

  std::ofstream out(path, std::ios::binary);
  uint64_t len = json.size();
  out.write(reinterpret_cast<const char*>(&len), 8);  // little-endian on x86
  out.write(json.data(), json.size());
  for (const auto& t : ts)
    out.write(reinterpret_cast<const char*>(t.data.data()), t.data.size());
  return path;
}

ToyTensor bf16_matrix(const std::string& name, uint32_t r, uint32_t c) {
  ToyTensor t{name, "BF16", {r, c}, {}};
  t.data.resize(size_t(r) * c * 2);
  for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = std::byte(i & 0xFF);
  return t;
}

TEST(Safetensors, MultiShardProbe) {
  auto a = write_file({bf16_matrix("a.weight", 2, 3)}, "ms_a");
  auto b = write_file({bf16_matrix("b.weight", 4, 2)}, "ms_b");
  auto fs = alice::safetensors::Files::open({a, b});
  EXPECT_EQ(fs.num_shards(), 2u);
  EXPECT_TRUE(fs.contains("a.weight"));
  EXPECT_TRUE(fs.contains("b.weight"));
  EXPECT_FALSE(fs.contains("c.weight"));
  EXPECT_EQ(fs.at("b.weight").rows, 4u);  // resolved from the second shard
  EXPECT_THROW((void)fs.at("c.weight"), std::out_of_range);
}

TEST(Safetensors, MultiShardDuplicateRejected) {
  auto a = write_file({bf16_matrix("dup.weight", 2, 3)}, "ms_da");
  auto b = write_file({bf16_matrix("dup.weight", 2, 3)}, "ms_db");
  try {
    auto fs = alice::safetensors::Files::open({a, b});
    FAIL() << "expected duplicate-across-shards to throw";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("dup.weight"), std::string::npos);
  }
}

TEST(Safetensors, MultiShardReadResolvesOwner) {
  const auto want = bf16_matrix("b.weight", 4, 2);
  auto a = write_file({bf16_matrix("a.weight", 2, 3)}, "ms_ra");
  auto b = write_file({want}, "ms_rb");
  auto fs = alice::safetensors::Files::open({a, b});
  const auto& info = fs.at("b.weight");
  std::vector<std::byte> buf(info.nbytes);
  fs.read(info, buf.data());
  EXPECT_EQ(buf, want.data);
}

TEST(Safetensors, RoundTrip) {
  auto path = write_file({bf16_matrix("a.weight", 2, 3)}, "roundtrip");
  auto f = File::open(path);
  ASSERT_EQ(f.tensors().size(), 1u);
  ASSERT_TRUE(f.contains("a.weight"));
  EXPECT_FALSE(f.contains("nope"));

  const auto& info = f.at("a.weight");
  EXPECT_EQ(info.dtype, DType::BF16);
  EXPECT_EQ(info.rows, 2u);
  EXPECT_EQ(info.cols, 3u);
  EXPECT_EQ(info.nbytes, 12u);

  std::vector<std::byte> buf(info.nbytes);
  f.read(info, buf.data());
  for (size_t i = 0; i < buf.size(); ++i) EXPECT_EQ(buf[i], std::byte(i & 0xFF));

  // the boundary check: right static type passes, wrong one is rejected
  File::check_as<alice::bf16>(info);
  EXPECT_THROW(File::check_as<float>(info), std::invalid_argument);
}

TEST(Safetensors, RankOneBecomesRowVector) {
  ToyTensor t{"norm.weight", "F32", {4}, {}};
  t.data.resize(4 * 4, std::byte(0));
  auto f = File::open(write_file({t}, "rank1"));
  const auto& info = f.at("norm.weight");
  EXPECT_EQ(info.rows, 1u);
  EXPECT_EQ(info.cols, 4u);
}

TEST(Safetensors, MultipleTensorsDistinctOffsets) {
  auto a = bf16_matrix("a", 2, 2);  // 8 bytes
  auto b = bf16_matrix("b", 4, 1);  // 8 bytes
  auto f = File::open(write_file({a, b}, "multi"));
  EXPECT_NE(f.at("a").offset, f.at("b").offset);
  std::vector<std::byte> buf(8);
  f.read(f.at("b"), buf.data());
  for (size_t i = 0; i < 8; ++i) EXPECT_EQ(buf[i], std::byte(i & 0xFF));
}

TEST(Safetensors, RejectsRank3) {
  ToyTensor t{"t", "BF16", {2, 2, 2}, {}};
  t.data.resize(16, std::byte(0));
  EXPECT_THROW(File::open(write_file({t}, "rank3")), std::runtime_error);
}

TEST(Safetensors, RejectsUnsupportedDtype) {
  ToyTensor t{"t", "F16", {2, 2}, {}};
  t.data.resize(8, std::byte(0));
  EXPECT_THROW(File::open(write_file({t}, "dtype")), std::runtime_error);
}

TEST(Safetensors, RejectsSizeMismatch) {
  ToyTensor t = bf16_matrix("t", 2, 3);
  t.data.resize(10);  // should be 12
  EXPECT_THROW(File::open(write_file({t}, "sizemis")), std::runtime_error);
}

TEST(Safetensors, RejectsMissingFile) {
  EXPECT_THROW(File::open("/nonexistent/model.safetensors"), std::runtime_error);
}

TEST(Safetensors, AtThrowsOnUnknown) {
  auto f = File::open(write_file({bf16_matrix("a", 1, 1)}, "unknown"));
  EXPECT_THROW(f.at("zzz"), std::out_of_range);
}

}  // namespace
