#include "alice/tensor.hpp"

#include <cstring>

#include <gtest/gtest.h>

namespace {

using alice::bf16;
using alice::BF16View;
using alice::DType;
using alice::dtype_of;
using alice::dtype_size;
using alice::F32View;

TEST(Bf16, ConversionIsExactShift) {
  EXPECT_EQ(bf16::from_f32(1.0f).bits, 0x3F80);
  EXPECT_EQ(bf16::from_f32(-2.5f).bits, 0xC020);
  EXPECT_FLOAT_EQ(bf16(0x3F80).to_f32(), 1.0f);
  EXPECT_FLOAT_EQ(bf16(0xBF80).to_f32(), -1.0f);
  EXPECT_FLOAT_EQ(bf16(0x0000).to_f32(), 0.0f);
  // roundtrip for exactly-representable values
  for (const float f : {0.0f, 1.0f, -2.5f, 128.0f, 0.125f, -0.0078125f})
    EXPECT_FLOAT_EQ(bf16::from_f32(f).to_f32(), f);
  // explicit operator float
  EXPECT_FLOAT_EQ(float(bf16(0x3F80)), 1.0f);
}

TEST(Bf16, DtypeVocabulary) {
  EXPECT_EQ(dtype_size(DType::BF16), 2u);
  EXPECT_EQ(dtype_size(DType::F32), 4u);
  EXPECT_EQ(dtype_of<bf16>(), DType::BF16);
  EXPECT_EQ(dtype_of<float>(), DType::F32);
}

TEST(TView, Geometry) {
  float buf[6] = {};
  F32View v{buf, 2, 3, 0};
  EXPECT_EQ(v.stride(), 3u);
  EXPECT_EQ(v.numel(), 6u);
  EXPECT_EQ(v.nbytes(), 24u);
  EXPECT_FALSE(v.is_vector());

  F32View vec{buf, 1, 5, 0};
  EXPECT_TRUE(vec.is_vector());
}

TEST(TView, StridedNbytes) {
  float buf[16] = {};
  F32View v{buf, 3, 4, 6};  // ld = 6 > cols
  EXPECT_EQ(v.stride(), 6u);
  EXPECT_EQ(v.nbytes(), (2 * 6 + 4) * sizeof(float));
}

TEST(TView, Row) {
  float buf[6] = {0, 1, 2, 3, 4, 5};
  F32View v{buf, 2, 3, 0};
  const auto r1 = v.row(1);
  EXPECT_EQ(r1.rows, 1u);
  EXPECT_EQ(r1.cols, 3u);
  EXPECT_EQ(r1.data, buf + 3);
  EXPECT_THROW(v.row(2), std::out_of_range);
}

TEST(TView, TypedPointers) {
  bf16 w[4] = {};
  BF16View bv{w, 2, 2, 0};
  bv.data[0] = bf16::from_f32(1.0f);
  EXPECT_FLOAT_EQ(bv.data[0].to_f32(), 1.0f);
  EXPECT_EQ(bv.nbytes(), 8u);
}

}  // namespace
