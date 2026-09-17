#include "alice/tensor.hpp"

#include <cstring>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

using alice::bf16;
using alice::DType;
using alice::dtype_of;
using alice::dtype_size;
using alice::Tensor;

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

TEST(Tensor, GeometryAndAccess) {
  Tensor<float> t(2, 3);
  EXPECT_EQ(t.rows(), 2u);
  EXPECT_EQ(t.cols(), 3u);
  EXPECT_EQ(t.numel(), 6u);
  EXPECT_EQ(t.nbytes(), 24u);

  t.data()[0] = 1.0f;
  t.data()[5] = 2.0f;
  EXPECT_FLOAT_EQ(t.data()[0], 1.0f);
  EXPECT_FLOAT_EQ(t.data()[5], 2.0f);

  Tensor<bf16> bv(2, 2);
  bv.data()[0] = bf16::from_f32(1.0f);
  EXPECT_FLOAT_EQ(bv.data()[0].to_f32(), 1.0f);
  EXPECT_EQ(bv.nbytes(), 8u);
}

TEST(Tensor, MoveOnlySingleOwner) {
  static_assert(!std::is_copy_constructible_v<Tensor<float>>);
  static_assert(!std::is_copy_assignable_v<Tensor<float>>);
  static_assert(std::is_nothrow_move_constructible_v<Tensor<float>>);

  Tensor<float> a(2, 2);
  a.data()[0] = 42.0f;
  const float* pa = a.data();
  Tensor<float> b(std::move(a));
  EXPECT_EQ(b.data(), pa);  // ownership moved, buffer unchanged
  EXPECT_EQ(a.data(), nullptr);  // source is empty
  EXPECT_FLOAT_EQ(b.data()[0], 42.0f);

  Tensor<float> c;
  c = std::move(b);
  EXPECT_EQ(c.data(), pa);
  EXPECT_FLOAT_EQ(c.data()[0], 42.0f);

  Tensor<float> empty;
  EXPECT_EQ(empty.data(), nullptr);
  EXPECT_EQ(empty.numel(), 0u);
}

TEST(Tensor, CopyOfCopiesSource) {
  float src[6] = {0, 1, 2, 3, 4, 5};
  Tensor<float> t = Tensor<float>::copy_of(src, 2, 3);
  EXPECT_NE(t.data(), src);  // a copy, not an alias
  EXPECT_EQ(std::memcmp(t.data(), src, sizeof(src)), 0);
  t.data()[0] = -1.0f;
  EXPECT_EQ(src[0], 0.0f);  // source untouched
}

}  // namespace
