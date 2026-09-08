#include <gtest/gtest.h>

#include "alice/tensor.hpp"

using namespace alice;

TEST(DType, Sizes) {
  EXPECT_EQ(dtype_size(DType::BF16), 2u);
  EXPECT_EQ(dtype_size(DType::F32), 4u);
}

TEST(TensorView, Basics) {
  float buf[6] = {};
  TensorView v{buf, DType::F32, 2, 3, 0};
  EXPECT_EQ(v.numel(), 6u);
  EXPECT_EQ(v.nbytes(), 24u);
  EXPECT_EQ(v.stride(), 3u);
  EXPECT_FALSE(v.is_vector());
}

TEST(TensorView, RowView) {
  float buf[6] = {0, 1, 2, 3, 4, 5};
  TensorView v{buf, DType::F32, 2, 3, 0};
  TensorView r = v.row(1);
  EXPECT_EQ(r.rows, 1u);
  EXPECT_EQ(r.cols, 3u);
  EXPECT_TRUE(r.is_vector());
  EXPECT_EQ(r.as<const float>(), buf + 3);
  EXPECT_THROW(v.row(2), std::out_of_range);
}

TEST(TensorView, StridedNbytes) {
  float buf[12] = {};
  TensorView v{buf, DType::F32, 2, 3, 6};  // padded rows
  EXPECT_EQ(v.stride(), 6u);
  EXPECT_EQ(v.nbytes(), ((2u - 1u) * 6u + 3u) * 4u);
}
