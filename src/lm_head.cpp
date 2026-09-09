#include "alice/lm_head.hpp"

#include <cstring>
#include <stdexcept>

#include "blocks/cpu_generic/kernels.hpp"

namespace alice {

LmHead::LmHead(TensorView weight) : vocab_(weight.rows), d_(weight.cols) {
  if (weight.dtype != DType::BF16 || weight.stride() != weight.cols)
    throw std::invalid_argument("LmHead: weight must be packed bf16 [vocab, d]");
  w_ = std::make_unique<std::byte[]>(weight.nbytes());
  std::memcpy(w_.get(), weight.data, weight.nbytes());
  logits_ = std::make_unique<float[]>(vocab_);
}

TensorView LmHead::project_last(const TensorView& io) {
  if (io.dtype != DType::F32 || io.cols != d_ || io.rows == 0)
    throw std::invalid_argument("LmHead: io must be fp32 [S, d], S >= 1");
  const float* last = io.as<float>() + size_t(io.rows - 1) * io.stride();
  const auto* w = reinterpret_cast<const uint16_t*>(w_.get());
  for (uint32_t o = 0; o < vocab_; ++o) {
    const uint16_t* w_row = w + size_t(o) * d_;
    float acc = 0.0f;
    for (uint32_t i = 0; i < d_; ++i) acc += last[i] * cpu_generic::bf16_to_f32(w_row[i]);
    logits_[o] = acc;
  }
  return TensorView{logits_.get(), DType::F32, 1, vocab_, 0};
}

}  // namespace alice
