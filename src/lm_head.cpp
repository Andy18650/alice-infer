#include "alice/lm_head.hpp"

#include <cstring>
#include <stdexcept>

#include "blocks/cpu_generic/kernels.hpp"

namespace alice {

LmHead::LmHead(TensorView weight, uint32_t max_prefill_chunk)
    : vocab_(weight.rows), d_(weight.cols), mpc_(max_prefill_chunk) {
  if (weight.dtype != DType::BF16 || weight.stride() != weight.cols)
    throw std::invalid_argument("LmHead: weight must be packed bf16 [vocab, d]");
  w_ = std::make_unique<std::byte[]>(weight.nbytes());
  std::memcpy(w_.get(), weight.data, weight.nbytes());
  wv_ = TensorView{w_.get(), DType::BF16, vocab_, d_, 0};
  logits_ = std::make_unique<float[]>(size_t(mpc_) * vocab_);
}

TensorView LmHead::project_last(const TensorView& io) {
  if (io.dtype != DType::F32 || io.cols != d_ || io.rows == 0)
    throw std::invalid_argument("LmHead: io must be fp32 [S, d], S >= 1");
  TensorView last{const_cast<float*>(io.as<float>()) + size_t(io.rows - 1) * io.stride(),
                  DType::F32, 1, d_, 0};  // read-only use; TensorView::data is non-const void*
  TensorView out{logits_.get(), DType::F32, 1, vocab_, 0};
  cpu_generic::matmul_wt(last, wv_, out);
  return out;
}

TensorView LmHead::project(const TensorView& io) {
  if (io.dtype != DType::F32 || io.cols != d_ || io.rows == 0 || io.rows > mpc_)
  throw std::invalid_argument("LmHead::project: io must be fp32 [S, d], 1 <= S <= mpc");
  TensorView out{logits_.get(), DType::F32, io.rows, vocab_, 0};
  cpu_generic::matmul_wt(io, wv_, out);
  return out;
}

}  // namespace alice
