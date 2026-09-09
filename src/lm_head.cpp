#include "alice/lm_head.hpp"

#include <cstring>
#include <stdexcept>

#include "blocks/cpu_generic/kernels.hpp"

namespace alice {

LmHead::LmHead(BF16View weight, uint32_t max_prefill_chunk)
    : vocab_(weight.rows), d_(weight.cols), mpc_(max_prefill_chunk) {
  if (weight.stride() != weight.cols)
    throw std::invalid_argument("LmHead: weight must be packed [vocab, d]");
  w_ = std::make_unique<bf16[]>(weight.numel());
  std::memcpy(w_.get(), weight.data, weight.nbytes());
  wv_ = BF16View{w_.get(), vocab_, d_, 0};
  logits_ = std::make_unique<float[]>(size_t(mpc_) * vocab_);
}

F32View LmHead::project_last(F32View io) {
  if (io.cols != d_ || io.rows == 0)
    throw std::invalid_argument("LmHead: io must be [S, d], S >= 1");
  F32View out{logits_.get(), 1, vocab_, 0};
  cpu_generic::matmul_wt(io.row(io.rows - 1), wv_, out);
  return out;
}

F32View LmHead::project(F32View io) {
  if (io.cols != d_ || io.rows == 0 || io.rows > mpc_)
    throw std::invalid_argument("LmHead::project: io must be [S, d], 1 <= S <= mpc");
  F32View out{logits_.get(), io.rows, vocab_, 0};
  cpu_generic::matmul_wt(io, wv_, out);
  return out;
}

}  // namespace alice
