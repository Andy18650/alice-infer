#include "alice/trunk.hpp"

#include <stdexcept>
#include <string>

#include "blocks/cpu_generic/kernels.hpp"

namespace alice {

RmsNormAddTrunk::RmsNormAddTrunk(std::vector<BF16View> pre_norms, BF16View final_norm,
                                 float eps, uint32_t hidden, uint32_t max_prefill_chunk)
    : d_(hidden), mpc_(max_prefill_chunk), eps_(eps) {
  residual_ = std::make_unique<float[]>(size_t(mpc_) * d_);
  io_ = std::make_unique<float[]>(size_t(mpc_) * d_);
  norms_.resize(pre_norms.size() * d_);
  final_.resize(d_);

  auto convert = [](BF16View src, float* dst, uint32_t d, const char* what) {
    if (src.numel() != d)
      throw std::invalid_argument(std::string("RmsNormAddTrunk: bad norm weight: ") + what);
    for (uint32_t i = 0; i < d; ++i) dst[i] = src.data[i].to_f32();
  };
  for (size_t i = 0; i < pre_norms.size(); ++i)
    convert(pre_norms[i], norms_.data() + i * d_, d_, "pre_norm");
  convert(final_norm, final_.data(), d_, "final_norm");
}

F32View RmsNormAddTrunk::begin_pass(uint32_t S) {
  if (S == 0 || S > mpc_)
    throw std::runtime_error("RmsNormAddTrunk: S out of range (max_prefill_chunk)");
  S_ = S;
  return F32View{residual_.get(), S_, d_, 0};
}

F32View RmsNormAddTrunk::io() { return F32View{io_.get(), S_, d_, 0}; }

void RmsNormAddTrunk::read(size_t block) {
  if (block * d_ >= norms_.size()) throw std::out_of_range("RmsNormAddTrunk::read");
  cpu_generic::copy_rows(io(), F32View{residual_.get(), S_, d_, 0});
  auto io_view = io();
  cpu_generic::rmsnorm_inplace(io_view, norms_.data() + block * d_, eps_);
}

void RmsNormAddTrunk::write(size_t block) {
  if (block * d_ >= norms_.size()) throw std::out_of_range("RmsNormAddTrunk::write");
  cpu_generic::add_rows(F32View{residual_.get(), S_, d_, 0}, io());
}

void RmsNormAddTrunk::output() {
  cpu_generic::copy_rows(io(), F32View{residual_.get(), S_, d_, 0});
  auto io_view = io();
  cpu_generic::rmsnorm_inplace(io_view, final_.data(), eps_);
}

}  // namespace alice
