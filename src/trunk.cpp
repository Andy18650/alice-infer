#include "alice/trunk.hpp"

#include <stdexcept>

#include "blocks/cpu_generic/kernels.hpp"

namespace alice {

RmsNormAddTrunk::RmsNormAddTrunk(std::vector<TensorView> pre_norms, TensorView final_norm,
                                 float eps, uint32_t hidden, uint32_t max_prefill_chunk)
    : d_(hidden), mpc_(max_prefill_chunk), eps_(eps) {
  residual_ = std::make_unique<float[]>(size_t(mpc_) * d_);
  io_ = std::make_unique<float[]>(size_t(mpc_) * d_);
  norms_.resize(pre_norms.size() * d_);
  final_.resize(d_);

  auto convert = [](const TensorView& src, float* dst, uint32_t d, const char* what) {
    if (src.dtype != DType::BF16 || src.numel() != d)
      throw std::invalid_argument(std::string("RmsNormAddTrunk: bad norm weight: ") + what);
    const auto* b = src.as<uint16_t>();
    for (uint32_t i = 0; i < d; ++i) dst[i] = cpu_generic::bf16_to_f32(b[i]);
  };
  for (size_t i = 0; i < pre_norms.size(); ++i)
    convert(pre_norms[i], norms_.data() + i * d_, d_, "pre_norm");
  convert(final_norm, final_.data(), d_, "final_norm");
}

TensorView RmsNormAddTrunk::begin_pass(uint32_t S) {
  if (S == 0 || S > mpc_)
    throw std::runtime_error("RmsNormAddTrunk: S out of range (max_prefill_chunk)");
  S_ = S;
  return TensorView{residual_.get(), DType::F32, S_, d_, 0};
}

TensorView RmsNormAddTrunk::io() { return TensorView{io_.get(), DType::F32, S_, d_, 0}; }

void RmsNormAddTrunk::read(size_t block) {
  if (block * d_ >= norms_.size()) throw std::out_of_range("RmsNormAddTrunk::read");
  auto residual = TensorView{residual_.get(), DType::F32, S_, d_, 0};
  auto io = TensorView{io_.get(), DType::F32, S_, d_, 0};
  cpu_generic::copy_rows(io, residual);
  cpu_generic::rmsnorm_inplace(io, norms_.data() + block * d_, eps_);
}

void RmsNormAddTrunk::write(size_t block) {
  if (block * d_ >= norms_.size()) throw std::out_of_range("RmsNormAddTrunk::write");
  auto residual = TensorView{residual_.get(), DType::F32, S_, d_, 0};
  auto io = TensorView{io_.get(), DType::F32, S_, d_, 0};
  cpu_generic::add_rows(residual, io);
}

void RmsNormAddTrunk::output() {
  auto residual = TensorView{residual_.get(), DType::F32, S_, d_, 0};
  auto io = TensorView{io_.get(), DType::F32, S_, d_, 0};
  cpu_generic::copy_rows(io, residual);
  cpu_generic::rmsnorm_inplace(io, final_.data(), eps_);
}

}  // namespace alice
