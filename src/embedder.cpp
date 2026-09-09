#include "alice/embedder.hpp"

#include <cstring>
#include <stdexcept>

#include "blocks/cpu_generic/kernels.hpp"

namespace alice {

Embedder::Embedder(TensorView table) : vocab_(table.rows), d_(table.cols) {
  if (table.dtype != DType::BF16 || table.stride() != table.cols)
    throw std::invalid_argument("Embedder: table must be packed bf16 [vocab, d]");
  table_ = std::make_unique<std::byte[]>(table.nbytes());
  std::memcpy(table_.get(), table.data, table.nbytes());
}

void Embedder::embed(std::span<const int32_t> ids, TensorView out) const {
  if (out.dtype != DType::F32 || out.cols != d_ || out.rows != ids.size())
    throw std::invalid_argument("Embedder: out must be fp32 [S, d] with S == ids.size()");
  const auto* t = reinterpret_cast<const uint16_t*>(table_.get());
  for (size_t s = 0; s < ids.size(); ++s) {
    if (ids[s] < 0 || uint32_t(ids[s]) >= vocab_)
      throw std::out_of_range("Embedder: token id out of vocab");
    const uint16_t* src = t + size_t(ids[s]) * d_;
    float* dst = out.as<float>() + s * out.stride();
    for (uint32_t i = 0; i < d_; ++i) dst[i] = cpu_generic::bf16_to_f32(src[i]);
  }
}

}  // namespace alice
