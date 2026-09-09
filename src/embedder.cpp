#include "alice/embedder.hpp"

#include <cstring>
#include <stdexcept>

namespace alice {

Embedder::Embedder(BF16View table) : vocab_(table.rows), d_(table.cols) {
  if (table.stride() != table.cols)
    throw std::invalid_argument("Embedder: table must be packed [vocab, d]");
  table_ = std::make_unique<bf16[]>(table.numel());
  std::memcpy(table_.get(), table.data, table.nbytes());
}

void Embedder::embed(std::span<const int32_t> ids, F32View out) const {
  if (out.cols != d_ || out.rows != ids.size())
    throw std::invalid_argument("Embedder: out must be [S, d] with S == ids.size()");
  for (size_t s = 0; s < ids.size(); ++s) {
    if (ids[s] < 0 || uint32_t(ids[s]) >= vocab_)
      throw std::out_of_range("Embedder: token id out of vocab");
    const bf16* src = table_.get() + size_t(ids[s]) * d_;
    float* dst = out.data + s * out.stride();
    for (uint32_t i = 0; i < d_; ++i) dst[i] = src[i].to_f32();
  }
}

}  // namespace alice
