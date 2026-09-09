#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "alice/tensor.hpp"

namespace alice {

// Token ids -> FP32 vectors (design §7.3: thin concrete class; no
// dtype×hardware matrix until a second implementation exists).
class Embedder {
 public:
  // table: validated bf16 [vocab, d]; copied into a private buffer and
  // considered freed on return.
  explicit Embedder(TensorView table);

  // out: fp32 [S, d]; one gathered row per id. Throws on id >= vocab.
  void embed(std::span<const int32_t> ids, TensorView out) const;

  uint32_t vocab_size() const { return vocab_; }

 private:
  uint32_t vocab_ = 0, d_ = 0;
  std::unique_ptr<std::byte[]> table_;  // private bf16 copy
};

}  // namespace alice
