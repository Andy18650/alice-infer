#pragma once

#include <cstdint>
#include <memory>

#include "alice/tensor.hpp"

namespace alice {

// FP32 hidden -> FP32 logits (design §7.3: thin concrete class). NOT a Block:
// input/output sizes differ, so it deliberately doesn't fit the forward
// signature.
class LmHead {
 public:
  // weight: validated bf16 [vocab, d]; copied into a private buffer and
  // considered freed on return. NOTE: with tied embeddings this duplicates
  // the Embedder's table (one extra vocab×d bf16 buffer) — accepted for now,
  // flagged for the memory audit.
  explicit LmHead(TensorView weight);

  // Projects ONLY the last row of io (design §5/§6). Returns a view of the
  // private logits buffer: fp32 [1, vocab]. `io` must be fp32 [S, d].
  TensorView project_last(const TensorView& io);

  uint32_t vocab_size() const { return vocab_; }

 private:
  uint32_t vocab_ = 0, d_ = 0;
  std::unique_ptr<std::byte[]> w_;       // private bf16 copy
  std::unique_ptr<float[]> logits_;      // private [vocab] fp32
};

}  // namespace alice
