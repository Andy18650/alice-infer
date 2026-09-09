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
  // max_prefill_chunk sizes the full-projection buffer.
  explicit LmHead(TensorView weight, uint32_t max_prefill_chunk = 1);

  // Projects ONLY the last row of io (design §5/§6). Returns a view of the
  // private logits buffer: fp32 [1, vocab]. `io` must be fp32 [S, d].
  TensorView project_last(const TensorView& io);

  // Projects ALL rows of io (test/probing path; full [S, vocab] projection
  // option of design §5). Returns fp32 [S, vocab] view of the private buffer.
  TensorView project(const TensorView& io);

  uint32_t vocab_size() const { return vocab_; }

 private:
  uint32_t vocab_ = 0, d_ = 0, mpc_ = 1;
  std::unique_ptr<std::byte[]> w_;       // private bf16 copy
  TensorView wv_;                        // view of w_ for the shared kernel
  std::unique_ptr<float[]> logits_;      // private [mpc * vocab] fp32
};

}  // namespace alice
