#pragma once

#include "alice/tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace alice {

class ActivationArena;

// Activation state threaded through the block loop. Weightless by
// construction: connection weights (hyper-connection mixing, attention-
// residual gather) belong to the Backbone. Shape is fixed per model at
// assembly time; the extension fields stay empty for a plain residual model.
struct Trunk {
  TensorView x;                     // single residual stream [S, d]
  std::vector<TensorView> streams;  // hyper-connection extension
  TensorView history;               // attention-residuals extension
};

// Owns the wiring scheme and all connection weights. Blocks never see the
// trunk; the scheme is a global property of the model and is expressed
// exactly once, here.
class Backbone {
public:
  virtual ~Backbone() = default;

  // Allocate the trunk for one forward pass of seq_len tokens.
  virtual Trunk make_trunk(ActivationArena& arena, uint32_t seq_len) const = 0;

  // Produce block i's input from the trunk (identity for "add").
  virtual TensorView read(const Trunk& trunk, size_t layer,
                          ActivationArena& scratch) const = 0;

  // Fold block i's output back into the trunk (x += out for "add").
  virtual void write(Trunk& trunk, size_t layer, TensorView block_out,
                     ActivationArena& scratch) const = 0;

  // The stream that feeds the lm_head after the last block.
  virtual TensorView output(const Trunk& trunk, ActivationArena& scratch) const = 0;
};

}  // namespace alice
