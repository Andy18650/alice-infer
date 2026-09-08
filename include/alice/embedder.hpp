#pragma once

#include "alice/tensor.hpp"
#include <cstdint>
#include <span>

namespace alice {

// Token-id -> hidden-state lookup. Owns the embedding table. Single
// component per model; weight sharing with the lm_head is expressed via
// manifest aliases, not by this interface.
class Embedder {
public:
  virtual ~Embedder() = default;
  virtual void embed(std::span<const int32_t> token_ids,
                     TensorView out) const = 0;  // out: [S, d]
};

}  // namespace alice
