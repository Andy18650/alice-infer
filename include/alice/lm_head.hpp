#pragma once

#include "alice/tensor.hpp"

namespace alice {

// Final hidden state -> logits projection. Owns its weight (or shares the
// embedding table via a manifest alias).
class LmHead {
public:
  virtual ~LmHead() = default;
  virtual void project(TensorView trunk_out, TensorView logits) const = 0;  // [S,d] -> [S,V]
};

}  // namespace alice
