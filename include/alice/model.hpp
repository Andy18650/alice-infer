#pragma once

#include "alice/tensor.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace alice {

class Backbone;
class Block;
class Embedder;
class InferenceContext;
class LmHead;

// Container, not architecture: owns the components and runs the wiring
// loop. Carries no weights of its own — embedder/backbone/blocks/lm_head
// carry theirs. All architecture knowledge lives in the components, so
// supporting a new wiring scheme never touches this class.
class Model {
public:
  struct Parts {
    std::unique_ptr<Embedder> embedder;
    std::unique_ptr<Backbone> backbone;
    std::vector<std::unique_ptr<Block>> blocks;
    std::unique_ptr<LmHead> lm_head;
  };

  explicit Model(Parts parts);  // throws if a required part is missing

  // One state-machine step: consume token_ids, produce logits for every
  // position (the caller typically reads only the last row at decode).
  void forward(std::span<const int32_t> token_ids, InferenceContext& ctx,
               TensorView logits_out);

  size_t num_blocks() const { return parts_.blocks.size(); }

private:
  Parts parts_;
};

}  // namespace alice
