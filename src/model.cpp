#include "alice/model.hpp"

#include "alice/arena.hpp"
#include "alice/backbone.hpp"
#include "alice/block.hpp"
#include "alice/context.hpp"
#include "alice/embedder.hpp"
#include "alice/lm_head.hpp"

#include <stdexcept>

namespace alice {

Model::Model(Parts parts) : parts_(std::move(parts)) {
  if (!parts_.embedder || !parts_.backbone || !parts_.lm_head)
    throw std::invalid_argument("Model: embedder, backbone and lm_head are required");
}

void Model::forward(std::span<const int32_t> token_ids, InferenceContext& ctx,
                    TensorView logits_out) {
  ActivationArena& arena = ctx.scratch();
  arena.reset();

  Trunk trunk = parts_.backbone->make_trunk(arena, uint32_t(token_ids.size()));
  parts_.embedder->embed(token_ids, trunk.x);

  for (size_t i = 0; i < parts_.blocks.size(); ++i) {
    TensorView in = parts_.backbone->read(trunk, i, arena);
    TensorView out = arena.alloc_like(in);
    parts_.blocks[i]->forward(in, out, ctx);
    parts_.backbone->write(trunk, i, out, arena);
  }

  parts_.lm_head->project(parts_.backbone->output(trunk, arena), logits_out);
  ctx.advance(token_ids.size());
}

}  // namespace alice
