#pragma once

#include "alice/tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace alice {

class ActivationArena;

// Append/read KV storage for one attention block. A dedicated struct on
// purpose: it has append/read semantics and its own layout, it is not a
// generic tensor. Slots are assigned to attention blocks at model assembly
// time (slot index != position in the block list).
struct KvSlot {
  TensorView k;      // [cap, n_kv_heads * head_dim]
  TensorView v;      // [cap, n_kv_heads * head_dim]
  uint32_t len = 0;  // tokens currently stored
  uint32_t cap = 0;  // capacity in tokens
};

// Mutable per-sequence runtime state. Blocks/Model are immutable after
// assembly; everything that changes token-to-token lives here. This is
// what makes the engine-level model a state machine:
//   step(tokens, state) -> (logits, state')
class InferenceContext {
public:
  InferenceContext(ActivationArena& scratch, size_t kv_slots)
      : scratch_(scratch), kv_(kv_slots) {}

  size_t position() const { return pos_; }
  void advance(size_t n) { pos_ += n; }

  KvSlot& kv(size_t slot) { return kv_.at(slot); }
  ActivationArena& scratch() { return scratch_; }

private:
  ActivationArena& scratch_;
  std::vector<KvSlot> kv_;
  size_t pos_ = 0;
};

}  // namespace alice
