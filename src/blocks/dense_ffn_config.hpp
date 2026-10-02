#pragma once

// dense_ffn block-type config (design §8): the static shapes a dense-FFN
// block needs — io (hidden) + intermediate. Impl-agnostic: every
// implementation of dense_ffn (gpu_rocm today, future archs later) takes
// this from its factory. This is the block TYPE's contract, not any one
// impl's, so it lives at the block layer (not under gpu_rocm/) and a second
// impl can reuse it without depending on the first. Plain POD; the JSON
// binding (NLOHMANN_DEFINE) lives in the factory .cpp.

#include <cstdint>

namespace alice::block {

struct DenseFfnConfig {
  uint32_t io_dim = 0;           // hidden_size
  uint32_t intermediate_dim = 0;
};

}  // namespace alice::block
