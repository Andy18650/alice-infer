#pragma once

// Shared loader plumbing (design §8 load flow): pread a weight into an
// owning Tensor. Used by the GPU-resident loader.

#include "alice/manifest.hpp"
#include "alice/safetensors.hpp"
#include "alice/tensor.hpp"

namespace alice::loader {

// The load-boundary step: check the dtype, allocate, read. The returned
// Tensor is the staged weight — move it into the consuming constructor,
// which copies/converts/packs into private layout and lets it die on
// return. (A tied lm_head simply stages the embedder's tensor again — the
// alias mechanism is gone; bytes were never shared on device anyway.)
template <typename T>
Tensor<T> stage(const safetensors::Files& f, const WeightSpec& w) {
  const auto& info = f.at(w.name);
  safetensors::File::check_as<T>(info);
  Tensor<T> t(info.rows, info.cols);
  f.read(info, t.data());
  return t;
}

}  // namespace alice::loader
