#pragma once

// Shared loader plumbing (design §8 load flow): pread a (possibly aliased)
// weight into staging RAM. Used by the GPU-resident loader.

#include <memory>
#include <stdexcept>

#include "alice/manifest.hpp"
#include "alice/safetensors.hpp"

namespace alice::loader {

template <typename T>
struct Staging {
  std::unique_ptr<T[]> buf;
  TView<T> view;
};

// view_as<T> is the single checked runtime->static dtype conversion at the
// load boundary (the manifest audit already validated it).
template <typename T>
Staging<T> stage(const safetensors::Files& f, const WeightSpec& w) {
  const auto& info = f.at(w.is_alias() ? w.alias_of : w.name);
  Staging<T> s;
  s.buf = std::make_unique<T[]>(info.numel());
  f.read(info, s.buf.get());
  s.view = safetensors::File::view_as<T>(info, s.buf.get());
  return s;
}

}  // namespace alice::loader
