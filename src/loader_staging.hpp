#pragma once

// Shared loader plumbing (design §8 load flow): pread a (possibly aliased)
// weight into staging RAM, and shard discovery. Used by both the host
// ModelLoader and the GPU-resident loader.

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

// All .safetensors shards in the directory, lexically sorted (HF shard
// names are zero-padded, so lexical order == shard order; probe order is
// semantically irrelevant but deterministic error messages are nice).
inline std::vector<std::string> find_safetensors(const std::string& dir) {
  std::vector<std::string> files;
  for (const auto& e : std::filesystem::directory_iterator(dir))
    if (e.path().extension() == ".safetensors") files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  if (files.empty()) throw std::runtime_error("ModelLoader: no .safetensors in " + dir);
  return files;
}

}  // namespace alice::loader
