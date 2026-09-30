#pragma once

// GPU-resident Model (design §3): the public entry point for the
// device-resident execution path. The block loop runs entirely on the
// device; token ids are the only regular host<->device traffic.
//
// The component members are INCOMPLETE TYPES (forward-declared below):
// this header stays free of internal- and vendor-includes (no trunk.hpp,
// no HIP/rocBLAS) — those are included only by the implementation TU
// (model.cpp). This is the "exploded pimpl" shape (user direction
// 2026-09): named members instead of one Impl indirection. Consequence
// to remember: a change to the component SET edits this header; a change
// inside a component never does.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "alice/engine_config.hpp"

namespace alice {

namespace gpu_rocm {
class Context;  // device binding + rocBLAS handle (per-GPU, Model-owned)
}

namespace gpu {

class Embedder;
class RmsNormAddTrunk;
class LmHead;
class Sampler;
class Block;  // abstract block interface (src/gpu/block.hpp)

class Model {
 public:
  // model path + manifest -> Model. Throws on contract violation or if the
  // build/machine lacks GPU-resident support.
  static std::unique_ptr<Model> load(const std::string& model_dir,
                                     const std::string& manifest_path,
                                     const EngineConfig& cfg);

  ~Model();

  int32_t forward(std::span<const int32_t> tokens);  // token ids in -> id out
  std::vector<float> forward_logits(std::span<const int32_t> tokens);      // D2H [V]
  std::vector<float> forward_logits_all(std::span<const int32_t> tokens);  // D2H [S*V]
  void reset();  // new session: clear all stateful blocks
  uint32_t vocab_size() const;

 private:
  Model();

  // Shared forward body: embed -> block loop -> trunk output -> project.
  const float* run(std::span<const int32_t> tokens, bool all_positions);

  EngineConfig cfg_;
  std::unique_ptr<gpu_rocm::Context> ctx_;      // device binding + rocBLAS handle
  std::unique_ptr<Embedder> embedder_;
  std::unique_ptr<RmsNormAddTrunk> trunk_;
  std::vector<std::unique_ptr<Block>> blocks_;
  std::unique_ptr<LmHead> lm_head_;
  std::unique_ptr<Sampler> sampler_;
  float* prefill_scratch_ = nullptr;  // shared v4 prefill partials (§14 scratch pool)
};

}  // namespace gpu
}  // namespace alice
