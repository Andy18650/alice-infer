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
#include <vector>

#include "alice/engine_config.hpp"

namespace alice {

namespace block {
class Block;  // abstract block interface (src/blocks/block.hpp)
}  // namespace block

namespace loader {
class LocalFileDataLoader;  // content-addressed bytes source (src/data_loader.hpp)
}  // namespace loader

namespace modelfmt {
struct Descriptor;  // parsed .model descriptor (src/model_format.hpp)
struct LoadStats;   // load timing/throughput (src/model_format.hpp)
}  // namespace modelfmt

namespace gpu {

class Resources;  // GPU resources (device + BLAS handle + allocator)
class Embedder;
class RmsNormAddTrunk;
class LmHead;

class Model {
 public:
  // Parsed .model descriptor + content-addressed bytes source -> Model.
  // The model-format load path: constructs the device model in place from
  // the descriptor's content-addressed blocks + members (same component
  // constructors as the manifest path; arena covers the block loop only).
  // Defined in src/model_format.cpp (the format loader TU). Throws on any
  // framing, parse, hash, or length failure.
  static std::unique_ptr<Model> load_from_format(
      const modelfmt::Descriptor& descriptor,
      const loader::LocalFileDataLoader& data_loader, const EngineConfig& ec,
      modelfmt::LoadStats& st);

  ~Model();

  // Run forward for these token ids without D2H or sampling. Used for
  // intermediate prefill chunks where only the final state matters.
  void forward_run(std::span<const int32_t> tokens);

  std::vector<float> forward_logits(std::span<const int32_t> tokens);      // D2H [V]
  std::vector<float> forward_logits_all(std::span<const int32_t> tokens);  // D2H [S*V]
  void reset();  // new session: clear all stateful blocks
  uint32_t vocab_size() const;

 private:
  Model();

  // Threading step B: all device state and loading live inside the worker
  // thread. The Model object is just a thin synchronous wrapper.
  struct Impl;
  std::unique_ptr<Impl> impl_;
  uint32_t vocab_size_ = 0;

  static void worker_loop(Model* model, int rank);
};

}  // namespace gpu
}  // namespace alice
