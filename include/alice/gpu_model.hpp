#pragma once

// GPU-resident Model (design §3): the public entry point for the
// device-resident execution path. The block loop runs entirely on the
// device; token ids are the only regular host<->device traffic.
//
// Available only in builds with ALICE_GPU_RESIDENT (hipcc + ROCm found);
// the loader throws std::runtime_error otherwise.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine_config.hpp"

namespace alice::gpu {

class Model {
 public:
  // model path + manifest -> Model. Throws on contract violation or if the
  // build/machine lacks GPU-resident support.
  static std::unique_ptr<Model> load(const std::string& model_dir,
                                     const std::string& manifest_path,
                                     const EngineConfig& cfg);

  ~Model();
  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;

  int32_t forward(std::span<const int32_t> tokens);  // token ids in -> id out
  std::vector<float> forward_logits(std::span<const int32_t> tokens);      // D2H [V]
  std::vector<float> forward_logits_all(std::span<const int32_t> tokens);  // D2H [S*V]
  void reset();  // new session: clear all stateful blocks
  uint32_t vocab_size() const;

 private:
  Model();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// True when the GPU-resident path was compiled in (hipcc + ROCm) AND a ROCm
// device is present. The REPL/loader use this for selection.
bool resident_available();

}  // namespace alice::gpu
