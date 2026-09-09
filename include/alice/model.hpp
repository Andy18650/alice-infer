#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "alice/block.hpp"
#include "alice/embedder.hpp"
#include "alice/engine_config.hpp"
#include "alice/lm_head.hpp"
#include "alice/sampler.hpp"
#include "alice/trunk.hpp"

namespace alice {

// Container + wiring loop (design §7.3): owns its parts directly, knows no
// architecture specifics. Top-level contract: token ids in -> token id out.
// The generation loop, EOS, and detokenization live in the frontend.
class Model {
 public:
  // Processes `tokens` (prefill chunk or single decode token) and returns
  // the sampled next token. Hard error if tokens.size() > max_prefill_chunk,
  // if the context would exceed max_seq_len, or on empty input.
  int32_t forward(std::span<const int32_t> tokens);

  // New session: resets all stateful blocks (KV caches), keeps allocations.
  void reset();

  uint32_t vocab_size() const { return embedder_->vocab_size(); }

 private:
  friend class ModelLoader;
  Model() = default;

  EngineConfig cfg_;
  std::unique_ptr<Embedder> embedder_;
  std::unique_ptr<RmsNormAddTrunk> trunk_;
  std::vector<std::unique_ptr<Block>> blocks_;
  std::unique_ptr<LmHead> lm_head_;
  std::unique_ptr<Sampler> sampler_;
};

// Factory (design §7.4): sniff-then-delegate; safetensors only for now.
// model_dir must contain exactly one *.safetensors (multi-shard assembly
// arrives with the 70B). Loads per the §8 flow: validate the manifest
// against the weight source, then per component — pread to staging,
// construct (copies into private buffers), free staging.
class ModelLoader {
 public:
  // Throws std::runtime_error on any validation or IO failure.
  static std::unique_ptr<Model> load(const std::string& model_dir,
                                     const std::string& manifest_path,
                                     const EngineConfig& cfg);
};

}  // namespace alice
