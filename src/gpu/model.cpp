// GPU-resident Model (design §3): the device-resident execution path. The
// block loop runs entirely on the device; token ids are the only regular
// host<->device traffic. Loading lives in src/model_format.cpp
// (init_engine / Model::load_from_format — the model-format path); this TU
// holds the forward/run runtime + the member destructors.
//
// The component members are INCOMPLETE TYPES (forward-declared in
// gpu_model.hpp): this file is the only TU that pulls the member headers
// (embedder/trunk/lm_head/sampler/resources). "exploded pimpl" shape (user
// direction 2026-09): named members instead of one Impl indirection.

#include "gpu/gpu_model.hpp"

#include <stdexcept>

#include <hip/hip_runtime_api.h>

#include "blocks/block.hpp"  // block::StatefulBlock (reset)
#include "embedder.hpp"
#include "lm_head.hpp"
#include "resources.hpp"
#include "sampler.hpp"
#include "trunk.hpp"

namespace alice::gpu {

const float* Model::run(std::span<const int32_t> tokens, bool all_positions) {
  if (tokens.empty()) throw std::runtime_error("gpu::Model: empty token span");
  if (tokens.size() > cfg_.max_prefill_chunk)
    throw std::runtime_error("gpu::Model: tokens exceed max_prefill_chunk");
  const uint32_t S = uint32_t(tokens.size());
  float* residual = trunk_->begin_pass(S);
  embedder_->embed(tokens, residual);
  for (size_t i = 0; i < blocks_.size(); ++i) {
    trunk_->read(i);
    blocks_[i]->forward(trunk_->io(), S);
    trunk_->write(i);
  }
  trunk_->output();
  return all_positions ? lm_head_->project_all(trunk_->io(), S)
                       : lm_head_->project_last(trunk_->io(), S);
}

Model::Model() = default;
Model::~Model() = default;

int32_t Model::forward(std::span<const int32_t> tokens) {
  const float* logits = run(tokens, false);
  return sampler_->sample(logits, lm_head_->vocab());
}

std::vector<float> Model::forward_logits(std::span<const int32_t> tokens) {
  const float* logits = run(tokens, false);
  const size_t v = lm_head_->vocab();
  std::vector<float> out(v);
  const hipError_t e =
      hipMemcpy(out.data(), logits, v * sizeof(float), hipMemcpyDeviceToHost);
  if (e != hipSuccess)
    throw std::runtime_error(std::string("gpu::Model: logits D2H failed: ") +
                             hipGetErrorString(e));
  return out;
}

std::vector<float> Model::forward_logits_all(std::span<const int32_t> tokens) {
  const float* logits = run(tokens, true);
  const size_t n = tokens.size() * size_t(lm_head_->vocab());
  std::vector<float> out(n);
  const hipError_t e =
      hipMemcpy(out.data(), logits, n * sizeof(float), hipMemcpyDeviceToHost);
  if (e != hipSuccess)
    throw std::runtime_error(std::string("gpu::Model: logits D2H failed: ") +
                             hipGetErrorString(e));
  return out;
}

void Model::reset() {
  for (auto& b : blocks_)
    if (auto* sb = dynamic_cast<block::StatefulBlock*>(b.get())) sb->reset_state();
}

uint32_t Model::vocab_size() const { return lm_head_->vocab(); }

}  // namespace alice::gpu
