#include "alice/model.hpp"

#include <stdexcept>

#include "alice/dense_ffn_bf16.hpp"
#include "alice/gqa_attention_bf16.hpp"
#include "alice/manifest.hpp"
#include "alice/safetensors.hpp"
#include "loader_staging.hpp"

namespace alice {

F32View Model::run(std::span<const int32_t> tokens, bool all_positions) {
  if (tokens.empty()) throw std::runtime_error("Model::forward: empty token span");
  if (tokens.size() > cfg_.max_prefill_chunk)
    throw std::runtime_error("Model::forward: " + std::to_string(tokens.size()) +
                             " tokens exceeds max_prefill_chunk " +
                             std::to_string(cfg_.max_prefill_chunk));

  F32View residual = trunk_->begin_pass(uint32_t(tokens.size()));
  embedder_->embed(tokens, residual);
  for (size_t i = 0; i < blocks_.size(); ++i) {
    trunk_->read(i);
    blocks_[i]->forward(trunk_->io());
    trunk_->write(i);
  }
  trunk_->output();
  return all_positions ? lm_head_->project(trunk_->io())
                       : lm_head_->project_last(trunk_->io());
}

int32_t Model::forward(std::span<const int32_t> tokens) {
  return sampler_->sample(run(tokens, false));
}

F32View Model::forward_logits(std::span<const int32_t> tokens) {
  return run(tokens, false);
}

F32View Model::forward_logits_all(std::span<const int32_t> tokens) {
  return run(tokens, true);
}

void Model::reset() {
  for (auto& b : blocks_)
    if (auto* sb = dynamic_cast<StatefulBlock*>(b.get())) sb->reset_state();
}

using loader::find_safetensors;
using loader::stage;
using loader::Staging;

std::unique_ptr<Model> ModelLoader::load(const std::string& model_dir,
                                         const std::string& manifest_path,
                                         const EngineConfig& cfg) {
  Manifest m = Manifest::parse(manifest_path);
  auto file = safetensors::Files::open(find_safetensors(model_dir));

  const auto errors = validate(m, file);  // two-way audit before any compute
  if (!errors.empty()) {
    std::string msg = "ModelLoader: manifest/weight-source contract violated:";
    for (const auto& e : errors) msg += "\n  - " + e;
    throw std::runtime_error(msg);
  }

  auto model = std::unique_ptr<Model>(new Model());
  model->cfg_ = cfg;

  {  // embedder
    auto s = stage<bf16>(file, m.token_embedding);
    model->embedder_ = std::make_unique<Embedder>(s.view);
  }
  {  // trunk: stage all norm views, construct (copies out), free
    std::vector<Staging<bf16>> keep;
    std::vector<BF16View> pre_views;
    keep.reserve(m.pre_norms.size() + 1);
    for (const auto& n : m.pre_norms) {
      keep.push_back(stage<bf16>(file, n));
      pre_views.push_back(keep.back().view);
    }
    keep.push_back(stage<bf16>(file, m.final_norm));
    model->trunk_ = std::make_unique<RmsNormAddTrunk>(
        std::move(pre_views), keep.back().view, m.config.rms_norm_eps, m.config.hidden_size,
        cfg.max_prefill_chunk);
    if (m.pre_norms.size() != m.blocks.size())
      throw std::runtime_error("ModelLoader: pre_norm count != block count");
  }
  for (const auto& b : m.blocks) {  // blocks, in execution order
    if (b.type == "gqa_attention_bf16") {
      std::vector<Staging<bf16>> keep;
      for (const char* role : {"wq", "wk", "wv", "wo"}) keep.push_back(stage<bf16>(file, b.weights.at(role)));
      model->blocks_.push_back(GqaAttentionBF16::create(cfg.hardware, keep[0].view, keep[1].view,
                                                        keep[2].view, keep[3].view, m.config, cfg));
    } else if (b.type == "dense_ffn_bf16") {
      std::vector<Staging<bf16>> keep;
      for (const char* role : {"w_gate", "w_up", "w_down"}) keep.push_back(stage<bf16>(file, b.weights.at(role)));
      model->blocks_.push_back(DenseFfnBF16::create(cfg.hardware, keep[0].view, keep[1].view,
                                                    keep[2].view, m.config, cfg));
    } else {
      throw std::runtime_error("ModelLoader: unknown block type '" + b.type + "'");
    }
  }
  {  // lm_head (tied: aliases the embedding table — private copy, noted in lm_head.hpp)
    auto s = stage<bf16>(file, m.lm_head);
    model->lm_head_ = std::make_unique<LmHead>(s.view, cfg.max_prefill_chunk);
  }
  model->sampler_ = std::make_unique<Sampler>(cfg.sampling);
  return model;
}

}  // namespace alice
