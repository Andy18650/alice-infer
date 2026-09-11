#include "alice/gpu_model.hpp"

#include <stdexcept>

#include <hip/hip_runtime_api.h>

#include "../loader_staging.hpp"
#include "../blocks/gpu_rocblas/platform.hpp"
#include "alice/manifest.hpp"
#include "block.hpp"
#include "dense_ffn.hpp"
#include "embedder.hpp"
#include "gqa_attention.hpp"
#include "lm_head.hpp"
#include "sampler.hpp"
#include "trunk.hpp"

namespace alice::gpu {

bool resident_available() { return gpu_rocblas::available_impl(); }

struct Model::Impl {
  EngineConfig cfg;
  std::unique_ptr<Embedder> embedder;
  std::unique_ptr<RmsNormAddTrunk> trunk;
  std::vector<std::unique_ptr<Block>> blocks;
  std::unique_ptr<LmHead> lm_head;
  std::unique_ptr<Sampler> sampler;

  const float* run(std::span<const int32_t> tokens, bool all_positions) {
    if (tokens.empty()) throw std::runtime_error("gpu::Model: empty token span");
    if (tokens.size() > cfg.max_prefill_chunk)
      throw std::runtime_error("gpu::Model: tokens exceed max_prefill_chunk");
    const uint32_t S = uint32_t(tokens.size());
    float* residual = trunk->begin_pass(S);
    embedder->embed(tokens, residual);
    for (size_t i = 0; i < blocks.size(); ++i) {
      trunk->read(i);
      blocks[i]->forward(trunk->io(), S);
      trunk->write(i);
    }
    trunk->output();
    return all_positions ? lm_head->project_all(trunk->io(), S)
                         : lm_head->project_last(trunk->io(), S);
  }
};

Model::Model() : impl_(std::make_unique<Impl>()) {}
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

int32_t Model::forward(std::span<const int32_t> tokens) {
  const float* logits = impl_->run(tokens, false);
  return impl_->sampler->sample(logits, impl_->lm_head->vocab());
}

std::vector<float> Model::forward_logits(std::span<const int32_t> tokens) {
  const float* logits = impl_->run(tokens, false);
  const size_t v = impl_->lm_head->vocab();
  std::vector<float> out(v);
  const hipError_t e =
      hipMemcpy(out.data(), logits, v * sizeof(float), hipMemcpyDeviceToHost);
  if (e != hipSuccess)
    throw std::runtime_error(std::string("gpu::Model: logits D2H failed: ") +
                             hipGetErrorString(e));
  return out;
}

std::vector<float> Model::forward_logits_all(std::span<const int32_t> tokens) {
  const float* logits = impl_->run(tokens, true);
  const size_t n = tokens.size() * size_t(impl_->lm_head->vocab());
  std::vector<float> out(n);
  const hipError_t e =
      hipMemcpy(out.data(), logits, n * sizeof(float), hipMemcpyDeviceToHost);
  if (e != hipSuccess)
    throw std::runtime_error(std::string("gpu::Model: logits D2H failed: ") +
                             hipGetErrorString(e));
  return out;
}

void Model::reset() {
  for (auto& b : impl_->blocks)
    if (auto* sb = dynamic_cast<StatefulBlock*>(b.get())) sb->reset_state();
}

uint32_t Model::vocab_size() const { return impl_->lm_head->vocab(); }

std::unique_ptr<Model> Model::load(const std::string& model_dir,
                                   const std::string& manifest_path,
                                   const EngineConfig& cfg) {
  Manifest m = Manifest::parse(manifest_path);
  auto file = safetensors::Files::open(loader::find_safetensors(model_dir));

  const auto errors = validate(m, file);  // two-way audit before any compute
  if (!errors.empty()) {
    std::string msg = "gpu::Model::load: manifest/weight-source contract violated:";
    for (const auto& e : errors) msg += "\n  - " + e;
    throw std::runtime_error(msg);
  }
  if (m.pre_norms.size() != m.blocks.size())
    throw std::runtime_error("gpu::Model::load: pre_norm count != block count");

  auto model = std::unique_ptr<Model>(new Model());
  model->impl_->cfg = cfg;

  {  // embedder
    auto s = loader::stage<bf16>(file, m.token_embedding);
    model->impl_->embedder = std::make_unique<Embedder>(s.view, cfg.max_prefill_chunk);
  }
  {  // trunk
    std::vector<loader::Staging<bf16>> keep;
    std::vector<BF16View> pre_views;
    keep.reserve(m.pre_norms.size() + 1);
    for (const auto& n : m.pre_norms) {
      keep.push_back(loader::stage<bf16>(file, n));
      pre_views.push_back(keep.back().view);
    }
    keep.push_back(loader::stage<bf16>(file, m.final_norm));
    model->impl_->trunk = std::make_unique<RmsNormAddTrunk>(
        std::move(pre_views), keep.back().view, m.config.rms_norm_eps,
        m.config.hidden_size, cfg.max_prefill_chunk);
  }
  for (const auto& b : m.blocks) {
    if (b.type == "gqa_attention_bf16") {
      std::vector<loader::Staging<bf16>> keep;
      for (const char* role : {"wq", "wk", "wv", "wo"})
        keep.push_back(loader::stage<bf16>(file, b.weights.at(role)));
      model->impl_->blocks.push_back(std::make_unique<GqaAttentionBF16>(
          keep[0].view, keep[1].view, keep[2].view, keep[3].view, m.config, cfg));
    } else if (b.type == "dense_ffn_bf16") {
      std::vector<loader::Staging<bf16>> keep;
      for (const char* role : {"w_gate", "w_up", "w_down"})
        keep.push_back(loader::stage<bf16>(file, b.weights.at(role)));
      model->impl_->blocks.push_back(std::make_unique<DenseFfnBF16>(
          keep[0].view, keep[1].view, keep[2].view, m.config, cfg));
    } else {
      throw std::runtime_error("gpu::Model::load: unknown block type '" + b.type + "'");
    }
  }
  {
    auto s = loader::stage<bf16>(file, m.lm_head);
    model->impl_->lm_head = std::make_unique<LmHead>(s.view, cfg.max_prefill_chunk);
  }
  model->impl_->sampler = std::make_unique<Sampler>(cfg.sampling);
  return model;
}

}  // namespace alice::gpu
