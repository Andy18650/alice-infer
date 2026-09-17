#include "alice/gpu_model.hpp"

#include <stdexcept>

#include <hip/hip_runtime_api.h>

#include <cstdlib>
#include <set>

#include <algorithm>
#include <filesystem>

#include "../loader_staging.hpp"
#include "../blocks/gpu_rocm/platform.hpp"
#include "alice/manifest.hpp"
#include "block.hpp"
#include "cpu_reference.hpp"
#include "../blocks/gpu_rocm/dense_ffn.hpp"
#include "embedder.hpp"
#include "../blocks/gpu_rocm/gqa_attention.hpp"
#include "lm_head.hpp"
#include "sampler.hpp"
#include "trunk.hpp"

namespace alice::gpu {

namespace {

// All .safetensors shards in the directory, lexically sorted (HF shard
// names are zero-padded, so lexical order == shard order; probe order is
// semantically irrelevant but deterministic error messages are nice).
// Discovery policy, loader-side by design (§4: sharding is physical
// packaging, invisible to the contract) — lives here, next to its only
// caller, until staging gets its planned rework.
std::vector<std::string> find_safetensors(const std::string& dir) {
  std::vector<std::string> files;
  for (const auto& e : std::filesystem::directory_iterator(dir))
    if (e.path().extension() == ".safetensors") files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  if (files.empty()) throw std::runtime_error("ModelLoader: no .safetensors in " + dir);
  return files;
}

}  // namespace
struct Model::Impl {
  EngineConfig cfg;
  std::unique_ptr<Embedder> embedder;
  std::unique_ptr<RmsNormAddTrunk> trunk;
  std::vector<std::unique_ptr<Block>> blocks;
  std::unique_ptr<LmHead> lm_head;
  std::unique_ptr<Sampler> sampler;
  float* prefill_scratch = nullptr;  // shared v4 partials (design §14 scratch pool)
  ~Impl() {
    if (prefill_scratch) (void)hipFree(prefill_scratch);
  }

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
  auto file = safetensors::Files::open(find_safetensors(model_dir));

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
  // Development-only fault-localization hook (design §3/§8.1): block
  // indices listed in ALICE_CPU_REF_BLOCKS (comma-separated) are built as
  // CpuReferenceBlock adapters (D2H -> cpu_generic compute -> H2D) instead
  // of device blocks. Never for production.
  std::set<size_t> cpu_ref;
  if (const char* env = std::getenv("ALICE_CPU_REF_BLOCKS")) {
    const std::string s(env);
    for (size_t off = 0; off < s.size();) {
      const size_t comma = s.find(',', off);
      cpu_ref.insert(std::stoul(s.substr(off, comma - off)));
      off = comma == std::string::npos ? s.size() : comma + 1;
    }
    std::fprintf(stderr, "[gpu::Model] ALICE_CPU_REF_BLOCKS: %zu block(s) on CPU "
                         "reference (dev-only)\n",
                 cpu_ref.size());
  }

  // Shared v4 prefill partials (design §14 scratch pool, first instance):
  // one buffer reused by every attention block; each block's partials are
  // produced and consumed within its own forward (split -> combine).
  const uint32_t max_chunks = (cfg.max_seq_len + 63) / 64;
  const size_t scratch_bytes = size_t(m.config.num_heads) * cfg.max_prefill_chunk *
                               max_chunks * (m.config.head_dim + 2) * sizeof(float);
  if (hipMalloc(&model->impl_->prefill_scratch, scratch_bytes) != hipSuccess)
    throw std::runtime_error("gpu::Model::load: prefill scratch allocation failed (" +
                             std::to_string(scratch_bytes >> 20) + " MiB)");

  size_t block_index = 0;
  for (const auto& b : m.blocks) {
    const bool ref = cpu_ref.contains(block_index++);
    if (b.type == "gqa_attention_bf16") {
      auto wq = loader::stage<bf16>(file, b.weights.at("wq"));
      auto wk = loader::stage<bf16>(file, b.weights.at("wk"));
      auto wv = loader::stage<bf16>(file, b.weights.at("wv"));
      auto wo = loader::stage<bf16>(file, b.weights.at("wo"));
      if (ref)
        model->impl_->blocks.push_back(std::make_unique<CpuReferenceBlock>(
            cpu_generic::GqaAttentionBF16CpuGeneric(wq.view, wk.view, wv.view, wo.view,
                                                    m.config, cfg),
            m.config.hidden_size, cfg.max_prefill_chunk));
      else
        model->impl_->blocks.push_back(std::make_unique<gpu_rocm::GqaAttentionBF16>(
            wq.view, wk.view, wv.view, wo.view, m.config, cfg,
            model->impl_->prefill_scratch));
    } else if (b.type == "dense_ffn_bf16") {
      auto w_gate = loader::stage<bf16>(file, b.weights.at("w_gate"));
      auto w_up = loader::stage<bf16>(file, b.weights.at("w_up"));
      auto w_down = loader::stage<bf16>(file, b.weights.at("w_down"));
      if (ref)
        model->impl_->blocks.push_back(std::make_unique<CpuReferenceBlock>(
            cpu_generic::DenseFfnBF16CpuGeneric(w_gate.view, w_up.view, w_down.view,
                                                m.config, cfg),
            m.config.hidden_size, cfg.max_prefill_chunk));
      else
        model->impl_->blocks.push_back(std::make_unique<gpu_rocm::DenseFfnBF16>(
            w_gate.view, w_up.view, w_down.view, m.config, cfg));
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
