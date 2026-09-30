#include "gpu/gpu_model.hpp"

#include <stdexcept>

#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>

#include <algorithm>
#include <filesystem>

#include "../loader_staging.hpp"
#include "../blocks/cpu_generic/dense_ffn.hpp"
#include "../blocks/cpu_generic/gqa_attention.hpp"
#include "../blocks/gpu_rocm/platform.hpp"
#include "manifest.hpp"
#include "block.hpp"
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

using Clock = std::chrono::steady_clock;

// RAII section timer for the load path: prints one line on scope exit.
struct LoadSection {
  const char* what;
  Clock::time_point t0;
  explicit LoadSection(const char* w) : what(w), t0(Clock::now()) {}
  ~LoadSection() {
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::fprintf(stderr, "[gpu::Model] %-28s %10.1f ms\n", what, ms);
  }
  LoadSection(const LoadSection&) = delete;
  LoadSection& operator=(const LoadSection&) = delete;
};

}  // namespace
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
Model::~Model() {
  if (prefill_scratch_) (void)hipFree(prefill_scratch_);
}

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
    if (auto* sb = dynamic_cast<StatefulBlock*>(b.get())) sb->reset_state();
}

uint32_t Model::vocab_size() const { return lm_head_->vocab(); }

std::unique_ptr<Model> Model::load(const std::string& model_dir,
                                   const std::string& manifest_path,
                                   const EngineConfig& cfg) {
  const Clock::time_point t_start = Clock::now();
  std::fprintf(stderr, "[gpu::Model] loading %s (manifest: %s)\n", model_dir.c_str(),
               manifest_path.c_str());

  Manifest m;
  std::unique_ptr<safetensors::Files> file;
  {
    LoadSection sec("manifest + audit");
    m = Manifest::parse(manifest_path);
    file = std::make_unique<safetensors::Files>(
        safetensors::Files::open(find_safetensors(model_dir)));

    const auto errors = validate(m, *file);  // two-way audit before any compute
    if (!errors.empty()) {
      std::string msg = "gpu::Model::load: manifest/weight-source contract violated:";
      for (const auto& e : errors) msg += "\n  - " + e;
      throw std::runtime_error(msg);
    }
    if (m.pre_norms.size() != m.blocks.size())
      throw std::runtime_error("gpu::Model::load: pre_norm count != block count");
  }

  auto model = std::unique_ptr<Model>(new Model());
  model->cfg_ = cfg;

  // Acquire the GPU context first: binds the device for this thread and
  // fails loudly if absent (the former lazy platform() singleton's job).
  model->ctx_ = std::make_unique<gpu_rocm::Context>(0);

  {  // embedder
    LoadSection sec("embedder");
    auto table = loader::stage<bf16>(*file, m.token_embedding);
    model->embedder_ =
        std::make_unique<Embedder>(std::move(table), cfg.max_prefill_chunk);
  }
  {  // trunk: staged norms move straight in (freed on return, §8.1)
    LoadSection sec("trunk (norms)");
    std::vector<Tensor<bf16>> pre_norms;
    pre_norms.reserve(m.pre_norms.size());
    for (const auto& n : m.pre_norms) pre_norms.push_back(loader::stage<bf16>(*file, n));
    auto final_norm = loader::stage<bf16>(*file, m.final_norm);
    model->trunk_ = std::make_unique<RmsNormAddTrunk>(
        std::move(pre_norms), std::move(final_norm), m.config.rms_norm_eps,
        m.config.hidden_size, cfg.max_prefill_chunk);
  }
  // Development-only fault-localization hook (design §3/§8.1): block
  // indices listed in ALICE_CPU_REF_BLOCKS (comma-separated) are built as
  // cpu_generic blocks (internal D2H -> host compute -> H2D) instead of
  // device blocks. Never for production.
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
  {  // shared v4 prefill partials allocation (the §14 scratch instance)
    LoadSection sec("prefill scratch");
    const uint32_t max_chunks = (cfg.max_seq_len + 63) / 64;
    const size_t scratch_bytes = size_t(m.config.num_heads) * cfg.max_prefill_chunk *
                                 max_chunks * (m.config.head_dim + 2) * sizeof(float);
    if (hipMalloc(&model->prefill_scratch_, scratch_bytes) != hipSuccess)
      throw std::runtime_error("gpu::Model::load: prefill scratch allocation failed (" +
                               std::to_string(scratch_bytes >> 20) + " MiB)");
  }

  std::string blocks_label = "blocks (" + std::to_string(m.blocks.size()) + ")";
  size_t block_index = 0;
  {  // blocks: stage + construct, one timer for the whole loop
    LoadSection sec(blocks_label.c_str());
  for (const auto& b : m.blocks) {
    const bool ref = cpu_ref.contains(block_index++);
    if (b.type == "gqa_attention_bf16") {
      auto wq = loader::stage<bf16>(*file, b.weights.at("wq"));
      auto wk = loader::stage<bf16>(*file, b.weights.at("wk"));
      auto wv = loader::stage<bf16>(*file, b.weights.at("wv"));
      auto wo = loader::stage<bf16>(*file, b.weights.at("wo"));
      if (ref)
        model->blocks_.push_back(std::make_unique<cpu_generic::GqaAttentionBF16>(
            std::move(wq), std::move(wk), std::move(wv), std::move(wo),
            m.config, cfg));
      else
        model->blocks_.push_back(std::make_unique<gpu_rocm::GqaAttentionBF16>(
            wq.view(), wk.view(), wv.view(), wo.view(), m.config, cfg,
            *model->ctx_, model->prefill_scratch_));
    } else if (b.type == "dense_ffn_bf16") {
      auto w_gate = loader::stage<bf16>(*file, b.weights.at("w_gate"));
      auto w_up = loader::stage<bf16>(*file, b.weights.at("w_up"));
      auto w_down = loader::stage<bf16>(*file, b.weights.at("w_down"));
      if (ref)
        model->blocks_.push_back(std::make_unique<cpu_generic::DenseFfnBF16>(
            std::move(w_gate), std::move(w_up), std::move(w_down),
            m.config, cfg));
      else
        model->blocks_.push_back(std::make_unique<gpu_rocm::DenseFfnBF16>(
            w_gate.view(), w_up.view(), w_down.view(), m.config, cfg,
            *model->ctx_));
    } else {
      throw std::runtime_error("gpu::Model::load: unknown block type '" + b.type + "'");
    }
  }
  }  // blocks section
  {
    LoadSection sec("lm_head");
    auto w = loader::stage<bf16>(*file, m.lm_head);
    model->lm_head_ =
        std::make_unique<LmHead>(std::move(w), cfg.max_prefill_chunk, model->ctx_->handle);
  }
  model->sampler_ = std::make_unique<Sampler>(cfg.sampling);
  const double total_s =
      std::chrono::duration<double>(Clock::now() - t_start).count();
  std::fprintf(stderr, "[gpu::Model] model loaded in %.2f s (vocab %u, %zu blocks, %s arch)\n",
               total_s, model->vocab_size(), m.blocks.size(),
               model->ctx_->arch.c_str());
  return model;
}

}  // namespace alice::gpu
