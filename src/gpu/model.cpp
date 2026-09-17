#include "alice/gpu_model.hpp"

#include <stdexcept>

#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdio>
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
struct Model::Impl {
  EngineConfig cfg;
  gpu_rocm::Context ctx;  // device binding + rocBLAS handle (per-GPU, owned)
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
  model->impl_->cfg = cfg;

  // Acquire the GPU context first: binds the device for this thread and
  // fails loudly if absent (the former lazy platform() singleton's job).
  model->impl_->ctx = gpu_rocm::Context(0);

  {  // embedder
    LoadSection sec("embedder");
    auto table = loader::stage<bf16>(*file, m.token_embedding);
    model->impl_->embedder =
        std::make_unique<Embedder>(std::move(table), cfg.max_prefill_chunk);
  }
  {  // trunk: staged norms move straight in (freed on return, §8.1)
    LoadSection sec("trunk (norms)");
    std::vector<Tensor<bf16>> pre_norms;
    pre_norms.reserve(m.pre_norms.size());
    for (const auto& n : m.pre_norms) pre_norms.push_back(loader::stage<bf16>(*file, n));
    auto final_norm = loader::stage<bf16>(*file, m.final_norm);
    model->impl_->trunk = std::make_unique<RmsNormAddTrunk>(
        std::move(pre_norms), std::move(final_norm), m.config.rms_norm_eps,
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
  {  // shared v4 prefill partials allocation (the §14 scratch instance)
    LoadSection sec("prefill scratch");
    const uint32_t max_chunks = (cfg.max_seq_len + 63) / 64;
    const size_t scratch_bytes = size_t(m.config.num_heads) * cfg.max_prefill_chunk *
                                 max_chunks * (m.config.head_dim + 2) * sizeof(float);
    if (hipMalloc(&model->impl_->prefill_scratch, scratch_bytes) != hipSuccess)
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
        model->impl_->blocks.push_back(std::make_unique<CpuReferenceBlock>(
            cpu_generic::GqaAttentionBF16CpuGeneric(std::move(wq), std::move(wk),
                                                    std::move(wv), std::move(wo),
                                                    m.config, cfg),
            m.config.hidden_size, cfg.max_prefill_chunk));
      else
        model->impl_->blocks.push_back(std::make_unique<gpu_rocm::GqaAttentionBF16>(
            std::move(wq), std::move(wk), std::move(wv), std::move(wo), m.config, cfg,
            model->impl_->ctx, model->impl_->prefill_scratch));
    } else if (b.type == "dense_ffn_bf16") {
      auto w_gate = loader::stage<bf16>(*file, b.weights.at("w_gate"));
      auto w_up = loader::stage<bf16>(*file, b.weights.at("w_up"));
      auto w_down = loader::stage<bf16>(*file, b.weights.at("w_down"));
      if (ref)
        model->impl_->blocks.push_back(std::make_unique<CpuReferenceBlock>(
            cpu_generic::DenseFfnBF16CpuGeneric(std::move(w_gate), std::move(w_up),
                                                std::move(w_down), m.config, cfg),
            m.config.hidden_size, cfg.max_prefill_chunk));
      else
        model->impl_->blocks.push_back(std::make_unique<gpu_rocm::DenseFfnBF16>(
            std::move(w_gate), std::move(w_up), std::move(w_down), m.config, cfg,
            model->impl_->ctx));
    } else {
      throw std::runtime_error("gpu::Model::load: unknown block type '" + b.type + "'");
    }
  }
  }  // blocks section
  {
    LoadSection sec("lm_head");
    auto w = loader::stage<bf16>(*file, m.lm_head);
    model->impl_->lm_head =
        std::make_unique<LmHead>(std::move(w), cfg.max_prefill_chunk, model->impl_->ctx.handle);
  }
  model->impl_->sampler = std::make_unique<Sampler>(cfg.sampling);
  const double total_s =
      std::chrono::duration<double>(Clock::now() - t_start).count();
  std::fprintf(stderr, "[gpu::Model] model loaded in %.2f s (vocab %u, %zu blocks, %s arch)\n",
               total_s, model->vocab_size(), m.blocks.size(),
               model->impl_->ctx.arch.c_str());
  return model;
}

}  // namespace alice::gpu
