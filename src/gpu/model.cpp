// Per-rank GPU Model implementation. One Model per worker thread; the
// Engine owns the threads and the shared work barrier.

#include "gpu/model.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime_api.h>

#include <nlohmann/json.hpp>

#include "alice/tensor.hpp"
#include "blocks/block_factory.hpp"
#include "data_loader.hpp"
#include "engine.hpp"

namespace alice::gpu {

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("gpu::Model: " + what);
}

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
  }
};

Tensor<bf16> blob_to_tensor(std::span<const std::byte> blob, uint32_t rows,
                            uint32_t cols, const char* what) {
  const size_t want = size_t(rows) * cols * sizeof(bf16);
  if (blob.size() != want)
    fail(std::string(what) + ": blob size mismatch (expected " +
         std::to_string(want) + " = " + std::to_string(rows) + "x" +
         std::to_string(cols) + " bf16, got " + std::to_string(blob.size()) + ")");
  return Tensor<bf16>::copy_of(reinterpret_cast<const bf16*>(blob.data()), rows, cols);
}

const engine::MemberDecl& require_member(const engine::Descriptor& d,
                                         const char* key) {
  const auto it = d.members.find(key);
  if (it == d.members.end())
    fail(std::string("descriptor missing required member '") + key + "'");
  return it->second;
}

}  // namespace

Model::Model(const ModelConfig& cfg) : cfg_(cfg) {}

Model::~Model() {
  if (nccl_comm_) (void)ncclCommDestroy(nccl_comm_);
}

void Model::allreduce(float* io, uint32_t S) {
  if (resources_->world_size() <= 1) return;
  if (!nccl_comm_) fail("allreduce called before NCCL communicator init");
  const size_t count = size_t(S) * hidden_dim_;
  ncclResult_t nr = ncclAllReduce(io, io, count, ncclFloat, ncclSum, nccl_comm_, 0);
  if (nr != ncclSuccess)
    fail("ncclAllReduce failed: " + std::string(ncclGetErrorString(nr)));
  const hipError_t e = hipStreamSynchronize(0);
  if (e != hipSuccess)
    fail(std::string("allreduce stream sync failed: ") + hipGetErrorString(e));
}

const float* Model::run(std::span<const int32_t> tokens, bool all_positions) {
  if (tokens.empty()) fail("empty token span");
  if (tokens.size() > cfg_.max_prefill_chunk)
    fail("tokens exceed max_prefill_chunk");
  const uint32_t S = uint32_t(tokens.size());
  float* residual = trunk_->begin_pass(S);
  embedder_->embed(tokens, residual);
  for (size_t i = 0; i < blocks_.size(); ++i) {
    trunk_->read(i);
    blocks_[i]->forward(trunk_->io(), S);
    allreduce(trunk_->io(), S);
    trunk_->write(i);
  }
  trunk_->output();
  return all_positions ? lm_head_->project_all(trunk_->io(), S)
                       : lm_head_->project_last(trunk_->io(), S);
}

void Model::execute(const WorkItem& work) {
  error_ = nullptr;
  try {
    switch (work.kind) {
      case WorkItem::Kind::ForwardRun: {
        (void)run(work.tokens, false);
        break;
      }
      case WorkItem::Kind::ForwardLogits: {
        const float* logits = run(work.tokens, false);
        const size_t v = lm_head_->vocab();
        result_logits_.resize(v);
        const hipError_t e = hipMemcpy(result_logits_.data(), logits, v * sizeof(float),
                                       hipMemcpyDeviceToHost);
        if (e != hipSuccess)
          fail(std::string("logits D2H failed: ") + hipGetErrorString(e));
        break;
      }
      case WorkItem::Kind::ForwardLogitsAll: {
        const float* logits = run(work.tokens, true);
        const size_t n = work.tokens.size() * size_t(lm_head_->vocab());
        result_logits_.resize(n);
        const hipError_t e = hipMemcpy(result_logits_.data(), logits, n * sizeof(float),
                                       hipMemcpyDeviceToHost);
        if (e != hipSuccess)
          fail(std::string("logits D2H failed: ") + hipGetErrorString(e));
        break;
      }
      case WorkItem::Kind::Reset: {
        for (auto& b : blocks_)
          if (auto* sb = dynamic_cast<block::StatefulBlock*>(b.get()))
            sb->reset_state();
        break;
      }
      case WorkItem::Kind::Stop:
        break;
    }
  } catch (...) {
    error_ = std::current_exception();
  }
}

void Model::load(int rank, int world_size,
                 const engine::Descriptor& descriptor,
                 const loader::LocalFileDataLoader& data_loader,
                 engine::LoadStats& st, ncclUniqueId nccl_id) {
  // Resources must be constructed in the worker thread so hipSetDevice is
  // thread-local; ncclCommInitRank uses the current thread's device.
  resources_ = std::make_unique<Resources>(rank, world_size);

  if (world_size > 1) {
    ncclResult_t nr = ncclCommInitRank(&nccl_comm_, world_size, nccl_id, rank);
    if (nr != ncclSuccess)
      fail("ncclCommInitRank failed for rank " + std::to_string(rank) + ": " +
           ncclGetErrorString(nr));
  }

  using nlohmann::json;

  // --- spec pre-resolution + composition check -------------------------
  std::map<std::string, std::unique_ptr<block::BlockFactory>> resolved;
  std::string iface;
  for (const auto& [name, decl] : descriptor.specs) {
    const std::string where = "block_specs." + name;
    const auto it = block::registry().find(decl.type);
    if (it == block::registry().end())
      fail(where + ": unknown block type '" + decl.type + "'");
    std::unique_ptr<block::BlockFactory> rs;
    try {
      rs = it->second(decl.version, decl.config);
    } catch (const json::exception& e) {
      fail(where + " (" + decl.type + " v" + std::to_string(decl.version) +
           "): " + e.what());
    }
    const std::string isig = rs->interface_sig();
    if (!isig.empty()) {
      if (iface.empty())
        iface = isig;
      else if (iface != isig)
        fail(where + ": interface " + isig + " disagrees with " + iface +
             " — composition does not bind");
    }
    resolved[name] = std::move(rs);
  }

  // --- arena + scratch -------------------------------------------------
  uint64_t arena_bytes = 0;
  for (const auto& entry : descriptor.blocks)
    arena_bytes += resolved.at(entry.spec)->device_bytes(cfg_);
  resources_->init_arena(arena_bytes);

  size_t scratch_bytes = 0;
  for (const auto& [_, rs] : resolved)
    scratch_bytes = std::max(scratch_bytes, rs->scratch_bytes(cfg_));
  resources_->init_scratch(scratch_bytes);
  st.scratch_bytes = std::max(st.scratch_bytes, scratch_bytes);

  // --- block loop ------------------------------------------------------
  for (size_t i = 0; i < descriptor.blocks.size(); ++i) {
    const engine::BlockEntry& e = descriptor.blocks[i];
    const std::string where = "blocks[" + std::to_string(i) + "]";
    const auto rit = resolved.find(e.spec);
    if (rit == resolved.end())
      fail(where + ": references unknown spec '" + e.spec + "'");
    const block::BlockFactory* rs = rit->second.get();
    data_loader.fetch(e.id, [&](std::span<const std::byte> data) {
      Timer tc;
      blocks_.push_back(rs->construct(data, *resources_, cfg_));
      st.construct_ms += tc.ms();
    });
  }

  // --- members ---------------------------------------------------------
  {  // embedder
    const engine::MemberDecl& mm = require_member(descriptor, "embedder");
    const uint32_t vocab = mm.config.at("vocab_size").get<uint32_t>();
    const uint32_t d = mm.config.at("embedding_dim").get<uint32_t>();
    data_loader.fetch(mm.id, [&](std::span<const std::byte> blob) {
      Timer tc;
      embedder_ = std::make_unique<Embedder>(
          blob_to_tensor(blob, vocab, d, "embedder"), cfg_.max_prefill_chunk);
      st.construct_ms += tc.ms();
    });
    hidden_dim_ = d;
  }
  {  // trunk
    const engine::MemberDecl& mm = require_member(descriptor, "trunk");
    const uint32_t d = mm.config.at("hidden_dim").get<uint32_t>();
    const float eps = mm.config.at("epsilon").get<float>();
    const size_t npre = blocks_.size();
    data_loader.fetch(mm.id, [&](std::span<const std::byte> blob) {
      Timer tc;
      const size_t per = size_t(d) * sizeof(bf16);
      const size_t want = (npre + 1) * per;
      if (blob.size() != want)
        fail("trunk: blob size mismatch (expected " + std::to_string(want) +
             " = " + std::to_string(npre + 1) + "x" + std::to_string(d) +
             " bf16, got " + std::to_string(blob.size()) + ")");
      std::vector<Tensor<bf16>> pre_norms;
      pre_norms.reserve(npre);
      const auto* p = reinterpret_cast<const bf16*>(blob.data());
      for (size_t i = 0; i < npre; ++i)
        pre_norms.push_back(Tensor<bf16>::copy_of(p + i * d, 1, d));
      auto final_norm = Tensor<bf16>::copy_of(p + npre * d, 1, d);
      trunk_ = std::make_unique<RmsNormAddTrunk>(
          std::move(pre_norms), std::move(final_norm), eps, d,
          cfg_.max_prefill_chunk);
      st.construct_ms += tc.ms();
    });
  }
  {  // language_head
    const engine::MemberDecl& mm = require_member(descriptor, "language_head");
    const uint32_t vocab = mm.config.at("vocab_size").get<uint32_t>();
    const uint32_t d = mm.config.at("hidden_dim").get<uint32_t>();
    data_loader.fetch(mm.id, [&](std::span<const std::byte> blob) {
      Timer tc;
      lm_head_ = std::make_unique<LmHead>(
          blob_to_tensor(blob, vocab, d, "language_head"),
          cfg_.max_prefill_chunk, resources_->handle());
      st.construct_ms += tc.ms();
    });
  }

  st.read_ms = data_loader.read_ms();
  st.hash_ms = data_loader.hash_ms();
  st.bytes_read = data_loader.bytes_read();
  st.hash_verifications = data_loader.hash_verifications();
  st.blocks = blocks_.size();
}

}  // namespace alice::gpu
