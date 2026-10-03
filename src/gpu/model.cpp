// GPU-resident Model implementation (design §3): the device-resident
// execution path. gpu::Model now owns its components directly (shape A),
// so this TU supplies the constructor/destructor and the worker/load/run
// paths.
//
// Threading: one worker thread per tensor-parallel rank. Each worker owns
// its own HIP context, rocBLAS handle, device allocations, and queue.

#include "gpu/model.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
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

// Verified bytes (BLAKE3==id) -> an owning Tensor<bf16> of the given shape;
// a size mismatch is a descriptor/packer contract violation.
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

// =============================================================================
// Model lifetime
// =============================================================================

Model::Model() = default;

Model::~Model() {
  // Signal shutdown to every joinable worker and join.
  for (auto& rk : ranks_) {
    if (rk && rk->worker.joinable()) {
      bool should_notify = false;
      {
        std::lock_guard<std::mutex> lock(rk->queue.mutex);
        if (!rk->queue.error) {
          rk->queue.kind = WorkQueue::Kind::Shutdown;
          rk->queue.tokens.clear();
          rk->queue.has_work = true;
          rk->queue.done = false;
          should_notify = true;
        }
      }
      if (should_notify) rk->queue.work_cv.notify_one();
    }
  }
  for (auto& rk : ranks_) {
    if (rk && rk->worker.joinable()) rk->worker.join();
  }
  // Destroy RCCL communicators after workers have exited.
  for (auto& rk : ranks_) {
    if (rk && rk->nccl_comm) {
      (void)ncclCommDestroy(rk->nccl_comm);
      rk->nccl_comm = nullptr;
    }
  }
}

// =============================================================================
// Model core methods
// =============================================================================

void Model::allreduce(int r, float* io, uint32_t S) {
  if (tp_degree_ <= 1) return;
  auto& rk = rank(r);
  if (!rk.nccl_comm) fail("allreduce called before NCCL communicator init");
  const size_t count = size_t(S) * hidden_dim_;
  ncclResult_t nr = ncclAllReduce(io, io, count, ncclFloat, ncclSum,
                                  rk.nccl_comm, 0);  // default stream
  if (nr != ncclSuccess)
    fail("ncclAllReduce failed: " + std::string(ncclGetErrorString(nr)));
  const hipError_t e = hipStreamSynchronize(0);
  if (e != hipSuccess)
    fail(std::string("allreduce stream sync failed: ") + hipGetErrorString(e));
}

const float* Model::run(int r, std::span<const int32_t> tokens,
                        bool all_positions) {
  auto& rk = rank(r);
  if (tokens.empty()) fail("empty token span");
  if (tokens.size() > cfg_.max_prefill_chunk)
    fail("tokens exceed max_prefill_chunk");
  const uint32_t S = uint32_t(tokens.size());
  float* residual = rk.trunk_->begin_pass(S);
  rk.embedder_->embed(tokens, residual);
  for (size_t i = 0; i < rk.blocks_.size(); ++i) {
    rk.trunk_->read(i);
    rk.blocks_[i]->forward(rk.trunk_->io(), S);
    allreduce(r, rk.trunk_->io(), S);
    rk.trunk_->write(i);
  }
  rk.trunk_->output();
  return all_positions ? rk.lm_head_->project_all(rk.trunk_->io(), S)
                       : rk.lm_head_->project_last(rk.trunk_->io(), S);
}

Model::WorkQueue& Model::submit_all(WorkQueue::Kind kind,
                                    std::span<const int32_t> tokens) {
  // Enqueue on every rank.
  for (auto& rkp : ranks_) {
    auto& q = rkp->queue;
    {
      std::lock_guard<std::mutex> lock(q.mutex);
      if (q.error) std::rethrow_exception(q.error);
      q.kind = kind;
      q.tokens.assign(tokens.begin(), tokens.end());
      q.has_work = true;
      q.done = false;
    }
    q.work_cv.notify_one();
  }
  // Wait for every rank.
  for (auto& rkp : ranks_) {
    auto& q = rkp->queue;
    std::unique_lock<std::mutex> lock(q.mutex);
    q.done_cv.wait(lock, [&] { return q.done; });
    q.done = false;
    if (q.error) std::rethrow_exception(q.error);
  }
  return ranks_[0]->queue;
}

void Model::load(int r) {
  auto& rk = rank(r);
  rk.resources_ = std::make_unique<Resources>(r, r, tp_degree_);
  Resources& res = *rk.resources_;

  // Initialize the per-rank RCCL communicator. All ranks must call
  // ncclCommInitRank concurrently with the same unique id. For tp_degree_==1
  // this is skipped entirely.
  if (tp_degree_ > 1) {
    if (!nccl_id_valid_) fail("NCCL unique id not generated before worker load");
    ncclResult_t nr = ncclCommInitRank(&rk.nccl_comm, tp_degree_, nccl_id_, r);
    if (nr != ncclSuccess)
      fail("ncclCommInitRank failed for rank " + std::to_string(r) + ": " +
           ncclGetErrorString(nr));
  }

  using nlohmann::json;

  // --- spec pre-resolution + composition check -------------------------
  std::map<std::string, std::unique_ptr<block::BlockFactory>> resolved;
  std::string iface;
  for (const auto& [name, decl] : load_descriptor_->specs) {
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
  for (const auto& entry : load_descriptor_->blocks)
    arena_bytes += resolved.at(entry.spec)->device_bytes(load_ec_);
  res.init_arena(arena_bytes);

  size_t scratch_bytes = 0;
  for (const auto& [_, rs] : resolved)
    scratch_bytes = std::max(scratch_bytes, rs->scratch_bytes(load_ec_));
  res.init_scratch(scratch_bytes);
  load_stats_->scratch_bytes = std::max(load_stats_->scratch_bytes, scratch_bytes);

  // --- block loop ------------------------------------------------------
  for (size_t i = 0; i < load_descriptor_->blocks.size(); ++i) {
    const engine::BlockEntry& e = load_descriptor_->blocks[i];
    const std::string where = "blocks[" + std::to_string(i) + "]";
    const auto rit = resolved.find(e.spec);
    if (rit == resolved.end())
      fail(where + ": references unknown spec '" + e.spec + "'");
    const block::BlockFactory* rs = rit->second.get();
    load_data_loader_->fetch(e.id, [&](std::span<const std::byte> data) {
      Timer tc;
      rk.blocks_.push_back(rs->construct(data, res, load_ec_));
      load_stats_->construct_ms += tc.ms();
    });
  }

  // --- members ---------------------------------------------------------
  {  // embedder
    const engine::MemberDecl& mm = require_member(*load_descriptor_, "embedder");
    const uint32_t vocab = mm.config.at("vocab_size").get<uint32_t>();
    const uint32_t d = mm.config.at("embedding_dim").get<uint32_t>();
    load_data_loader_->fetch(mm.id, [&](std::span<const std::byte> blob) {
      Timer tc;
      rk.embedder_ = std::make_unique<Embedder>(
          blob_to_tensor(blob, vocab, d, "embedder"), load_ec_.max_prefill_chunk);
      load_stats_->construct_ms += tc.ms();
    });
    hidden_dim_ = d;
  }
  {  // trunk
    const engine::MemberDecl& mm = require_member(*load_descriptor_, "trunk");
    const uint32_t d = mm.config.at("hidden_dim").get<uint32_t>();
    const float eps = mm.config.at("epsilon").get<float>();
    const size_t npre = rk.blocks_.size();
    load_data_loader_->fetch(mm.id, [&](std::span<const std::byte> blob) {
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
      rk.trunk_ = std::make_unique<RmsNormAddTrunk>(
          std::move(pre_norms), std::move(final_norm), eps, d,
          load_ec_.max_prefill_chunk);
      load_stats_->construct_ms += tc.ms();
    });
  }
  {  // language_head
    const engine::MemberDecl& mm = require_member(*load_descriptor_, "language_head");
    const uint32_t vocab = mm.config.at("vocab_size").get<uint32_t>();
    const uint32_t d = mm.config.at("hidden_dim").get<uint32_t>();
    load_data_loader_->fetch(mm.id, [&](std::span<const std::byte> blob) {
      Timer tc;
      rk.lm_head_ = std::make_unique<LmHead>(
          blob_to_tensor(blob, vocab, d, "language_head"),
          load_ec_.max_prefill_chunk, res.handle());
      load_stats_->construct_ms += tc.ms();
    });
  }
  load_stats_->read_ms = load_data_loader_->read_ms();
  load_stats_->hash_ms = load_data_loader_->hash_ms();
  load_stats_->bytes_read = load_data_loader_->bytes_read();
  load_stats_->hash_verifications = load_data_loader_->hash_verifications();
  load_stats_->blocks = rk.blocks_.size();
}

// =============================================================================
// Public API
// =============================================================================

void Model::forward_run(std::span<const int32_t> tokens) {
  submit_all(WorkQueue::Kind::ForwardRun, tokens);
}

std::vector<float> Model::forward_logits(std::span<const int32_t> tokens) {
  auto& q = submit_all(WorkQueue::Kind::ForwardLogits, tokens);
  return std::move(q.result_logits);
}

std::vector<float> Model::forward_logits_all(std::span<const int32_t> tokens) {
  auto& q = submit_all(WorkQueue::Kind::ForwardLogitsAll, tokens);
  return std::move(q.result_logits);
}

void Model::reset() { submit_all(WorkQueue::Kind::Reset); }

uint32_t Model::vocab_size() const { return vocab_size_; }

// =============================================================================
// Worker thread
// =============================================================================

void Model::worker_loop(Model* model, int rank) {
  auto& rk = *model->ranks_[rank];
  try {
    model->load(rank);
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(rk.queue.mutex);
      rk.queue.error = std::current_exception();
    }
    {
      std::lock_guard<std::mutex> lock(model->load_mutex_);
      model->loaded_count_++;
    }
    model->load_cv_.notify_all();
    return;
  }
  {
    std::lock_guard<std::mutex> lock(model->load_mutex_);
    model->loaded_count_++;
  }
  model->load_cv_.notify_all();

  try {
    std::vector<int32_t> tokens;
    for (;;) {
      auto kind = rk.queue.wait_for_work(tokens);
      try {
        switch (kind) {
          case WorkQueue::Kind::ForwardRun: {
            (void)model->run(rank, tokens, false);
            break;
          }
          case WorkQueue::Kind::ForwardLogits: {
            const float* logits = model->run(rank, tokens, false);
            const size_t v = rk.lm_head_->vocab();
            rk.queue.result_logits.resize(v);
            const hipError_t e = hipMemcpy(rk.queue.result_logits.data(), logits,
                                           v * sizeof(float), hipMemcpyDeviceToHost);
            if (e != hipSuccess)
              fail(std::string("logits D2H failed: ") + hipGetErrorString(e));
            break;
          }
          case WorkQueue::Kind::ForwardLogitsAll: {
            const float* logits = model->run(rank, tokens, true);
            const size_t n = tokens.size() * size_t(rk.lm_head_->vocab());
            rk.queue.result_logits.resize(n);
            const hipError_t e = hipMemcpy(rk.queue.result_logits.data(), logits,
                                           n * sizeof(float), hipMemcpyDeviceToHost);
            if (e != hipSuccess)
              fail(std::string("logits D2H failed: ") + hipGetErrorString(e));
            break;
          }
          case WorkQueue::Kind::Reset: {
            for (auto& b : rk.blocks_)
              if (auto* sb = dynamic_cast<block::StatefulBlock*>(b.get()))
                sb->reset_state();
            break;
          }
          case WorkQueue::Kind::Shutdown:
            rk.queue.signal_done();
            return;
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(rk.queue.mutex);
        rk.queue.error = std::current_exception();
      }
      rk.queue.signal_done();
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(rk.queue.mutex);
    rk.queue.error = std::current_exception();
    rk.queue.signal_done();
  }
}

std::unique_ptr<Model> Model::load_from_format(
    const engine::Descriptor& descriptor,
    const loader::LocalFileDataLoader& data_loader, const EngineConfig& ec,
    engine::LoadStats& st) {
  auto m = std::unique_ptr<Model>(new Model());
  m->cfg_ = ec;
  m->tp_degree_ = ec.tp_degree;
  m->load_descriptor_ = &descriptor;
  m->load_data_loader_ = &data_loader;
  m->load_ec_ = ec;
  m->load_stats_ = &st;

  if (ec.tp_degree > 1) {
    ncclResult_t nr = ncclGetUniqueId(&m->nccl_id_);
    if (nr != ncclSuccess)
      fail("ncclGetUniqueId failed: " + std::string(ncclGetErrorString(nr)));
    m->nccl_id_valid_ = true;
  }

  m->ranks_.resize(ec.tp_degree);
  for (uint32_t r = 0; r < ec.tp_degree; ++r) {
    m->ranks_[r] = std::make_unique<Rank>();
    m->ranks_[r]->worker = std::thread(worker_loop, m.get(), r);
  }

  {
    std::unique_lock<std::mutex> lock(m->load_mutex_);
    m->load_cv_.wait(lock,
                     [&] { return m->loaded_count_ == m->tp_degree_; });
  }

  std::exception_ptr err;
  for (auto& rkp : m->ranks_) {
    if (rkp->queue.error) {
      err = rkp->queue.error;
      break;
    }
  }
  if (err) {
    for (auto& rkp : m->ranks_)
      if (rkp->worker.joinable()) rkp->worker.join();
    std::rethrow_exception(err);
  }

  m->vocab_size_ = m->ranks_[0]->lm_head_->vocab();
  return m;
}

}  // namespace alice::gpu
