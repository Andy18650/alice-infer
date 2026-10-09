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
#include "gpu/kernels.hpp"

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
  if (resources_) (void)hipSetDevice(rank_);
  if (decode_exec_) (void)hipGraphExecDestroy(decode_exec_);
  if (decode_graph_) (void)hipGraphDestroy(decode_graph_);
  if (prefill_comm_) (void)ncclCommDestroy(prefill_comm_);
  if (decode_epoch_base_) (void)hipFree(decode_epoch_base_);
  if (ar_flag_) (void)hipFree(ar_flag_);
  if (ar_flags_dev_) (void)hipFree(ar_flags_dev_);
  if (ar_slabs_dev_) (void)hipFree(ar_slabs_dev_);
  if (ar_contributions_) (void)hipFree(ar_contributions_);
}

// Device-flag P2P allreduce (replaces the retired rank-0 hipMemcpyPeer
// reduce+broadcast, which was host-bound: two hipStreamSynchronize + host
// atomics per call). Cross-GCD sync rides fine-grained device flags with
// system-scope atomic release/poll/acquire — the probe-verified protocol
// from research/p2p-allreduce (atomics.md: release emits buffer_wbl2 so the
// writer's io reaches HBM, which is what XGMI reads are served by; one
// acquire per barrier emits a broadcast buffer_invl2 + L1 invalidate so the
// reader can't hit stale lines). No host API on the critical path at all.
//
// Immutable per-block input slots, separate trunk output; no consume epoch.
// S=1: tiny rows retain fused single-CTA; 32 KiB uses the measured
// pointer-hoisted/dephased data-major decode (ready + gather).
// S>1: out-of-place RCCL, the vendor prefill winner. Both enqueue on the rank
// stream. Gaps in P2P epochs from RCCL are intentional; counters never reset.
// Dephasing changes FP32 summation order (including across ranks); equivalence
// is tolerance-based and validated by canonical fixtures, not bit identity.
void Model::allreduce(float* out, uint32_t S, size_t slot_offset) {
  if (world_size_ <= 1) return;
  const uint32_t n = S * hidden_dim_;
  const uint64_t vprod = ++ar_epoch_;
  if (S > 1) {
    const auto e = ncclAllReduce(ar_contributions_ + slot_offset, out, n, ncclFloat,
                                 ncclSum, prefill_comm_, resources_->stream());
    if (e != ncclSuccess) fail(std::string("prefill allreduce: ") + ncclGetErrorString(e));
    return;
  }
  // The 8/16 KiB shapes do not benefit from a second launch/CU; end-to-end
  // 1B validation caught a regression when using the 32 KiB strategy there.
  if (n <= 4096) {
    kernels::ar_fused_f4(resources_->stream(), out, ar_slabs_dev_, ar_flag_, ar_flags_dev_,
                         world_size_, rank_, n, slot_offset, vprod,
                         recording_decode_ ? decode_epoch_base_ : nullptr);
    return;
  }
  kernels::ar_flagbar(resources_->stream(), ar_flag_, ar_flags_dev_, world_size_,
                      rank_, vprod, recording_decode_ ? decode_epoch_base_ : nullptr);
  kernels::ar_decode_gather_f4(resources_->stream(), ar_slabs_dev_, out, world_size_,
                               rank_, n, slot_offset);
}

const float* Model::run(std::span<const int32_t> tokens, bool all_positions) {
  if (tokens.empty()) fail("empty token span");
  if (tokens.size() > cfg_.max_prefill_chunk)
    fail("tokens exceed max_prefill_chunk");
  const uint32_t S = uint32_t(tokens.size());
  float* residual = trunk_->begin_pass(S);
  embedder_->embed(tokens, residual);
  if (S == 1 && decode_exec_) {
    for (auto& b : blocks_)
      if (auto* sb = dynamic_cast<block::StatefulBlock*>(b.get()))
        sb->advance_decode_state();
    if (world_size_ > 1) {
      decode_epoch_base_host_ = ar_epoch_;
      const auto e = hipMemcpyAsync(decode_epoch_base_, &decode_epoch_base_host_,
                                    sizeof(uint64_t), hipMemcpyHostToDevice,
                                    resources_->stream());
      if (e != hipSuccess) fail(std::string("decode epoch upload: ") + hipGetErrorString(e));
      ar_epoch_ += blocks_.size();
    }
    const auto e = hipGraphLaunch(decode_exec_, resources_->stream());
    if (e != hipSuccess) fail(std::string("decode graph launch: ") + hipGetErrorString(e));
    return decode_logits_;  // S=1 project_all and project_last have identical shape
  }
  return run_compute(S, all_positions);
}

const float* Model::run_compute(uint32_t S, bool all_positions) {
  for (size_t i = 0; i < blocks_.size(); ++i) {
    const size_t offset = i * ar_slot_stride_;
    float* contribution = world_size_ > 1 ? ar_contributions_ + offset : trunk_->io();
    trunk_->read(i, contribution);
    blocks_[i]->forward(contribution, S);
    allreduce(trunk_->io(), S, offset);
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
        // Completion, not just submission: Engine waits for all workers before
        // the next forward may overwrite any rank's published slab slots.
        const hipError_t e = hipStreamSynchronize(resources_->stream());
        if (e != hipSuccess)
          fail(std::string("ForwardRun sync failed: ") + hipGetErrorString(e));
        break;
      }
      case WorkItem::Kind::ForwardLogits: {
        const float* logits = run(work.tokens, false);
        const size_t v = lm_head_->vocab();
        result_logits_.resize(v);
        hipError_t e = hipMemcpyAsync(result_logits_.data(), logits, v * sizeof(float),
                                      hipMemcpyDeviceToHost, resources_->stream());
        if (e == hipSuccess) e = hipStreamSynchronize(resources_->stream());
        if (e != hipSuccess)
          fail(std::string("logits D2H failed: ") + hipGetErrorString(e));
        break;
      }
      case WorkItem::Kind::ForwardLogitsAll: {
        const float* logits = run(work.tokens, true);
        const size_t n = work.tokens.size() * size_t(lm_head_->vocab());
        result_logits_.resize(n);
        hipError_t e = hipMemcpyAsync(result_logits_.data(), logits, n * sizeof(float),
                                      hipMemcpyDeviceToHost, resources_->stream());
        if (e == hipSuccess) e = hipStreamSynchronize(resources_->stream());
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

void Model::finish_initialization(std::span<const std::unique_ptr<Model>> peers,
                                  const ncclUniqueId& prefill_id) {
  if (world_size_ <= 1) {
    if (cfg_.decode_graph) initialize_decode_graph();
    return;
  }
  if (peers.size() != static_cast<size_t>(world_size_))
    fail("peer count does not match world_size");

  // Every rank has loaded; its slab and flag addresses are stable. Peer access
  // was enabled before allocations. Only pointer tables are copied here,
  // not peer data; remote reads happen in the allreduce kernels.
  std::vector<float*> slabs;
  std::vector<uint64_t*> flags;
  for (const auto& peer : peers) {
    if (peer->ar_slot_stride_ != ar_slot_stride_ || peer->blocks_.size() != blocks_.size())
      fail("peer contribution slab layout does not match");
    slabs.push_back(peer->ar_contributions_);
    flags.push_back(peer->ar_flag_);
  }
  auto ck = [](hipError_t e, const char* what) {
    if (e != hipSuccess)
      fail(std::string(what) + ": " + hipGetErrorString(e));
  };
  ck(hipMalloc(&ar_slabs_dev_, sizeof(float*) * world_size_),
     "ar_slabs_dev hipMalloc");
  ck(hipMemcpy(ar_slabs_dev_, slabs.data(), sizeof(float*) * world_size_,
               hipMemcpyHostToDevice), "ar_slabs_dev hipMemcpy");
  ck(hipMalloc(&ar_flags_dev_, sizeof(uint64_t*) * world_size_),
     "ar_flags_dev hipMalloc");
  ck(hipMemcpy(ar_flags_dev_, flags.data(), sizeof(uint64_t*) * world_size_,
               hipMemcpyHostToDevice), "ar_flags_dev hipMemcpy");
  // Every worker binds its own device and enters initialization concurrently.
  // The ID was created once by Engine after validating the runtime library.
  const auto e = ncclCommInitRank(&prefill_comm_, world_size_, prefill_id, rank_);
  if (e != ncclSuccess) fail(std::string("prefill communicator: ") + ncclGetErrorString(e));
  if (cfg_.decode_graph) initialize_decode_graph();
}

void Model::initialize_decode_graph() {
  auto ck = [](hipError_t e, const char* what) {
    if (e != hipSuccess) fail(std::string(what) + ": " + hipGetErrorString(e));
  };
  const auto stream = resources_->stream();
  if (world_size_ > 1)
    ck(hipMalloc(&decode_epoch_base_, sizeof(uint64_t)), "decode epoch hipMalloc");

  // Warm only the vendor head, not a full forward (a rank waiting in its
  // allreduce before peers are submitted would deadlock). No model state
  // is advanced. All blocks' custom kernels are precompiled.
  ck(hipMemsetAsync(trunk_->io(), 0, size_t(hidden_dim_) * sizeof(float), stream),
     "decode warm input");
  (void)lm_head_->project_last(trunk_->io(), 1);
  ck(hipStreamSynchronize(stream), "decode warm sync");
  trunk_->begin_pass(1);
  const uint64_t old_epoch = ar_epoch_;
  ar_epoch_ = 0;  // captured kernel values are ordinals, not absolute epochs
  ck(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "decode begin capture");
  recording_decode_ = true;
  try {
    decode_logits_ = run_compute(1, false);
  } catch (...) {
    recording_decode_ = false;
    hipGraph_t discarded = nullptr;
    (void)hipStreamEndCapture(stream, &discarded);
    if (discarded) (void)hipGraphDestroy(discarded);
    throw;
  }
  recording_decode_ = false;
  ck(hipStreamEndCapture(stream, &decode_graph_), "decode end capture");
  ck(hipGraphInstantiate(&decode_exec_, decode_graph_, nullptr, nullptr, 0),
     "decode instantiate");
  // Recording executes host bookkeeping but no model kernels. Restore host
  // cursors, and initialize private device cursors on the execution stream.
  ar_epoch_ = old_epoch;
  for (auto& b : blocks_)
    if (auto* sb = dynamic_cast<block::StatefulBlock*>(b.get())) sb->reset_state();
  ck(hipStreamSynchronize(stream), "decode cursor reset sync");
}

void Model::load(int rank, int world_size,
                 const engine::Descriptor& descriptor,
                 const loader::LocalFileDataLoader& data_loader,
                 engine::LoadStats& st) {
  // Resources must be constructed in the worker thread so hipSetDevice is
  // thread-local.
  rank_ = rank;
  world_size_ = world_size;

  // Cross-device peer access for the device-flag allreduce: enable BEFORE
  // any allocation on this rank, serialized across all ranks, with every
  // rank held until all enables complete (so no hipMalloc anywhere overlaps
  // an enable). This ordering is load-bearing: post-load enables trip a
  // deterministic HSA `MappedHandleAllowedAgent` assert on state created
  // during load (arena/KV/etc.), and concurrent enable + hipMalloc asserts
  // too (research/p2p-allreduce README). Enabled early, peer access is a
  // sticky device-pair capability that covers all subsequent allocations.
  if (world_size > 1) {
    static std::atomic<int> peer_enable_turn{0};
    while (peer_enable_turn.load(std::memory_order_acquire) != rank) {
    }
    (void)hipSetDevice(rank);
    for (int j = 0; j < world_size; ++j) {
      if (j == rank) continue;
      int can = 0;
      hipError_t e = hipDeviceCanAccessPeer(&can, rank, j);
      if (e == hipSuccess && can) {
        e = hipDeviceEnablePeerAccess(j, 0);
        // Already-enabled is benign (re-init in one process).
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
          fail(std::string("hipDeviceEnablePeerAccess: ") +
               hipGetErrorString(e));
      }
    }
    if (peer_enable_turn.fetch_add(1, std::memory_order_acq_rel) + 1 ==
        world_size)
      peer_enable_turn.store(0, std::memory_order_release);  // re-initable
    while (peer_enable_turn.load(std::memory_order_acquire) != 0 &&
           peer_enable_turn.load(std::memory_order_acquire) != world_size) {
    }
  }

  resources_ = std::make_unique<Resources>(rank, world_size);

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
          resources_->stream(), blob_to_tensor(blob, vocab, d, "embedder"),
          cfg_.max_prefill_chunk);
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
          resources_->stream(), std::move(pre_norms), std::move(final_norm), eps,
          d, cfg_.max_prefill_chunk);
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
          resources_->stream(), blob_to_tensor(blob, vocab, d, "language_head"),
          cfg_.max_prefill_chunk, resources_->handle());
      st.construct_ms += tc.ms();
    });
  }

  st.read_ms = data_loader.read_ms();
  st.hash_ms = data_loader.hash_ms();
  st.bytes_read = data_loader.bytes_read();
  st.hash_verifications = data_loader.hash_verifications();
  st.blocks = blocks_.size();

  // TP contribution slots and my fine-grained flag must exist before the
  // load barrier and finish_initialization(). One fixed slot per block; no slab
  // for TP=1. Flags start at zero and increase once per collective.
  if (world_size_ > 1) {
    if (hidden_dim_ % 4 != 0)
      fail("P2P allreduce requires hidden_dim divisible by 4");
    ar_slot_stride_ = size_t(cfg_.max_prefill_chunk) * hidden_dim_;
    const size_t slab_bytes = blocks_.size() * ar_slot_stride_ * sizeof(float);
    size_t free = 0, total = 0;
    hipError_t e = hipMemGetInfo(&free, &total);
    if (e != hipSuccess)
      fail(std::string("contribution slab hipMemGetInfo: ") + hipGetErrorString(e));
    if (slab_bytes > free)
      fail("contribution slab needs " + std::to_string(slab_bytes) +
           " bytes, only " + std::to_string(free) +
           " free — reduce max_prefill_chunk");
    e = hipMalloc(&ar_contributions_, slab_bytes);
    if (e != hipSuccess)
      fail(std::string("contribution slab hipMalloc: ") + hipGetErrorString(e));
    e = hipExtMallocWithFlags(reinterpret_cast<void**>(&ar_flag_),
                                         sizeof(uint64_t),
                                         hipDeviceMallocFinegrained);
    if (e != hipSuccess)
      fail(std::string("ar_flag hipExtMallocWithFlags: ") +
           hipGetErrorString(e));
    e = hipMemset(ar_flag_, 0, sizeof(uint64_t));
    if (e != hipSuccess)
      fail(std::string("ar_flag hipMemset: ") + hipGetErrorString(e));
  }
}

}  // namespace alice::gpu
