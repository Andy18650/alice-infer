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
  if (ar_flag_) (void)hipFree(ar_flag_);
  if (ar_flags_dev_) (void)hipFree(ar_flags_dev_);
  if (ar_ios_dev_) (void)hipFree(ar_ios_dev_);
  if (ar_scratch_) (void)hipFree(ar_scratch_);
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
// Read-gather dataflow (producers only ever write their own io locally;
// consumers do all cross-GCD traffic as float4 remote READS, the measured
// fast path): every rank sums all ranks' io in slot order j = 0..ws-1, so
// the result is bitwise-identical on all ranks.
//
// Dispatch by payload (= S * hidden_dim_ * 4B), measured crossover at ~64
// KB (flagprobe, ws=4):
//   <= 64 KB (decode / tiny S): fused single-block kernel — produce
//       barrier, gather into registers, consume barrier, in-place write.
//       1 launch, ~18 us at ws=4 32 KB (vs RCCL ~53 us sync-per-call).
//   > 64 KB (prefill): 4-launch pipe — flagbar, wide gather -> scratch,
//       flagbar, copy scratch -> io. Wide grid scales LSU bandwidth
//       (single block is one CU's memory pipe: 5-12x slower at MB sizes).
void Model::allreduce(float* io, uint32_t S) {
  if (world_size_ <= 1) return;
  const uint32_t n = S * hidden_dim_;
  const uint64_t vprod = ar_epoch_ + 1;
  const uint64_t vcons = ar_epoch_ + 2;
  ar_epoch_ += 2;
  constexpr uint32_t kFusedMaxFloats = 16384;  // 64 KB (measured crossover)
  if (n <= kFusedMaxFloats) {
    kernels::ar_fused_f4(io, ar_ios_dev_, ar_flag_, ar_flags_dev_, world_size_,
                         rank_, n, vprod, vcons);
  } else {
    kernels::ar_flagbar(ar_flag_, ar_flags_dev_, world_size_, rank_, vprod);
    kernels::ar_gather_f4(ar_ios_dev_, ar_scratch_, world_size_, n);
    kernels::ar_flagbar(ar_flag_, ar_flags_dev_, world_size_, rank_, vcons);
    const hipError_t e = hipMemcpyAsync(io, ar_scratch_, size_t(n) * 4,
                                        hipMemcpyDeviceToDevice, 0);
    if (e != hipSuccess)
      fail(std::string("allreduce copy: ") + hipGetErrorString(e));
  }
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

void Model::initialize_peers(std::span<const std::unique_ptr<Model>> peers) {
  if (world_size_ <= 1) return;
  if (peers.size() != static_cast<size_t>(world_size_))
    fail("peer count does not match world_size");

  // Every rank has loaded; its io and flag addresses are stable. Peer access
  // was enabled before allocations. Only pointer tables are copied here,
  // not peer data; remote reads happen in the allreduce kernels.
  std::vector<float*> ios;
  std::vector<uint64_t*> flags;
  for (const auto& peer : peers) {
    ios.push_back(peer->trunk_->io());
    flags.push_back(peer->ar_flag_);
  }
  auto ck = [](hipError_t e, const char* what) {
    if (e != hipSuccess)
      fail(std::string(what) + ": " + hipGetErrorString(e));
  };
  ck(hipMalloc(&ar_ios_dev_, sizeof(float*) * world_size_),
     "ar_ios_dev hipMalloc");
  ck(hipMemcpy(ar_ios_dev_, ios.data(), sizeof(float*) * world_size_,
               hipMemcpyHostToDevice), "ar_ios_dev hipMemcpy");
  ck(hipMalloc(&ar_flags_dev_, sizeof(uint64_t*) * world_size_),
     "ar_flags_dev hipMalloc");
  ck(hipMemcpy(ar_flags_dev_, flags.data(), sizeof(uint64_t*) * world_size_,
               hipMemcpyHostToDevice), "ar_flags_dev hipMemcpy");
  ck(hipMalloc(&ar_scratch_, size_t(cfg_.max_prefill_chunk) * hidden_dim_ *
                              sizeof(float)), "ar_scratch hipMalloc");
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

  // Device-flag allreduce: my fine-grained flag word (world_size > 1). Must
  // exist before the load barrier and initialize_peers(); zeroed
  // (epoch counters start at 0, first allreduce uses values 1 and 2).
  if (world_size_ > 1) {
    hipError_t e = hipExtMallocWithFlags(reinterpret_cast<void**>(&ar_flag_),
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
