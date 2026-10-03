#pragma once

// GPU-resident Model (design §3): the public entry point for the
// device-resident execution path. The block loop runs entirely on the
// device; token ids are the only regular host<->device traffic.
//
// Shape A (user direction): Model owns its components directly as value
// members. This header therefore includes the component headers and RCCL.
// gpu::Model is an internal engine class, not a stable ABI, so the compile-
// time cost is acceptable.

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include <rccl/rccl.h>

#include "alice/engine_config.hpp"
#include "../blocks/block.hpp"
#include "embedder.hpp"
#include "lm_head.hpp"
#include "resources.hpp"
#include "trunk.hpp"

namespace alice {

namespace loader {
class LocalFileDataLoader;  // content-addressed bytes source (src/data_loader.hpp)
}  // namespace loader

namespace engine {
struct Descriptor;  // parsed .model descriptor (src/engine.hpp)
struct LoadStats;   // load timing/throughput (src/engine.hpp)
}  // namespace engine

namespace gpu {

class Model {
 public:
  // Parsed .model descriptor + content-addressed bytes source -> Model.
  // The model-format load path: constructs the device model in place from
  // the descriptor's content-addressed blocks + members (same component
  // constructors as the manifest path; arena covers the block loop only).
  // Defined in src/gpu/model.cpp. Throws on any framing, parse, hash, or
  // length failure.
  static std::unique_ptr<Model> load_from_format(
      const engine::Descriptor& descriptor,
      const loader::LocalFileDataLoader& data_loader, const EngineConfig& ec,
      engine::LoadStats& st);

  ~Model();

  // Run forward for these token ids without D2H or sampling. Used for
  // intermediate prefill chunks where only the final state matters.
  void forward_run(std::span<const int32_t> tokens);

  std::vector<float> forward_logits(std::span<const int32_t> tokens);      // D2H [V]
  std::vector<float> forward_logits_all(std::span<const int32_t> tokens);  // D2H [S*V]
  void reset();  // new session: clear all stateful blocks
  uint32_t vocab_size() const;

 private:
  Model();

  // Work queue for one worker thread. Forward/reset requests are submitted
  // from the main thread; the worker executes them and signals completion.
  struct WorkQueue {
    enum class Kind {
      ForwardRun,
      ForwardLogits,
      ForwardLogitsAll,
      Reset,
      Shutdown,
    };

    std::mutex mutex;
    std::condition_variable work_cv;
    std::condition_variable done_cv;

    bool has_work = false;
    bool done = false;
    Kind kind = Kind::Shutdown;
    std::vector<int32_t> tokens;

    std::vector<float> result_logits;
    std::exception_ptr error;

    // Called by the worker to wait for the next work item.
    Kind wait_for_work(std::vector<int32_t>& out_tokens) {
      std::unique_lock<std::mutex> lock(mutex);
      work_cv.wait(lock, [&] { return has_work; });
      has_work = false;
      out_tokens = std::move(tokens);
      return kind;
    }

    // Called by the worker to signal completion of a work item.
    void signal_done() {
      {
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
      }
      done_cv.notify_one();
    }
  };

  struct Rank {
    std::unique_ptr<Resources> resources_;
    std::unique_ptr<Embedder> embedder_;
    std::unique_ptr<RmsNormAddTrunk> trunk_;
    std::vector<std::unique_ptr<block::Block>> blocks_;
    std::unique_ptr<LmHead> lm_head_;
    ncclComm_t nccl_comm = nullptr;

    WorkQueue queue;
    std::thread worker;
  };

  EngineConfig cfg_;
  int tp_degree_ = 1;

  // Load parameters (valid until loading completes).
  const engine::Descriptor* load_descriptor_ = nullptr;
  const loader::LocalFileDataLoader* load_data_loader_ = nullptr;
  EngineConfig load_ec_;
  engine::LoadStats* load_stats_ = nullptr;

  // Global load coordination.
  std::mutex load_mutex_;
  std::condition_variable load_cv_;
  int loaded_count_ = 0;

  // RCCL collective state. For tp_degree_==1 no communicator is created.
  ncclUniqueId nccl_id_{};
  bool nccl_id_valid_ = false;

  uint32_t hidden_dim_ = 0;  // cached for AllReduce count

  std::vector<std::unique_ptr<Rank>> ranks_;
  uint32_t vocab_size_ = 0;

  Rank& rank(int r) { return *ranks_.at(r); }

  // AllReduce the fp32 stream across ranks. For tp_degree_==1 this is a
  // no-op. Each rank contributes its partial sum in-place; the result is
  // written back to the same buffer.
  void allreduce(int r, float* io, uint32_t S);

  const float* run(int r, std::span<const int32_t> tokens, bool all_positions);

  // Submit the same work to every rank's queue and wait for all to complete.
  // If any rank has a queued error, rethrow. Returns rank 0's queue so the
  // caller can read the result.
  WorkQueue& submit_all(WorkQueue::Kind kind,
                        std::span<const int32_t> tokens = {});

  void load(int r);

  static void worker_loop(Model* model, int rank);
};

}  // namespace gpu
}  // namespace alice
