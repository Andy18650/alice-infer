#pragma once

// Per-rank GPU Model (design §3). One Model object is owned by each worker
// thread; the Engine handles main-thread coordination and the shared work
// barrier. Model loads one rank's weights, executes one WorkItem at a time,
// and exposes its result/error to the Engine after each barrier.

#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <vector>

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

// One unit of work broadcast to all rank Models by the Engine.
struct WorkItem {
  enum class Kind {
    ForwardRun,
    ForwardLogits,
    ForwardLogitsAll,
    Reset,
    Stop,
  };
  Kind kind = Kind::Stop;
  std::vector<int32_t> tokens;
};

class Model {
 public:
  explicit Model(const ModelConfig& cfg);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // Load this rank's weights from the descriptor and data loader. Must be
  // called once in the worker thread before any execute() call.
  void load(int rank, int world_size, const engine::Descriptor& descriptor,
            const loader::LocalFileDataLoader& data_loader, engine::LoadStats& st);

  // Execute one work item. Called by the worker thread.
  void execute(const WorkItem& work);

  // Complete initialization in the owner worker, after every rank has loaded.
  // Installs device pointer tables for P2P allreduce; not a runtime command.
  void initialize_peers(std::span<const std::unique_ptr<Model>> peers);

  // Result/error accessors (read by the Engine after the completion barrier).
  const std::vector<float>& last_logits() const { return result_logits_; }
  std::exception_ptr error() const { return error_; }
  void set_error(std::exception_ptr e) { error_ = e; }

 private:
  ModelConfig cfg_;

  std::unique_ptr<Resources> resources_;
  std::unique_ptr<Embedder> embedder_;
  std::unique_ptr<RmsNormAddTrunk> trunk_;
  std::vector<std::unique_ptr<block::Block>> blocks_;
  std::unique_ptr<LmHead> lm_head_;

  uint32_t hidden_dim_ = 0;  // cached for AllReduce count

  // Device-flag P2P allreduce state (world_size > 1 only). See allreduce()
  // and research/p2p-allreduce/atomics.md.
  int rank_ = 0;
  int world_size_ = 1;
  uint64_t* ar_flag_ = nullptr;      // my fine-grained flag word (device)
  uint64_t** ar_flags_dev_ = nullptr;  // device table: flag ptr per rank
  float** ar_ios_dev_ = nullptr;     // device table: io ptr per rank
  float* ar_scratch_ = nullptr;      // pipe path: gather result (max_n floats)
  uint64_t ar_epoch_ = 0;            // +2 per allreduce (produce/consume)

  std::vector<float> result_logits_;
  std::exception_ptr error_;

  void allreduce(float* io, uint32_t S);
  const float* run(std::span<const int32_t> tokens, bool all_positions);
};

}  // namespace gpu
}  // namespace alice
