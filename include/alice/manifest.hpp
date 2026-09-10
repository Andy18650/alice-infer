#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "alice/safetensors.hpp"
#include "alice/tensor.hpp"

namespace alice {

// The manifest is the TOTAL CONTRACT between a weight source and the engine
// (design §3): it declares model composition, per-Block types, and every
// weight by {name, shape, dtype} (or {alias_of} for tied tensors).
// validate() performs the two-way audit before any compute happens.

struct WeightSpec {
  std::string name;      // safetensors tensor name (empty iff alias)
  std::string alias_of;  // aliased tensor name (empty iff direct)
  uint32_t rows = 0;
  uint32_t cols = 0;
  DType dtype = DType::BF16;

  bool is_alias() const { return !alias_of.empty(); }
};

struct RopeScaling {  // llama3-type scaling; presence flagged in ModelConfig
  float factor = 1.0f;
  float low_freq_factor = 1.0f;
  float high_freq_factor = 1.0f;
  uint32_t original_max_position_embeddings = 0;
};

struct ModelConfig {
  uint32_t hidden_size = 0;
  uint32_t intermediate_size = 0;
  uint32_t num_layers = 0;
  uint32_t num_heads = 0;
  uint32_t num_kv_heads = 0;
  uint32_t head_dim = 0;
  uint32_t vocab_size = 0;
  float rms_norm_eps = 1e-5f;
  float rope_theta = 500000.0f;
  bool has_rope_scaling = false;
  RopeScaling rope_scaling;
  // Token metadata (frontend concern; engine ignores them). Optional.
  uint32_t bos_token_id = 128000;
  std::vector<uint32_t> eos_token_ids;
};

struct BlockSpec {
  std::string type;  // "gqa_attention_bf16" | "dense_ffn_bf16" (registry grows)
  std::map<std::string, WeightSpec> weights;  // role -> spec
};

struct Manifest {
  std::string model_name;
  ModelConfig config;

  WeightSpec token_embedding;  // embedder
  std::string trunk_type;      // "rmsnorm_add"
  std::vector<WeightSpec> pre_norms;  // one per block, execution order
  WeightSpec final_norm;
  std::vector<BlockSpec> blocks;  // execution order
  WeightSpec lm_head;

  // Both throw std::runtime_error on malformed input.
  static Manifest parse(const std::string& path);
  static Manifest parse_string(const std::string& json_text);
};

// Two-way audit of the manifest against an opened weight source (single- or
// multi-shard). Returns ALL problems found (missing tensor, shape/dtype
// mismatch, duplicate claim, unclaimed tensor — globally across shards,
// bad alias, config inconsistency, unknown block type or role, pre_norm
// count mismatch). Empty vector == the contract holds.
std::vector<std::string> validate(const Manifest& m, const safetensors::Files& f);

}  // namespace alice
