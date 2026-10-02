#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "safetensors.hpp"
#include "alice/tensor.hpp"
#include "model_config.hpp"  // ModelConfig (ModelConfig is a block-layer concept;
                            // relocated to its own header; manifest still
                            // carries it via ModelConfig in Manifest::config)

namespace alice {

// The manifest is the TOTAL CONTRACT between a weight source and the engine
// (design §3): it declares model composition, per-Block types, and every
// weight by {name, shape, dtype} (or {alias_of} for tied tensors).
// validate() performs the two-way audit before any compute happens.

struct WeightSpec {
  std::string name;  // safetensors tensor name
  uint32_t rows = 0;
  uint32_t cols = 0;
};
// Manifest diet (2026-09): no per-weight dtype (derived — block types
// encode it, `File::check_as<T>` enforces it at the load boundary, the
// audit checks the FILE's dtype against the derivation); no alias_of —
// tied embeddings are expressed by naming the EMBEDDER's tensor in the
// lm_head role, the one legal shared claim (validate() exempts exactly
// that pair; a mistied lm_head still fails the shape check).

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

  // Parses an engine manifest JSON file.
  //
  // EXPERIMENTAL (pending design review): if the path is a DIRECTORY, it is
  // treated as a Hugging Face model directory — config.json is read and, when
  // "architectures" names "LlamaForCausalLM", the manifest is synthesized
  // from the Llama config format (the same vocabulary as tools/gen_manifest.py,
  // in-engine). This deliberately bypasses the externally-authored-manifest
  // contract (design §3) so the design question can be judged on running code;
  // see the synthesis notes in manifest.cpp.
  //
  // Throws std::runtime_error on: malformed JSON, unopenable path, a directory
  // without config.json, an unsupported architecture, or a config.json with
  // insufficient information (all missing fields reported in one message).
  static Manifest parse(const std::filesystem::path& path);
  static Manifest parse_string(const std::string& json_text);
};

// Two-way audit of the manifest against an opened weight source (single- or
// multi-shard). Returns ALL problems found (missing tensor, shape/dtype
// mismatch, duplicate claim — the tied lm_head exempt, unclaimed tensor —
// globally across shards, config inconsistency, unknown block type or
// role, pre_norm count mismatch). Empty vector == the contract holds.
std::vector<std::string> validate(const Manifest& m, const safetensors::Files& f);

}  // namespace alice
