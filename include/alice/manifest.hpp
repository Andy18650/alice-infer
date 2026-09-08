#pragma once

#include "alice/safetensors.hpp"
#include "alice/tensor.hpp"
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace alice {

class ManifestError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Scalar parameters (per-component or model-level). Nested structures
// (e.g. RoPE scaling tables) are TBD — flat scalars cover day one.
using ParamValue = std::variant<int64_t, double, bool, std::string>;
using Params = std::map<std::string, ParamValue>;

// One declared weight: either a named tensor in the source, or an alias
// of another component's weight (tied embeddings).
struct WeightRef {
  std::string name;      // empty iff this is an alias
  std::string alias_of;  // role path, e.g. "embedder/table"; empty iff named
  DType dtype = DType::BF16;
  uint32_t rows = 0, cols = 0;
};

struct ComponentSpec {
  std::string type;
  std::map<std::string, WeightRef> weights;  // role -> ref
  Params params;
};

struct BackboneSpec {
  std::string type;  // "add"; future: "hc", "attn_res"
  Params params;
};

// The model description: a single, total contract between the engine and
// a weight source. Hand-written for now; a generator from HF layouts comes
// later. See validate() for the two-way check.
struct Manifest {
  uint32_t format_version = 0;
  Params params;  // model-level scalars
  BackboneSpec backbone;
  ComponentSpec embedder;
  std::vector<ComponentSpec> blocks;
  ComponentSpec lm_head;

  static Manifest parse(std::string_view json_text);  // throws ManifestError

  // Total two-way validation against a weight source. Returns *all*
  // errors (empty == OK):
  //  - every declared weight exists in the source with matching shape/dtype
  //  - every tensor in the source is claimed exactly once (aliases excepted)
  //  - aliases resolve to an existing, named weight
  std::vector<std::string> validate(const TensorSource& source) const;
};

}  // namespace alice
