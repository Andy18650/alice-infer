#pragma once

#include "alice/tensor.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace alice {

class InferenceContext;
class TensorSource;
struct ComponentSpec;

// A pure compute block (attention, FFN, MoE, ...). Owns its weights and
// config; contains NO residual logic and never sees the trunk — wiring is
// the Backbone's job, which is why block types never multiply by backbone
// scheme. Immutable after construction; per-sequence runtime state lives
// in InferenceContext.
class Block {
public:
  virtual ~Block() = default;
  virtual void forward(TensorView in, TensorView out, InferenceContext& ctx) const = 0;
};

// Type-name -> factory registry. A factory resolves its weights from the
// (already validated) source. The manifest names only types and shapes;
// implementation selection (naive / BLAS / fused) happens behind the
// factory, per the backend registry — the model never knows.
class BlockRegistry {
public:
  using Factory =
      std::function<std::unique_ptr<Block>(const ComponentSpec&, const TensorSource&)>;

  static BlockRegistry& instance();

  void register_type(std::string name, Factory f);  // throws on duplicate
  std::unique_ptr<Block> create(const ComponentSpec& spec,
                                const TensorSource& src) const;  // throws on unknown type

private:
  std::map<std::string, Factory> factories_;
};

}  // namespace alice
