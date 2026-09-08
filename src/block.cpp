#include "alice/block.hpp"

#include "alice/manifest.hpp"

#include <stdexcept>

namespace alice {

BlockRegistry& BlockRegistry::instance() {
  static BlockRegistry r;
  return r;
}

void BlockRegistry::register_type(std::string name, Factory f) {
  if (!factories_.emplace(std::move(name), std::move(f)).second)
    throw std::invalid_argument("BlockRegistry: duplicate block type '" + name + "'");
}

std::unique_ptr<Block> BlockRegistry::create(const ComponentSpec& spec,
                                             const TensorSource& src) const {
  const auto it = factories_.find(spec.type);
  if (it == factories_.end())
    throw std::invalid_argument("BlockRegistry: unknown block type '" + spec.type + "'");
  return it->second(spec, src);
}

}  // namespace alice
