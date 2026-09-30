// The block-factory registry. With self-registration (design §15), the
// registry is a function-local static map populated by each concrete
// factory TU's static initializer — there is no central table enumerating
// the built-in types. See block_factory.hpp for the retention note
// (factory TUs must be force-linked by consumers of the registry).

#include "block_factory.hpp"

#include <map>
#include <string>

namespace alice::block {

namespace {

// The single registry map. Function-local static => constructed on first
// use, which register_factory() forces (so it is live before any
// registrar's static init completes). SIoD-safe.
std::map<std::string, ResolveFn*>& registry_mut() {
  static std::map<std::string, ResolveFn*> r;
  return r;
}

}  // namespace

void register_factory(const std::string& type, ResolveFn* fn) {
  registry_mut()[type] = fn;
}

const std::map<std::string, ResolveFn*>& registry() {
  return registry_mut();
}

}  // namespace alice::block
