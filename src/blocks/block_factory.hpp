#pragma once

// BlockFactory (design §15): the loader's block-construction seam. It is
// the *derivative* of a parsed spec — captured layout knowledge (derived
// byte size, scratch need, stream-interface signature) plus the construct
// step that reconciles a staged chunk and runs the block constructor. It
// carries derivatives, not the spec itself (hence the rename from the
// earlier "ResolvedSpec").
//
// Concrete factories live next to their block implementations
// (src/blocks/gpu_rocm/*_factory.{hpp,cpp}); each (type, version) is one
// factory class. The registry maps a type name to a resolve function that
// produces a factory from a BlockSpecDecl.

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "alice/engine_config.hpp"
#include "../chunk.hpp"
#include "block.hpp"

namespace alice::modelfmt {
struct BlockSpecDecl;
class DeviceAllocator;
}  // namespace alice::modelfmt
namespace alice::gpu_rocm {
struct Context;
}  // namespace alice::gpu_rocm

namespace alice::block {

class BlockFactory {
 public:
  virtual ~BlockFactory() = default;

  // Layout-derived byte size of the block's weight record (config → bytes).
  virtual uint64_t derived_bytes() const = 0;

  // Scratch requirement: a pure function of (type, version, config, engine
  // knobs). The loader takes the max over factories and allocates once.
  // [open: scratch is a kernel-impl property, not a (type,version) one —
  // moves to the implementation with the factory round; learnings §5.]
  virtual size_t scratch_bytes(const EngineConfig& ec) const = 0;

  // Composition invariant (docs/model-loader.md §1): factories may state
  // the interface they present to the stream (e.g. "io:2048"); empty = no
  // opinion. The loader checks all present signatures agree.
  virtual std::string interface_sig() const { return {}; }

  // Reconciles the chunk length against the derived length (exact), slices
  // tensor views, runs the block constructor (private H2D), returns the
  // block. The chunk is consumed; the caller frees it after return.
  virtual std::unique_ptr<Block> construct(Chunk chunk,
                                           modelfmt::DeviceAllocator& alloc,
                                           const gpu_rocm::Context& ctx,
                                           const EngineConfig& ec) const = 0;
};

// resolve: BlockSpecDecl -> BlockFactory. Type-version validation + config
// parsing happen here; the returned factory is the captured layout result.
using ResolveFn = std::unique_ptr<BlockFactory>(const modelfmt::BlockSpecDecl&);

// Self-registration: each concrete factory TU registers its (type -> resolve
// fn) pair via a static initializer. The registry map is a function-local
// static (construction-on-first-use), so registration is SIoD-safe against
// the registrars' static init.
//
// Retention note: a factory TU's only externally-visible effect is the
// registration side effect, so in a static archive the linker drops it
// unless force-linked. Consumers of the registry must link the factory
// objects with --whole-archive (or equivalent); see research/model-format.
void register_factory(const std::string& type, ResolveFn* fn);

// The type registry: type name -> resolve function. Unknown type in a file
// = loud load error (design §15: "no entry = loud failure").
const std::map<std::string, ResolveFn*>& registry();

}  // namespace alice::block
