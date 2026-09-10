#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "alice/tensor.hpp"

namespace alice {

// Hardware classes (design §7.1). One concrete `final` implementation per
// hardware class per abstract block class. CPU_GENERIC is the naive reference
// implementation and the permanent test oracle. Detection + fallback chains
// live in the per-abstract-class factories (added with the implementations).
enum class HardwareClass { CPU_GENERIC, CPU_AVX2, CPU_AVX512, CPU_AMX, GPU_ROCBLAS, AUTO };

// True when a usable ROCm GPU is present (device 0, "the first GPU").
// Compiles to a constant false in builds without ROCm. Loader/hardware
// auto-selection uses this; the EngineConfig override still forces a class.
// (Interim mechanism — the loader is slated for an overhaul.)
bool gpu_available();

// Block interface (design §7.1).
//
// Boundary contract: `io` is FP32 host memory, [S, d] row-major, IN-PLACE —
// the Block reads its input from io on entry and writes its output into the
// same io before returning. The dtype is static at type level (F32View);
// the single static io buffer is owned by the Trunk.
class Block {
 public:
  virtual ~Block() = default;
  virtual void forward(F32View io) = 0;
};

// A Block that owns non-weight state (attention KV cache, SSM recurrent
// state). State is append-only; position is a private counter (design §6).
class StatefulBlock : public Block {
 public:
  // New session: zero/empty the state, keep the allocation. Needed now.
  virtual void reset_state() = 0;

  // Snapshot/stash support is a FUTURE REVISION (see docs/session-api.md).
  // Declared now with default traps so the interface is stable.
  virtual std::vector<std::byte> serialize_state() const {
    throw std::logic_error("serialize_state: future revision");
  }
  virtual size_t snapshot_create() {
    throw std::logic_error("snapshot_create: future revision");
  }
  virtual void snapshot_rollback(size_t /*snapshot_id*/) {
    throw std::logic_error("snapshot_rollback: future revision");
  }
};

}  // namespace alice
