#pragma once

// gpu::Resources (PROTOTYPE 2026-09 — NOT frozen): the Model's GPU resources
// — device binding, the BLAS handle, and the device-memory allocator
// (ledger-tracked). Merges the former gpu_rocm::Context (device + handle)
// and modelfmt::DeviceAllocator (named buffers + scratch) into one
// session-scoped object owned by Model.
//
// NOT involved in forward(): component constructors call alloc() / borrow
// handle()/scratch() at construction; forward() uses the borrowed values,
// never the Resources object. Session-scoped: destroyed after the components
// (Model's member order puts resources_ ahead of blocks_), freeing
// everything in the ledger.
//
// This first cut is on-demand allocation (hipMalloc per alloc, ledger-tracked
// + OOM context). The arena optimization — plan via factory device_bytes(),
// one hipMalloc, bump-carve, plus the upfront §15 fit-check — lands next, on
// this foundation. Scratch (shared prefill partials) and named buffers keep
// their own allocation paths (separate hipMallocs) for clean buffer-type
// separation (user direction: a handful of allocation calls is fine).

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "blas.hpp"

namespace alice::gpu {

// Arena alignment: every device allocation is carved aligned + padded to
// this boundary (256 B satisfies hipMalloc's default + every GPU load the
// kernels do). One fixed arena alignment keeps device_bytes() a single
// predictable number per block.
inline constexpr size_t kDeviceAlign = 256;

class Resources {
 public:
  // Validates the device exists, binds it as the calling thread's current
  // device (hipSetDevice is per-thread), creates the BLAS handle. Throws
  // std::runtime_error loudly.
  explicit Resources(int device);
  ~Resources();
  Resources(const Resources&) = delete;
  Resources& operator=(const Resources&) = delete;

  int device() const { return device_; }
  BlasHandle handle() const { return handle_; }
  const std::string& arch() const { return arch_; }  // e.g. "gfx90a" (reporting)

  // Device-memory allocation. If init_arena() was called, carves an aligned
  // + padded slice from the arena (bump); otherwise hipMallocs on demand.
  // Either way the returned ptr is Resources-owned (freed at teardown) and
  // kDeviceAlign-aligned; callers must NOT hipFree it. Throws on failure
  // (arena overflow / OOM, with free-MiB + ledger context).
  void* alloc(size_t bytes);

  // Arena pre-allocation (the optimization): one hipMalloc of `bytes`, then
  // alloc() bump-carves from it. Does a fit-check against hipMemGetInfo free
  // first (fails loud, naming the bytes + free). device_bytes() summed over
  // the blocks is the intended input. Optional — without it, alloc() is
  // on-demand. Not re-callable.
  void init_arena(size_t bytes);

  // Shared scratch (the v4 prefill partials buffer, design §14). Sized +
  // allocated once via init_scratch; NaN-poison-filled so read-before-first-
  // write is observable in tests. One buffer, reused by every attention
  // block per forward (produced + consumed within each forward).
  void init_scratch(size_t bytes);
  float* scratch() const { return scratch_; }

  // Named-buffer interner (workspace pattern): same name + size -> same ptr
  // (intended sharing; the strictly-sequential model loop makes time-sharing
  // safe); same name + different size -> loud failure. The name is the
  // sharing contract. (Reserved for future shared transients; unused in v1.)
  void* named(const std::string& name, size_t bytes);

 private:
  int device_ = 0;
  std::string arch_;
  BlasHandle handle_ = nullptr;
  float* scratch_ = nullptr;
  size_t scratch_bytes_ = 0;
  // Arena (optional, via init_arena): one hipMalloc'd region, bump-carved.
  std::byte* arena_ = nullptr;
  size_t arena_cap_ = 0;
  size_t arena_offset_ = 0;
  // Owned allocations freed at teardown (on-demand alloc + named + the arena
  // base). scratch_ is freed separately (it carries the §14 poison invariant).
  std::vector<std::pair<void*, size_t>> ledger_;
  std::vector<std::pair<std::string, std::pair<void*, size_t>>> named_;
};

}  // namespace alice::gpu
