// gpu::Resources impl (PROTOTYPE 2026-09 — NOT frozen). See resources.hpp.

#include "resources.hpp"

#include <stdexcept>
#include <string>
#include <mutex>

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

namespace alice::gpu {

namespace {

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("gpu::Resources: " + what);
}

#define HIP_CHECK(x)                                                            \
  do {                                                                         \
    const hipError_t e_ = (x);                                                 \
    if (e_ != hipSuccess)                                                      \
      fail(std::string(#x) + " failed: " + hipGetErrorString(e_));             \
  } while (0)

size_t total_free_device_bytes() {
  size_t free = 0, total = 0;
  if (hipMemGetInfo(&free, &total) != hipSuccess) return 0;
  return free;
}

}  // namespace

Resources::Resources(int rank, int world_size)
    : rank_(rank), world_size_(world_size) {
  int n = 0;
  HIP_CHECK(hipGetDeviceCount(&n));
  if (rank < 0 || rank >= n)
    fail("no ROCm device " + std::to_string(rank) + " (found " +
         std::to_string(n) + ")");
  HIP_CHECK(hipSetDevice(rank));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, rank));
  arch_ = prop.gcnArchName;  // "gfx90a:sramecc+:xnack-" -> "gfx90a"
  arch_ = arch_.substr(0, arch_.find(':'));
  if (rocblas_create_handle(&handle_) != rocblas_status_success)
    fail("rocblas_create_handle failed");
  // Note: hipDeviceEnablePeerAccess is NOT called here. Enabling the HSA peer
  // mapping racing the 4 concurrent per-rank hipMallocs during weight load
  // trips a runtime MappedHandleAllowedAgent assert on this stack. The
  // hand-rolled allreduce uses hipMemcpyPeer, which the HIP runtime services
  // without the persistent peer-access mapping (and without the race).
}

Resources::~Resources() {
  for (auto& [p, n] : ledger_)
    if (p) (void)hipFree(p);
  for (auto& [name, np] : named_)
    if (np.first) (void)hipFree(np.first);
  if (scratch_) (void)hipFree(scratch_);
  if (handle_) (void)rocblas_destroy_handle(handle_);
}

void* Resources::alloc(size_t bytes) {
  if (arena_) {
    // Explicit start-alignment: enforced invariant (the size-padded advance
    // below keeps the offset aligned inductively, but this holds regardless
    // of how offset arrived). Up to kDeviceAlign-1 bytes may be skipped —
    // absorbed by the +kDeviceAlign slack in init_arena.
    arena_offset_ = (arena_offset_ + kDeviceAlign - 1) & ~(kDeviceAlign - 1);
    const size_t padded = (bytes + kDeviceAlign - 1) & ~(kDeviceAlign - 1);
    if (arena_offset_ + padded > arena_cap_)
      fail("arena overflow: need " + std::to_string(arena_offset_ + padded) +
           " (offset " + std::to_string(arena_offset_) + " + alloc " +
           std::to_string(padded) + "), cap " + std::to_string(arena_cap_) +
           " — device_bytes() underestimates the block's actual allocations");
    void* p = arena_ + arena_offset_;
    arena_offset_ += padded;
    return p;
  }
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess) {
    const size_t free = total_free_device_bytes();
    fail("hipMalloc(" + std::to_string(bytes) + ") failed; " +
         std::to_string(free >> 20) + " MiB free on rank " +
         std::to_string(rank_) + " device, ledger holds " +
         std::to_string(ledger_.size()) + " allocations");
  }
  ledger_.push_back({p, bytes});
  return p;
}

void Resources::init_arena(size_t bytes) {
  if (arena_) fail("arena already initialized");
  // +kDeviceAlign slack: the explicit start-alignment in alloc() can skip up
  // to kDeviceAlign-1 bytes at the front; size the arena to absorb it so the
  // device_bytes() sum always fits.
  const size_t alloc_bytes = bytes + kDeviceAlign;
  size_t free = 0, total = 0;
  if (hipMemGetInfo(&free, &total) != hipSuccess)
    fail("hipMemGetInfo failed before arena alloc");
  if (alloc_bytes > free)
    fail("arena pre-allocation " + std::to_string(alloc_bytes) + " bytes (" +
         std::to_string(alloc_bytes >> 20) + " MiB; " + std::to_string(bytes >> 20) +
         " MiB device_bytes + " + std::to_string(kDeviceAlign) +
         " align slack) exceeds " + std::to_string(free >> 20) +
         " MiB free on rank " + std::to_string(rank_) +
         " device — reduce max_seq_len / max_prefill_chunk, or use a leaner model");
  if (hipMalloc(reinterpret_cast<void**>(&arena_), alloc_bytes) != hipSuccess)
    fail("arena hipMalloc(" + std::to_string(alloc_bytes) +
         ") failed (fit-check passed; concurrent allocation?)");
  arena_cap_ = alloc_bytes;
  ledger_.push_back({arena_, alloc_bytes});  // freed at teardown as one owned region
}

void Resources::init_scratch(size_t bytes) {
  if (scratch_) fail("scratch already initialized");
  scratch_bytes_ = bytes;
  if (bytes == 0) return;
  if (hipMalloc(reinterpret_cast<void**>(&scratch_), bytes) != hipSuccess)
    fail("scratch hipMalloc(" + std::to_string(bytes) + ") failed");
  // Poison fill (quiet NaN): read-before-first-write becomes visible.
  if (hipMemsetD32(scratch_, 0x7fc00000, uint32_t(bytes / 4)) != hipSuccess)
    fail("scratch poison fill failed");
}

void* Resources::named(const std::string& name, size_t bytes) {
  for (auto& [n, np] : named_) {
    if (n == name) {
      if (np.second != bytes)
        fail("named buffer '" + name + "' requested at " +
             std::to_string(bytes) + " bytes but exists at " +
             std::to_string(np.second) + " bytes");
      return np.first;
    }
  }
  // Named buffers keep their own alloc path (separate hipMalloc, off-arena)
  // per the clean buffer-type separation.
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess)
    fail("named hipMalloc(" + std::to_string(bytes) + ") failed for '" + name + "'");
  ledger_.push_back({p, bytes});
  named_.push_back({name, {p, bytes}});
  return p;
}

}  // namespace alice::gpu
