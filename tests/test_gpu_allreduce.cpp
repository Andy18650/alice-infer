// P2P slab collectives: published slots stay immutable, outputs are separate,
// and produced epochs continue across forwards. Peer setup precedes allocation.
#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include "../src/gpu/kernels.hpp"

namespace {
namespace gk = alice::gpu::kernels;

struct RankBuffers {
  int device = 0;
  float* slab = nullptr;
  float* out = nullptr;
  uint64_t* flag = nullptr;
  float** slabs = nullptr;
  uint64_t** flags = nullptr;
  ~RankBuffers() {
    (void)hipSetDevice(device);
    (void)hipFree(slabs);
    (void)hipFree(flags);
    (void)hipFree(out);
    (void)hipFree(slab);
    (void)hipFree(flag);
  }
};

TEST(GpuAllreduce, SlabSlotsPreservedAcrossCollectivesAndForwards) {
  constexpr int ws = 4;
  int count = 0;
  ASSERT_EQ(hipGetDeviceCount(&count), hipSuccess);
  if (count < ws) GTEST_SKIP() << "requires four GPUs";
  for (int r = 0; r < ws; ++r) {
    for (int peer = 0; peer < ws; ++peer) {
      if (peer == r) continue;
      int can = 0;
      ASSERT_EQ(hipDeviceCanAccessPeer(&can, r, peer), hipSuccess);
      if (!can) GTEST_SKIP() << "requires all-to-all peer access";
    }
  }
  for (int r = 0; r < ws; ++r) {
    ASSERT_EQ(hipSetDevice(r), hipSuccess);
    for (int peer = 0; peer < ws; ++peer) {
      if (peer == r) continue;
      const auto e = hipDeviceEnablePeerAccess(peer, 0);
      ASSERT_TRUE(e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled);
    }
  }

  constexpr std::array<uint32_t, 3> sizes{8, 16384, 16388};
  constexpr size_t stride = 16448;  // fixed, float4-aligned; includes padding
  constexpr size_t elements = sizes.size() * stride;
  constexpr size_t bytes = elements * sizeof(float);
  std::array<RankBuffers, ws> ranks;
  for (int r = 0; r < ws; ++r) {
    auto& rank = ranks[r];
    rank.device = r;
    ASSERT_EQ(hipSetDevice(r), hipSuccess);
    ASSERT_EQ(hipMalloc(&rank.slab, bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&rank.out, bytes), hipSuccess);
    ASSERT_EQ(hipExtMallocWithFlags(reinterpret_cast<void**>(&rank.flag),
                                    sizeof(uint64_t), hipDeviceMallocFinegrained),
              hipSuccess);
    ASSERT_EQ(hipMemset(rank.flag, 0, sizeof(uint64_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&rank.slabs, ws * sizeof(float*)), hipSuccess);
    ASSERT_EQ(hipMalloc(&rank.flags, ws * sizeof(uint64_t*)), hipSuccess);
  }
  std::array<float*, ws> slabs;
  std::array<uint64_t*, ws> flags;
  for (int r = 0; r < ws; ++r) {
    slabs[r] = ranks[r].slab;
    flags[r] = ranks[r].flag;
  }
  for (int r = 0; r < ws; ++r) {
    ASSERT_EQ(hipSetDevice(r), hipSuccess);
    ASSERT_EQ(hipMemcpy(ranks[r].slabs, slabs.data(), sizeof(slabs),
                        hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(ranks[r].flags, flags.data(), sizeof(flags),
                        hipMemcpyHostToDevice), hipSuccess);
  }

  for (uint64_t forward = 0; forward < 4; ++forward) {
    std::array<std::vector<float>, ws> inputs;
    const std::vector<float> empty_out(elements, -123.0f);
    for (int r = 0; r < ws; ++r) {
      inputs[r].resize(elements);
      for (size_t i = 0; i < elements; ++i)
        inputs[r][i] = float(forward * 100 + r * 10 + i % 7);
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      ASSERT_EQ(hipMemcpy(ranks[r].slab, inputs[r].data(), bytes,
                          hipMemcpyHostToDevice), hipSuccess);
      ASSERT_EQ(hipMemcpy(ranks[r].out, empty_out.data(), bytes,
                          hipMemcpyHostToDevice), hipSuccess);
    }
    // Queue a whole forward on each rank before visiting the next. Never
    // wait until all are submitted; slots have distinct epochs/fixed offsets.
    for (int r = 0; r < ws; ++r) {
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      auto& rank = ranks[r];
      for (size_t slot = 0; slot < sizes.size(); ++slot) {
        const uint64_t epoch = forward * sizes.size() + slot + 1;
        const size_t offset = slot * stride;
        if (sizes[slot] <= 16384) {
          gk::ar_fused_f4(nullptr, rank.out + offset, rank.slabs, rank.flag, rank.flags,
                          ws, r, sizes[slot], offset, epoch);
        } else {
          gk::ar_flagbar(nullptr, rank.flag, rank.flags, ws, r, epoch);
          gk::ar_gather_f4(nullptr, rank.slabs, rank.out + offset, ws, sizes[slot], offset);
        }
      }
    }
    // This is the engine's explicit all-rank forward completion boundary.
    for (int r = 0; r < ws; ++r) {
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
    }
    for (int r = 0; r < ws; ++r) {
      ASSERT_EQ(hipSetDevice(r), hipSuccess);
      std::vector<float> slab(elements), out(elements);
      ASSERT_EQ(hipMemcpy(slab.data(), ranks[r].slab, bytes,
                          hipMemcpyDeviceToHost), hipSuccess);
      ASSERT_EQ(hipMemcpy(out.data(), ranks[r].out, bytes,
                          hipMemcpyDeviceToHost), hipSuccess);
      EXPECT_EQ(slab, inputs[r]) << "published slab was overwritten";
      for (size_t slot = 0; slot < sizes.size(); ++slot) {
        for (size_t i = 0; i < stride; ++i) {
          const size_t index = slot * stride + i;
          float want = -123.0f;
          if (i < sizes[slot]) {
            want = 0.0f;
            for (int peer = 0; peer < ws; ++peer) want += inputs[peer][index];
          }
          ASSERT_EQ(out[index], want) << "rank " << r << ", slot " << slot
                                      << ", element " << i;
        }
      }
      uint64_t epoch = 0;
      ASSERT_EQ(hipMemcpy(&epoch, ranks[r].flag, sizeof(epoch),
                          hipMemcpyDeviceToHost), hipSuccess);
      EXPECT_EQ(epoch, (forward + 1) * sizes.size());
    }
  }
}
}  // namespace
