// Named buffers are fixed-size inter-layer state, separate from the arena.
#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include "../src/gpu/resources.hpp"

TEST(GpuResources, NamedBuffersShareFixedSizeStateOutsideArena) {
  alice::gpu::Resources res(0);
  res.init_arena(alice::gpu::kDeviceAlign);
  void* private_buffer = res.alloc(sizeof(uint32_t));
  void* shared = res.named("shared_layer_state", sizeof(uint32_t));
  void* other = res.named("other_layer_state", sizeof(uint32_t));
  ASSERT_NE(shared, private_buffer);
  ASSERT_NE(other, shared);

  const uint32_t value = 12345;
  ASSERT_EQ(hipMemcpy(shared, &value, sizeof(value), hipMemcpyHostToDevice),
            hipSuccess);
  EXPECT_EQ(res.named("shared_layer_state", sizeof(uint32_t)), shared);
  // Reject both growth and shrinkage without replacing the shared pointer.
  EXPECT_THROW(res.named("shared_layer_state", 2 * sizeof(uint32_t)),
               std::runtime_error);
  EXPECT_THROW(res.named("shared_layer_state", sizeof(uint16_t)),
               std::runtime_error);
  EXPECT_EQ(res.named("shared_layer_state", sizeof(uint32_t)), shared);

  ASSERT_EQ(hipMemset(private_buffer, 0, sizeof(uint32_t)), hipSuccess);
  ASSERT_EQ(hipMemset(other, 0, sizeof(uint32_t)), hipSuccess);
  uint32_t observed = 0;
  ASSERT_EQ(hipMemcpy(&observed, shared, sizeof(observed), hipMemcpyDeviceToHost),
            hipSuccess);
  EXPECT_EQ(observed, value);
  // Resources tears down arena and named allocations through distinct owners.
}
