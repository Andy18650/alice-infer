// Diagnostic only: compare two VGPR source-pair positions in resident MFMA.
// Every lane uses the same exact FP16 value (1/16); this does not establish
// that register banking is irrelevant for all data/register layouts.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define HIP(expression)                                                        \
    do {                                                                       \
        const auto status = (expression);                                      \
        if (status != hipSuccess) {                                            \
            throw std::runtime_error(hipGetErrorString(status));                \
        }                                                                      \
    } while (0)

using F16 = float __attribute__((ext_vector_type(16)));

template<bool Same>
__global__ __launch_bounds__(256) void banks(float* out, uint64_t* ts, int groups) {
    F16 c0 = {}, c1 = {}, c2 = {}, c3 = {};
    uint64_t b, e;

    // Initialize both candidate B pairs identically; only their register
    // positions differ in the measured loops below.
    asm volatile(
        "v_mov_b32 v0, 0x2c002c00\n\t"
        "v_mov_b32 v1, v0\n\t"
        "v_mov_b32 v2, v0\n\t"
        "v_mov_b32 v3, v0\n\t"
        "v_mov_b32 v4, v0\n\t"
        "v_mov_b32 v5, v0\n\t"
        "s_memtime %0\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=&s"(b), "+a"(c0), "+a"(c1), "+a"(c2), "+a"(c3)
        :
        : "v0", "v1", "v2", "v3", "v4", "v5", "memory");

    if constexpr (Same) {
        asm volatile(
            ".Lbank_same_%=:\n\t"
            "v_mfma_f32_32x32x8f16 %0, v[0:1], v[4:5], %0\n\t"
            "v_mfma_f32_32x32x8f16 %1, v[0:1], v[4:5], %1\n\t"
            "v_mfma_f32_32x32x8f16 %2, v[0:1], v[4:5], %2\n\t"
            "v_mfma_f32_32x32x8f16 %3, v[0:1], v[4:5], %3\n\t"
            "s_sub_u32 %4, %4, 1\n\t"
            "s_cmp_lg_u32 %4, 0\n\t"
            "s_cbranch_scc1 .Lbank_same_%="
            : "+a"(c0), "+a"(c1), "+a"(c2), "+a"(c3), "+&s"(groups)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "scc", "memory");
    } else {
        asm volatile(
            ".Lbank_diff_%=:\n\t"
            "v_mfma_f32_32x32x8f16 %0, v[0:1], v[2:3], %0\n\t"
            "v_mfma_f32_32x32x8f16 %1, v[0:1], v[2:3], %1\n\t"
            "v_mfma_f32_32x32x8f16 %2, v[0:1], v[2:3], %2\n\t"
            "v_mfma_f32_32x32x8f16 %3, v[0:1], v[2:3], %3\n\t"
            "s_sub_u32 %4, %4, 1\n\t"
            "s_cmp_lg_u32 %4, 0\n\t"
            "s_cbranch_scc1 .Lbank_diff_%="
            : "+a"(c0), "+a"(c1), "+a"(c2), "+a"(c3), "+&s"(groups)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "scc", "memory");
    }

    asm volatile(
        "s_nop 15\n\t"
        "s_nop 15\n\t"
        "s_nop 15\n\t"
        "s_nop 15\n\t"
        "s_memtime %0\n\t"
        "s_waitcnt lgkmcnt(0)"
        : "=&s"(e), "+a"(c0), "+a"(c1), "+a"(c2), "+a"(c3)
        :
        : "memory");

    if (threadIdx.x % 64 == 0) {
        ts[threadIdx.x / 64 * 2] = b;
        ts[threadIdx.x / 64 * 2 + 1] = e;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        out[threadIdx.x * 64 + j] = c0[j];
        out[threadIdx.x * 64 + 16 + j] = c1[j];
        out[threadIdx.x * 64 + 32 + j] = c2[j];
        out[threadIdx.x * 64 + 48 + j] = c3[j];
    }
}

int main() try {
    HIP(hipSetDevice(0));
    float* output;
    uint64_t* timestamps;
    HIP(hipMalloc(&output, 256 * 64 * 4));
    HIP(hipMalloc(&timestamps, 64));
    std::puts("same_pair_position,round,cycles_per_four_wave_mfma,valid");

    for (int same_pair_position : {0, 1}) {
        for (int round = 0; round < 4; ++round) {
            if (same_pair_position) {
                banks<true><<<1, 256>>>(output, timestamps, 65536);
            } else {
                banks<false><<<1, 256>>>(output, timestamps, 65536);
            }
            HIP(hipGetLastError());
            HIP(hipDeviceSynchronize());
            uint64_t clocks[8];
            HIP(hipMemcpy(clocks, timestamps, sizeof(clocks), hipMemcpyDeviceToHost));
            std::vector<float> values(256 * 64);
            HIP(hipMemcpy(values.data(), output, values.size() * 4, hipMemcpyDeviceToHost));
            for (float value : values) {
                // Each update contributes 8*(1/16)^2=1/32; 65536 updates/chain.
                if (value != 2048) {
                    throw std::runtime_error("bank control mismatch");
                }
            }
            uint64_t begin = UINT64_MAX;
            uint64_t end = 0;
            for (int wave = 0; wave < 4; ++wave) {
                begin = std::min(begin, clocks[wave * 2]);
                end = std::max(end, clocks[wave * 2 + 1]);
            }
            // Four instructions/loop per wave; do not multiply divisor by waves.
            std::printf("%d,%d,%.9f,1\n",
                        same_pair_position, round, double(end - begin) / 262144);
        }
    }
    (void)hipFree(output);
    (void)hipFree(timestamps);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
