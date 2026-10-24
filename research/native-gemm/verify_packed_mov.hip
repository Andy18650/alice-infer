// Raw-bit gate independent of timestamp ratios or GEMM numerics.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define HIP_CHECK(expr) do { const auto status = (expr); \
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); \
} while (0)

template<bool Correct>
__global__ void copy_pairs(const uint64_t* input, uint64_t* output) {
    const uint64_t bits = input[blockIdx.x];
    const uint32_t low = __builtin_amdgcn_readfirstlane(uint32_t(bits));
    const uint32_t high = __builtin_amdgcn_readfirstlane(uint32_t(bits >> 32));
    const uint64_t source = uint64_t(low) | (uint64_t(high) << 32);
    uint64_t destination;
    if constexpr (Correct) {
        asm volatile("v_pk_mov_b32 %0, %1, %1 op_sel:[0,1]"
                     : "=&v"(destination) : "s"(source));
    } else {
        asm volatile("v_pk_mov_b32 %0, %1, %1"
                     : "=&v"(destination) : "s"(source));
    }
    output[size_t(blockIdx.x) * 64 + threadIdx.x] = destination;
}

int main() try {
    std::vector<uint64_t> input{0, UINT64_MAX, 0x0123456789abcdefull,
                               0x7fc0000080000000ull, 0x00023ebd21b61674ull};
    for (int bit = 0; bit < 64; ++bit) {
        input.push_back(uint64_t(1) << bit);
        input.push_back(~(uint64_t(1) << bit));
    }
    uint64_t state = 0x89abcdef01234567ull;
    for (int index = 0; index < 1024; ++index) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        input.push_back(state);
    }
    std::vector<uint64_t> output(input.size() * 64);
    uint64_t *device_input, *device_output;
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&device_input, input.size() * 8));
    HIP_CHECK(hipMalloc(&device_output, output.size() * 8));
    HIP_CHECK(hipMemcpy(device_input, input.data(), input.size() * 8, hipMemcpyHostToDevice));
    copy_pairs<true><<<input.size(),64>>>(device_input, device_output);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(output.data(), device_output, output.size() * 8, hipMemcpyDeviceToHost));
    for (size_t index = 0; index < output.size(); ++index) {
        if (output[index] != input[index / 64]) throw std::runtime_error("packed raw-bit copy mismatch");
    }
    copy_pairs<false><<<input.size(),64>>>(device_input, device_output);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(output.data(), device_output, output.size() * 8, hipMemcpyDeviceToHost));
    size_t mismatches = 0;
    for (size_t index = 0; index < output.size(); ++index) {
        const uint32_t low = uint32_t(input[index / 64]);
        const uint64_t duplicate = uint64_t(low) | (uint64_t(low) << 32);
        if (output[index] != duplicate) throw std::runtime_error("negative-control behavior changed");
        mismatches += output[index] != input[index / 64];
    }
    if (!mismatches) throw std::runtime_error("negative control failed to reject duplicated low word");
    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
    std::printf("PASS %zu exact64-bit copies; default-selector negative control rejected %zu outputs (duplicated low32)\n",
                output.size(), mismatches);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr,"FAIL %s\n",error.what());
    return 1;
}
