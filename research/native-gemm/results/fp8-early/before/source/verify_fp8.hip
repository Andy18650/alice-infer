// Independent finite-E4M3FN bit oracle and logical-coordinate packing checks.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "data_fp8.hpp"
using F16 = float __attribute__((ext_vector_type(16)));
#include "compute.hpp"

#define CHECK_HIP(expr) do { const auto status = (expr); \
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); \
} while (0)

__global__ void check_blocks(const uint4* input, uint32_t* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const ExpandedFp8 expanded = expand_fp8_block_for_check(input[index]);
        for (int word = 0; word < 8; ++word) {
            output[index * 8 + word] = expanded.words[word];
        }
    }
}

void check_packing() {
    size_t checked = 0;
    for (int k : {0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 129}) {
        for (int pattern : {0, 1}) {
            if (pattern && k > 64) continue;
            Fp8Matrices matrices(129, 131, k, 1730, 2719);
            matrices.set_finite_pattern(pattern);
            for (bool b : {false, true}) {
                const auto packed = matrices.pack_operand(b, 128, 32);
                const int extent = b ? matrices.columns : matrices.rows;
                const int panels = (extent + 127) / 128;
                const int panel_words = b ? 1024 : 2048;
                // Walk logical padded coordinates, derive the producer lane,
                // then inspect packed bits. Not the packer's loop nesting.
                for (int stage = 0; stage < (k + 31) / 32; ++stage) {
                    for (int position = 0; position < panels * 128; ++position) {
                        for (int inner = 0; inner < 32; ++inner) {
                            const int lane = position % 32 + ((inner % 8) / 4) * 32;
                            const int q = inner / 8;
                            const size_t base = (size_t(stage) * panels + position / 128) * panel_words;
                            unsigned actual, expected = 0;
                            const int logical_k = stage * 32 + inner;
                            if (b) {
                                const size_t word = base + q * 256 + lane * 4 + (position % 128) / 32;
                                actual = (packed[word] >> ((inner % 4) * 8)) & 255;
                                if (position < extent && logical_k < k)
                                    expected = matrices.b_code(logical_k, position);
                            } else {
                                const size_t word = base + q * 512 + ((position % 128) / 64) * 256
                                                  + lane * 4 + ((position % 64) / 32) * 2 + (inner % 4) / 2;
                                actual = (packed[word] >> ((inner % 2) * 16)) & 65535;
                                if (position < extent && logical_k < k)
                                    expected = half_bits_from_unit(matrices.row_major_a_units[size_t(position) * k + logical_k]);
                            }
                            if (actual != expected) throw std::runtime_error("CPU packing mismatch");
                            ++checked;
                        }
                    }
                }
            }
        }
    }
    std::printf("PASS independent logical-coordinate packing: %zu padded elements\n", checked);
}

int main() try {
    check_packing();
    CHECK_HIP(hipSetDevice(0));
    constexpr int count = 254 * 16;
    std::vector<uint32_t> input(count * 4), output(count * 8);
    auto finite_code = [](unsigned ordinal) {
        ordinal %= 254;
        return ordinal < 127 ? ordinal : ordinal + 1;
    };
    for (int code_index = 0; code_index < 254; ++code_index) {
        for (int position = 0; position < 16; ++position) {
            const int block = code_index * 16 + position;
            for (int byte = 0; byte < 16; ++byte) {
                const unsigned code = byte == position ? finite_code(code_index)
                    : finite_code(mix_bits(unsigned(block * 16 + byte)));
                input[block * 4 + byte / 4] |= code << ((byte % 4) * 8);
            }
        }
    }
    uint32_t *device_input, *device_output;
    CHECK_HIP(hipMalloc(&device_input, input.size() * 4));
    CHECK_HIP(hipMalloc(&device_output, output.size() * 4));
    CHECK_HIP(hipMemcpy(device_input, input.data(), input.size() * 4, hipMemcpyHostToDevice));
    check_blocks<<<(count + 63) / 64, 64>>>(reinterpret_cast<uint4*>(device_input), device_output, count);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipMemcpy(output.data(), device_output, output.size() * 4, hipMemcpyDeviceToHost));
    for (int block = 0; block < count; ++block) {
        for (int byte = 0; byte < 16; ++byte) {
            const uint8_t code = uint8_t(input[block * 4 + byte / 4] >> ((byte % 4) * 8));
            const uint16_t actual = uint16_t(output[block * 8 + byte / 2] >> ((byte % 2) * 16));
            if (actual != fp8_reference_half(code)) {
                std::fprintf(stderr, "block%d byte%d FP8=%02x expected=%04x actual=%04x\n",
                             block, byte, code, fp8_reference_half(code), actual);
                throw std::runtime_error("GPU in-place FP8 conversion mismatch");
            }
        }
    }
    CHECK_HIP(hipFree(device_input));
    CHECK_HIP(hipFree(device_output));
    std::printf("PASS GPU in-place expansion: %d blocks/%d values; all254 finite codes in every byte position, mixed neighbors, signed zero/subnormals\n",
                count, count * 16);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
}
