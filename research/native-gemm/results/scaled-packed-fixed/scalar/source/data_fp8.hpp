#pragma once

#include <stdexcept>
#include "data.hpp"

// Finite E4M3FN, bias7. Independent integer decoder in units of 1/512.
inline int fp8_quanta(uint8_t code) {
    const int exponent = (code >> 3) & 15;
    const int fraction = code & 7;
    if (exponent == 15 && fraction == 7) {
        throw std::runtime_error("NaN outside finite E4M3FN contract");
    }
    const int magnitude = exponent ? (8 + fraction) << (exponent - 1) : fraction;
    return code & 128 ? -magnitude : magnitude;
}

inline uint16_t fp8_reference_half(uint8_t code) {
    const int exponent = (code >> 3) & 15;
    const int fraction = code & 7;
    const uint16_t sign = uint16_t(code & 128) << 8;
    if (exponent) {
        (void)fp8_quanta(code); // Reject the two excluded NaNs.
        return sign | uint16_t((exponent + 8) << 10) | uint16_t(fraction << 7);
    }
    if (!fraction) {
        return sign;
    }
    const int leading = fraction >= 4 ? 2 : fraction >= 2 ? 1 : 0;
    return sign | uint16_t((6 + leading) << 10)
                | uint16_t((fraction - (1 << leading)) << (10 - leading));
}

inline uint8_t unit_as_fp8(int unit) {
    constexpr uint8_t positive[] = {0, 0x18, 0x20, 0x24, 0x28};
    const int magnitude = unit < 0 ? -unit : unit;
    return positive[magnitude] | (unit < 0 ? 128 : 0);
}

struct Fp8Matrices : TestMatrices {
    bool finite_pattern = false;
    uint32_t pattern_seed;

    Fp8Matrices(int m, int n, int k, uint32_t seed_a, uint32_t seed_b)
        : TestMatrices(m, n, k, seed_a, seed_b), pattern_seed(seed_b) {}

    void set_finite_pattern(bool enabled) {
        finite_pattern = enabled;
        if (!enabled) {
            return;
        }
        if (reduction > 64) {
            throw std::runtime_error("finite-code exact oracle requires K<=64");
        }
        // |A|=1/16 and K<=64 keep every partial sum on the exact FP32 grid:
        // 64*448/16*8192 < 2^24. All254 finite B encodings can participate.
        for (size_t index = 0; index < row_major_a_units.size(); ++index) {
            row_major_a_units[index] = (mix_bits(uint32_t(index)) & 1) ? 1 : -1;
        }
    }

    uint8_t b_code(int inner, int column) const {
        if (!finite_pattern) {
            return unit_as_fp8(row_major_b_units[size_t(inner) * columns + column]);
        }
        const unsigned index = unsigned((size_t(inner) * columns + column) * 73
                                        + pattern_seed) % 254;
        return uint8_t(index < 127 ? index : index + 1);
    }

    // A: [stage][128-row tile][q][64-row half][lane][4 FP16 dwords].
    // B: [stage][128-col tile][q][lane][4 FP8 dwords]. Each B dword
    // holds four K values for one32-column microfragment.
    std::vector<uint32_t> pack_operand(bool operand_b, int tile, int stage_k) const {
        if (tile != 128 || stage_k != 32) {
            throw std::runtime_error("FP8 experiment supports128x128x32 only");
        }
        const int extent = operand_b ? columns : rows;
        const int panels = (extent + 127) / 128;
        const int stages = (reduction + 31) / 32;
        const int panel_words = operand_b ? 1024 : 2048;
        std::vector<uint32_t> packed(size_t(stages) * panels * panel_words);
        for (int stage = 0; stage < stages; ++stage) {
            for (int panel = 0; panel < panels; ++panel) {
                const size_t base = (size_t(stage) * panels + panel) * panel_words;
                for (int q = 0; q < 4; ++q) {
                    for (int lane = 0; lane < 64; ++lane) {
                        if (operand_b) {
                            for (int word = 0; word < 4; ++word) {
                                const int column = panel * 128 + word * 32 + lane % 32;
                                uint32_t value = 0;
                                for (int byte = 0; byte < 4; ++byte) {
                                    const int inner = stage * 32 + q * 8 + (lane / 32) * 4 + byte;
                                    if (inner < reduction && column < columns) {
                                        value |= uint32_t(b_code(inner, column)) << (byte * 8);
                                    }
                                }
                                packed[base + q * 256 + lane * 4 + word] = value;
                            }
                        } else {
                            for (int half_panel = 0; half_panel < 2; ++half_panel) {
                                for (int word = 0; word < 4; ++word) {
                                    const int row = panel * 128 + half_panel * 64
                                                  + (word / 2) * 32 + lane % 32;
                                    uint32_t value = 0;
                                    for (int half = 0; half < 2; ++half) {
                                        const int inner = stage * 32 + q * 8 + (lane / 32) * 4
                                                        + (word % 2) * 2 + half;
                                        if (row < rows && inner < reduction) {
                                            const int unit = row_major_a_units[size_t(row) * reduction + inner];
                                            value |= uint32_t(half_bits_from_unit(unit)) << (half * 16);
                                        }
                                    }
                                    packed[base + q * 512 + half_panel * 256 + lane * 4 + word] = value;
                                }
                            }
                        }
                    }
                }
            }
        }
        return packed;
    }

    float reference_dot(int row, int column, int limit = -1) const {
        if (!finite_pattern) {
            return TestMatrices::reference_dot(row, column, limit);
        }
        int64_t sum = 0;
        for (int inner = 0; inner < (limit < 0 ? reduction : limit); ++inner) {
            sum += int64_t(row_major_a_units[size_t(row) * reduction + inner])
                   * fp8_quanta(b_code(inner, column));
        }
        return float(double(sum) / 8192.0);
    }
};
