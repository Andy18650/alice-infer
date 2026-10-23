#pragma once

#include <cstdint>
#include <vector>

// Deterministic test inputs. The kernel itself does not generate its operands.
inline uint32_t mix_bits(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

inline int input_unit(bool operand_b, int row, int column, uint32_t seed) {
    const uint32_t key = uint32_t(row + 1) * 0x9e3779b9u
                       ^ uint32_t(column + 1) * 0x85ebca6bu
                       ^ seed
                       ^ (operand_b ? 0xc8013ea4u : 0xa341316cu);
    return int(mix_bits(key) % 9) - 4;
}

// Encode integer/16 exactly as FP16, for test units in [-4, 4]. Positive
// magnitudes 0..4 map to 0000, 2c00, 3000, 3200, 3400 respectively.
inline uint16_t half_bits_from_unit(int unit) {
    const int magnitude = unit < 0 ? -unit : unit;
    uint16_t magnitude_bits;
    if (magnitude == 0) {
        magnitude_bits = 0;
    } else if (magnitude <= 2) {
        magnitude_bits = uint16_t(0x2800 + (magnitude << 10));
    } else {
        magnitude_bits = uint16_t(0x2c00 + (magnitude << 9));
    }
    return magnitude_bits | (unit < 0 ? 0x8000 : 0);
}

// Original matrices are kept in logical row-major coordinates for an oracle
// independent of the GPU's packed layout. Values are integer units of 1/16.
struct TestMatrices {
    int rows;
    int columns;
    int reduction;
    std::vector<int8_t> row_major_a_units;  // A[rows, reduction]
    std::vector<int8_t> row_major_b_units;  // B[reduction, columns]

    TestMatrices(int m, int n, int k, uint32_t seed_a, uint32_t seed_b)
        : rows(m), columns(n), reduction(k),
          row_major_a_units(size_t(m) * k),
          row_major_b_units(size_t(k) * n) {
        for (int row = 0; row < rows; ++row) {
            for (int inner = 0; inner < reduction; ++inner) {
                row_major_a_units[size_t(row) * reduction + inner] =
                    int8_t(input_unit(false, row, inner, seed_a));
            }
        }
        for (int inner = 0; inner < reduction; ++inner) {
            for (int column = 0; column < columns; ++column) {
                row_major_b_units[size_t(inner) * columns + column] =
                    int8_t(input_unit(true, inner, column, seed_b));
            }
        }
    }

    // Physical order:
    // [StageK chunk][half-CTA panel][K8 fragment][micro-pair][lane][4 dwords]
    //
    // A panel contains rows; B contains columns. Each 16-byte lane vector
    // interleaves two 32x8 MFMA fragments. Words 0/1 belong to the first
    // microfragment and words 2/3 to the second. Each word holds two FP16s.
    // Padding is zero, including all M/N/K tails.
    std::vector<uint32_t> pack_operand(bool operand_b, int tile, int stage_k) const {
        const int extent = operand_b ? columns : rows;
        const int padded_extent = (extent + tile - 1) / tile * tile;
        const int rows_or_columns_per_panel = tile / 2;
        const int panel_count = padded_extent / rows_or_columns_per_panel;
        const int stage_count = (reduction + stage_k - 1) / stage_k;
        const int pairs_per_fragment = tile / 128;
        const size_t words_per_panel = size_t(rows_or_columns_per_panel) * stage_k / 2;

        std::vector<uint32_t> packed(size_t(stage_count) * panel_count * words_per_panel);
        for (int stage = 0; stage < stage_count; ++stage) {
            for (int panel = 0; panel < panel_count; ++panel) {
                for (int fragment = 0; fragment < stage_k / 8; ++fragment) {
                    for (int pair = 0; pair < pairs_per_fragment; ++pair) {
                        for (int lane = 0; lane < 64; ++lane) {
                            for (int word = 0; word < 4; ++word) {
                                const int position = panel * rows_or_columns_per_panel
                                                   + (pair * 2 + word / 2) * 32
                                                   + lane % 32;
                                const int first_inner = stage * stage_k + fragment * 8
                                                      + (lane / 32) * 4 + (word % 2) * 2;
                                uint32_t packed_word = 0;
                                for (int half = 0; half < 2; ++half) {
                                    if (position < extent && first_inner + half < reduction) {
                                        const int unit = operand_b
                                            ? row_major_b_units[
                                                  size_t(first_inner + half) * columns + position]
                                            : row_major_a_units[
                                                  size_t(position) * reduction + first_inner + half];
                                        packed_word |= uint32_t(half_bits_from_unit(unit))
                                                     << (16 * half);
                                    }
                                }

                                const size_t offset =
                                    (size_t(stage) * panel_count + panel) * words_per_panel
                                    + size_t(fragment * pairs_per_fragment + pair) * 256
                                    + lane * 4 + word;
                                packed[offset] = packed_word;
                            }
                        }
                    }
                }
            }
        }
        return packed;
    }

    // limit >= 0 is used only by the resident-first-fragment diagnostic.
    // Ordinary GEMM always uses the entire logical reduction dimension.
    float reference_dot(int row, int column, int limit = -1) const {
        int64_t integer_dot = 0;
        for (int inner = 0; inner < (limit < 0 ? reduction : limit); ++inner) {
            integer_dot += int(row_major_a_units[size_t(row) * reduction + inner])
                         * int(row_major_b_units[size_t(inner) * columns + column]);
        }
        return float(double(integer_dot) / 256.0);
    }
};
