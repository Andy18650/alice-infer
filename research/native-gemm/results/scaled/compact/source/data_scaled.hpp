#pragma once

#include <cmath>
#include "data_fp8.hpp"

// Independent logical K128/N128 scale table. Every M tile shares the same B
// scales. Group partials in the test domain are exact FP32; std::fma specifies
// FP32 rounding of each scaled update, independently of packed GPU addressing.
struct ScaledMatrices : Fp8Matrices {
    std::vector<float> scales;
    bool scaling_enabled = true;

    ScaledMatrices(int m, int n, int k, uint32_t seed_a, uint32_t seed_b)
        : Fp8Matrices(m, n, k, seed_a, seed_b),
          scales(size_t((k + 127) / 128) * ((n + 127) / 128), 1.0f) {}

    void set_scales(int pattern, bool enabled) {
        scaling_enabled = enabled;
        for (size_t index = 0; index < scales.size(); ++index) {
            const uint32_t value = mix_bits(uint32_t(index) ^ 0x91e10da5u);
            switch (pattern) {
            case 0: scales[index] = 1.0f; break;
            case 1: scales[index] = std::ldexp(1.0f, int(value % 4) - 2); break;
            case 2: scales[index] = float(0.25 + double(value % 10001) / 5713.0); break;
            case 3: scales[index] = value % 3 ? float((value % 7) + 1) / 8.0f : 0.0f; break;
            default: throw std::runtime_error("unknown scale pattern");
            }
        }
    }

    float reference_dot(int row, int column, int limit = -1) const {
        if (!scaling_enabled) return Fp8Matrices::reference_dot(row, column, limit);
        const int end = limit < 0 ? reduction : limit;
        float result = 0;
        const int scale_columns = (columns + 127) / 128;
        for (int begin = 0; begin < end; begin += 128) {
            int64_t sum = 0;
            for (int inner = begin; inner < std::min(begin + 128, end); ++inner) {
                const int a = row_major_a_units[size_t(row) * reduction + inner];
                const int b = finite_pattern ? fp8_quanta(b_code(inner, column))
                    : int(row_major_b_units[size_t(inner) * columns + column]) * 32;
                sum += int64_t(a) * b;
            }
            const float partial = float(double(sum) / 8192.0);
            result = std::fma(partial, scales[size_t(begin / 128) * scale_columns + column / 128], result);
        }
        return result;
    }
};
