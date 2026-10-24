#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "/home/andyhu/projects/inference/alice-infer/research/native-gemm/data.hpp"
namespace baseline {
#include "/home/andyhu/projects/inference/alice-infer/research/native-gemm/results/final/source/data.hpp"
}

int main() {
    struct Shape { int m, n, k; };
    const Shape shapes[] = {
        {1, 1, 0}, {3, 7, 1}, {17, 23, 7}, {31, 35, 8},
        {129, 131, 33}, {129, 131, 65}, {255, 257, 127},
        {257, 259, 129}, {1, 3, 65535},
    };
    size_t packed_vectors = 0;
    size_t matrix_pairs = 0;
    for (const auto shape : shapes) {
        for (uint32_t changed_seed : {0u, 1u}) {
            baseline::Data old(shape.m, shape.n, shape.k,
                               1729 + changed_seed, 2718 + changed_seed);
            TestMatrices current(shape.m, shape.n, shape.k,
                                 1729 + changed_seed, 2718 + changed_seed);
            if (old.a != current.row_major_a_units || old.b != current.row_major_b_units)
                throw std::runtime_error("logical matrix generation changed");
            for (int unit = -4; unit <= 4; ++unit)
                if (baseline::half_bits(unit) != half_bits_from_unit(unit))
                    throw std::runtime_error("FP16 encoding changed");
            for (int tile : {128, 256}) {
                for (int stage : {32, 64}) {
                    for (bool operand_b : {false, true}) {
                        if (old.pack(operand_b, tile, stage)
                            != current.pack_operand(operand_b, tile, stage))
                            throw std::runtime_error("packed input bits changed");
                        ++packed_vectors;
                    }
                    for (int row : {0, shape.m - 1}) {
                        for (int column : {0, shape.n - 1}) {
                            if (old.ref(row, column) != current.reference_dot(row, column)
                                || old.ref(row, column, std::min(shape.k, stage))
                                   != current.reference_dot(row, column, std::min(shape.k, stage)))
                                throw std::runtime_error("row-major reference changed");
                        }
                    }
                }
            }
            ++matrix_pairs;
        }
    }
    std::printf("matrix_pairs=%zu packed_vectors=%zu bitwise_equal=1\n",
                matrix_pairs, packed_vectors);
}
