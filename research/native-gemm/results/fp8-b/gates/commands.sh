# Commands used for the independent conversion/packing and full-finite gates.
# Run from the repository root. Numerical checks are outside performance timing.
numactl --cpunodebind=0 --membind=0 temp/native-gemm/fp8-b/s32-t128/verify-fp8
for k in 1 7 8 9 31 32 33 63 64; do
    numactl --cpunodebind=0 --membind=0 temp/native-gemm/fp8-b/s32-t128/probe \
        --m 129 --n 131 --k "$k" --b-pattern 1 --rounds 2 --warmup-ms 0
done
