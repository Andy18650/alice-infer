#!/usr/bin/env bash
# Fresh balanced W8A16 namespace; no measured FP16 binary is rebuilt.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
schedule=${2:-interleaved}
out=${1:-"$root/temp/native-gemm/fp8-latency-host-bridge/$schedule"}
NATIVE_FP8_B=1 NATIVE_FP8_SCHEDULE="$schedule" bash "$here/build.sh" 128 "$out" gemm 32
out=$(cd "$out" && pwd)
hipcc -O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee \
    -Wall -Wextra -I"$out" "$here/verify_fp8.hip" -o "$out/verify-fp8"
sha256sum "$here/verify_fp8.hip" "$here/data_fp8.hpp" "$here/data.hpp" \
    "$here/generate_fp8.py" "$here/generate.py" "$out/compute.hpp" \
    "$out/verify-fp8" > "$out/conversion-hashes.txt"
