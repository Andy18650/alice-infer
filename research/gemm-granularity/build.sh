#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out="$root/temp/gemm-granularity"
mkdir -p "$out" "$out/tmp" "$out/cache"
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache"
common=(-O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra -pthread)
# Retire only our generated candidate objects; historical executables live in
# snapshotted run directories. Explicit symbols allow multiple K tile sizes.
rm -f "$out"/ck_*.o
for spec in \
 '16 16 64 1 1 16 register_16x16' \
 '16 32 64 1 2 16 register_16x32' \
 '16 16 128 1 1 16 register_16x16_k128' \
 '16 32 128 1 2 16 register_16x32_k128' \
 '16 32 256 1 2 16 register_16x32_k256' \
 '16 64 64 1 4 16 register_16x64' \
 '32 32 32 2 2 16 register_32x32' \
 '64 64 64 2 2 16 register_64x64' \
 '64 128 64 2 2 16 register_64x128' \
 '128 128 64 2 2 32 register_128x128' \
 '128 256 64 2 2 32 register_128x256' \
 '256 256 64 2 2 32 register_256x256'; do
  read -r m n k wm wn wt reg <<< "$spec"
  hipcc "${common[@]}" -DTILE_M="$m" -DTILE_N="$n" -DTILE_K="$k" \
    -DWARP_M="$wm" -DWARP_N="$wn" -DWARP_TILE="$wt" \
    -DREGISTER="$reg" -DKERNEL_NAME="\"ck_${m}x${n}x${k}\"" \
    -c "$here/ck.hip" -o "$out/ck_${m}x${n}x${k}.o"
done
hipcc "${common[@]}" -c "$here/simt.hip" -o "$out/simt.o"
hipcc "${common[@]}" -c "$here/bench.hip" -o "$out/bench.o"
hipcc "${common[@]}" "$out"/*.o -lrocblas -o "$out/bench"
printf 'Built %s\n' "$out/bench"
