#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/gemm-compute-bound"}
mode=${2:-native}
mkdir -p "$out/tmp" "$out/cache"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra -pthread)
case "$mode" in
 native) ;;
 ckflags) flags+=(-mllvm -enable-post-misched=0 -mllvm -enable-noalias-to-md-conversion=0) ;;
 *) echo 'mode must be native or ckflags' >&2;exit 1;;
esac
printf '%q ' hipcc "${flags[@]}" > "$out/compiler-flags.txt";printf '\n' >> "$out/compiler-flags.txt"
for spec in \
 '1 128 128 64 2 2 32 add_v1_128' \
 '3 128 128 64 2 2 32 add_v3_128' \
 '4 128 128 64 2 2 32 add_v4_128' \
 '3 256 128 64 2 2 32 add_v3_256x128' \
 '3 256 256 64 2 2 32 add_v3_256' \
 '4 256 256 32 2 2 32 add_v4_256' \
 '3 128 128 64 2 2 16 add_v3_128_w16' \
 '3 256 256 64 2 2 16 add_v3_256_w16' \
 '2 128 128 64 2 2 32 add_mem_128'; do
 read -r pipe m n k wm wn wt reg <<< "$spec"
 name="v${pipe}_${m}x${n}x${k}";[[ "$wt" == 32 ]] || name+="_w$wt"
 hipcc "${flags[@]}" -DPIPE="$pipe" -DTM="$m" -DTN="$n" -DTK="$k" -DWM="$wm" -DWN="$wn" -DWT="$wt" -DWK=16 \
   -DREGISTER="$reg" -DKERNEL_NAME="\"$name\"" \
   -c "$here/kernel.hip" -o "$out/$reg.o"
done
for spec in '128 128 add_v3_128_wk8' '256 256 add_v3_256_wk8' '256 128 add_v3_256x128_wk8'; do
 read -r m n reg <<< "$spec"
 hipcc "${flags[@]}" -DPIPE=3 -DTM="$m" -DTN="$n" -DTK=64 -DWM=2 -DWN=2 -DWT=32 -DWK=8 \
   -DREGISTER="$reg" -DKERNEL_NAME="\"v3_${m}x${n}x64_wk8\"" \
   -c "$here/kernel.hip" -o "$out/$reg.o"
done
for spec in '128 128 16 add_classic_128' '256 256 32 add_classic_256'; do
 read -r m n xdl reg <<< "$spec"
 hipcc "${flags[@]}" -DTM="$m" -DTN="$n" -DXDL="$xdl" -DREGISTER="$reg" \
   -DKERNEL_NAME="\"classic_${m}x${n}x64\"" -c "$here/classic.hip" -o "$out/$reg.o"
done
for stages in 4 8; do
 hipcc "${flags[@]}" -DSTAGES="$stages" -DREGISTER="add_direct$stages" \
   -DKERNEL_NAME="\"direct_$stages\"" -c "$here/direct.hip" -o "$out/add_direct$stages.o"
done
hipcc "${flags[@]}" -c "$here/bench.hip" -o "$out/bench.o"
hipcc "${flags[@]}" "$out"/*.o -lrocblas -o "$out/bench"
echo "Built $out/bench ($mode)"
