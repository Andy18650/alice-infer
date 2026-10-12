#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:?usage: run.sh OUT PHASE [extra benchmark options]}
phase=${2:?phase required};shift 2
mkdir -p "$out/tmp" "$out/cache"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" AMD_COMGR_CACHE_DIR="$out/cache/comgr"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
cp "$root/temp/gemm-granularity/bench" "$out/bench"
cp "$here/bench.hip" "$here/ck.hip" "$here/simt.hip" "$here/api.hpp" "$here/weights.hpp" "$here/build.sh" "$out/"
sha256sum "$out/bench" > "$out/executable.sha256"
hipcc --version > "$out/compiler.txt"
numactl --cpunodebind=0 --membind=0 numactl --show > "$out/placement.txt" 2>&1
printf '%q ' numactl --cpunodebind=0 --membind=0 "$out/bench" --phase "$phase" "$@" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
numactl --cpunodebind=0 --membind=0 "$out/bench" --phase "$phase" "$@" > "$out/results.csv" 2> "$out/run.log"
