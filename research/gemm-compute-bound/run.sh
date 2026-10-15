#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:?usage: run.sh OUT BINARY [benchmark arguments]}
binary=${2:?binary required};shift 2
mkdir -p "$out/tmp" "$out/cache";out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp" XDG_CACHE_HOME="$out/cache" AMD_COMGR_CACHE_DIR="$out/cache/comgr"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
cp "$binary" "$out/bench"
cp "$here"/{bench.hip,kernel.hip,classic.hip,direct.hip,api.hpp,build.sh} "$out/"
sha256sum "$out/bench" > "$out/executable.sha256"
hipcc --version > "$out/compiler.txt"
numactl --cpunodebind=0 --membind=0 numactl --show > "$out/placement.txt" 2>&1
printf '%q ' numactl --cpunodebind=0 --membind=0 "$out/bench" "$@" > "$out/command.txt";printf '\n' >> "$out/command.txt"
numactl --cpunodebind=0 --membind=0 "$out/bench" "$@" > "$out/results.csv" 2> "$out/run.log"
