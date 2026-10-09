#!/bin/sh
# Single GCD 0, one process; separate CTA-count and kernel-count axes.
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
out="${1:-$root/temp/kernel-launch/results}"
"$here/build.sh"
mkdir -p "$out"
{
  echo "O3 C++20 gfx90a; GCD 0 only; CPU+memory NUMA node 0; no attention/model"
  echo "per-wave entry/exit core+steady timestamps; no entry barrier"
  git -C "$root" rev-parse HEAD
  hipcc --version
  sha256sum "$here/probe.hip" "$root/temp/kernel-launch/probe"
} > "$out/metadata.txt"
status=0
timeout 45s numactl --cpunodebind=0 --membind=0 \
  "$root/temp/kernel-launch/probe" --sweep "$out" > "$out/summary.csv" 2> "$out/run.log" || status=$?
if [ "$status" -eq 0 ] || [ "$status" -eq 2 ]; then
  echo "saved $out/summary.csv and per-wave traces"
fi
if [ "$status" -eq 2 ]; then
  echo "Runtime accepted an oversubscribed cooperative graph; rows are marked UNEXPECTED_ACCEPTED, not timed as valid cases" >&2
fi
exit "$status"
