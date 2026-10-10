#!/bin/sh
# No production build dependency; single idle GCD, no clock/power changes.
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
out="${1:-$root/temp/vram-ceiling/results}"
device="${2:-0}"
"$here/build.sh"
mkdir -p "$out"
{
  echo "O3 HIP C++20 gfx90a; GCD $device; CPU+memory NUMA node 0; no model/context"
  echo "read-only input, checksum stores; decimal GB/s; no locked clocks"
  git -C "$root" rev-parse HEAD
  hipcc --version
  sha256sum "$here/ceiling.hip" "$root/temp/vram-ceiling/ceiling"
} > "$out/metadata.txt"
cp "$here/ceiling.hip" "$out/ceiling.hip"
cp "$root/temp/vram-ceiling/ceiling" "$out/ceiling"
rocm-smi --showbus --showuse --showclocks --showtemp --showpower > "$out/smi-before.txt"
numactl --cpunodebind=0 --membind=0 "$out/ceiling" "$out" "$device" \
  > "$out/stdout.txt" 2> "$out/run.log" &
pid=$!
python3 "$here/monitor.py" "$pid" "$out" &
monitor=$!
status=0
wait "$pid" || status=$?
wait "$monitor" || true
rocm-smi --showbus --showuse --showclocks --showtemp --showpower > "$out/smi-after.txt"
[ "$status" -eq 0 ] || exit "$status"
python3 "$here/summarize.py" "$out" > "$out/summary.txt"
echo "saved $out"
