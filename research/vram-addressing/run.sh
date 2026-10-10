#!/bin/sh
# Run a compiled probe without modifying clock/power/system configuration.
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
out="${1:?output directory}"
phase="${2:?stride|offset|bits|pair|pair-bits|conflict|planes|profile|confirm}"
device="${3:-0}"
payload="${4:-32}"
rounds="${5:-5}"
plane="${6:-8}"
if [ ! -x "$root/temp/vram-addressing/probe" ] || [ "$root/temp/vram-addressing/probe" -ot "$here/probe.hip" ]; then
  echo "Missing/stale probe: run $here/build.sh first" >&2
  exit 1
fi
mkdir -p "$out"
cp "$here/probe.hip" "$out/probe.hip"
cp "$root/temp/vram-addressing/probe" "$out/probe"
{
  echo "O3 HIP C++20 gfx90a; HIP device $device; CPU+memory NUMA node 0; no model/context"
  echo "phase=$phase payload_MiB=$payload rounds=$rounds plane_GiB=$plane; no clock overrides"
  git -C "$root" rev-parse HEAD
  hipcc --version
  sha256sum "$here/probe.hip" "$out/probe"
} > "$out/metadata.txt"
rocm-smi --showbus --showuse --showclocks --showtemp --showpower > "$out/smi-before.txt"
numactl --cpunodebind=0 --membind=0 "$out/probe" "$out" "$phase" "$device" "$payload" "$rounds" "$plane" \
  > "$out/stdout.txt" 2> "$out/run.log" &
pid=$!
python3 "$here/../vram-ceiling/monitor.py" "$pid" "$out" &
monitor=$!
status=0
wait "$pid" || status=$?
wait "$monitor" || true
rocm-smi --showbus --showuse --showclocks --showtemp --showpower > "$out/smi-after.txt"
[ "$status" -eq 0 ] || exit "$status"
python3 "$here/summarize.py" "$out"
echo "saved $out"
