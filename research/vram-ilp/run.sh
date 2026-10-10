#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
out="${1:?output directory}"
original="${2:-$root/temp/vram-ceiling/final-1/ceiling}"
rounds="${3:-9}"
phase="${4:-all}"
if [ ! -x "$root/temp/vram-ilp/probe" ] || [ "$root/temp/vram-ilp/probe" -ot "$here/probe.hip" ]; then
  echo "Missing/stale probe: run $here/build.sh first" >&2; exit 1
fi
mkdir -p "$out"
cp "$here/probe.hip" "$out/probe.hip"
cp "$root/temp/vram-ilp/probe" "$out/probe"
cp "$original" "$out/original"
llvm-objdump --offloading "$out/original" > "$out/offloading-original.txt"
object="$out/original.0.hipv4-amdgcn-amd-amdhsa--gfx90a"
{
  echo "O3 HIP C++20 gfx90a; HIP device 0; CPU+memory NUMA node 0; no model/context"
  if [ "$phase" = trace ]; then
    echo "phase=trace; 1 GiB; seven per-wave timestamp samples; no holds/barriers/atomics"
  else
    echo "phase=$phase; 1/2/4 GiB shuffled jointly; $rounds rounds; 12 reads/batch; no clock overrides"
  fi
  git -C "$root" rev-parse HEAD
  hipcc --version
  sha256sum "$out/probe.hip" "$out/probe" "$out/original" "$object"
} > "$out/metadata.txt"
rocm-smi --showbus --showuse --showclocks --showtemp --showpower > "$out/smi-before.txt"
numactl --cpunodebind=0 --membind=0 "$out/probe" "$out" "$object" "$rounds" "$phase" > "$out/stdout.txt" 2> "$out/run.log" &
pid=$!
python3 "$here/../vram-ceiling/monitor.py" "$pid" "$out" &
monitor=$!
status=0
wait "$pid" || status=$?
wait "$monitor" || true
rocm-smi --showbus --showuse --showclocks --showtemp --showpower > "$out/smi-after.txt"
[ "$status" -eq 0 ] || exit "$status"
if [ "$phase" != trace ]; then python3 "$here/summarize.py" "$out"; fi
echo "saved $out"
