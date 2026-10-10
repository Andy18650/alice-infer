#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
out="${1:?output directory}"
mode="${2:-controlled}"
phase="${3:-quick}"
rounds="${4:-9}"
case "$mode" in controlled|default) ;; *) echo "mode: controlled|default" >&2; exit 1;; esac
binary="$root/temp/vram-coalescing/$mode"
if [ ! -x "$binary" ] || [ "$binary" -ot "$here/probe.hip" ] || [ "$binary" -ot "$here/readers.hpp" ]; then
  echo "Missing/stale probe: run $here/build.sh first" >&2; exit 1
fi
mkdir -p "$out"
cp "$here/probe.hip" "$out/probe.hip"
cp "$here/readers.hpp" "$out/readers.hpp"
cp "$binary" "$out/probe"
{
  echo "O3 HIP C++20 gfx90a; HIP device 0; CPU+memory NUMA node 0; no model/context"
  echo "mode=$mode phase=$phase rounds=$rounds; 12 reads/batch; no clock overrides"
  if [ "$mode" = controlled ]; then
    echo "-fno-vectorize -fno-slp-vectorize -mllvm -amdgpu-load-store-vectorizer=false"
  fi
  git -C "$root" rev-parse HEAD
  hipcc --version
  sha256sum "$out/probe.hip" "$out/readers.hpp" "$out/probe" "$here/inspect.py"
} > "$out/metadata.txt"
llvm-objdump --offloading "$out/probe" > "$out/offloading.txt"
llvm-objdump -d --demangle "$out/probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" > "$out/disassembly.txt"
python3 "$here/inspect.py" "$out" "$mode" > "$out/inspection.txt"
rocm-smi --showbus --showuse --showclocks --showtemp --showpower > "$out/smi-before.txt"
numactl --cpunodebind=0 --membind=0 "$out/probe" "$out" "$phase" "$rounds" > "$out/stdout.txt" 2> "$out/run.log" &
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
