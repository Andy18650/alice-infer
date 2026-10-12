#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
out="${1:?output directory}"
suite="${2:-quick}"
rounds="${3:-9}"
iters="${4:-12}"
publication="${5:-release}"
suffix=""
case "$publication" in release) ;; strong) suffix="-strong";; *) echo "publication: release|strong" >&2; exit 1;; esac
binary="$root/temp/gemv-layout/bench$suffix"
if [ ! -x "$binary" ] || [ "$binary" -ot "$here/bench.hip" ] || [ "$binary" -ot "$here/kernels.hpp" ]; then
  echo "Missing/stale benchmark: run $here/build.sh first" >&2; exit 1
fi
mkdir -p "$out"
cp "$binary" "$out/bench"
cp "$here/bench.hip" "$here/kernels.hpp" "$here/check-codegen.py" "$here/check-fused.py" "$here/summarize.py" "$out/"
{
  echo "O3 HIP C++20 gfx90a; HIP device 0; CPU+memory NUMA node 0; no model/context; no clock overrides"
  echo "BF16 weights; FP32 x/accum/y, no x rounding; packing/upload/oracle outside timing"
  echo "suite=$suite rounds=$rounds iters=$iters publication=$publication; synthetic weights; useful and padded traffic reported separately"
  git -C "$root" rev-parse HEAD
  hipcc --version
  sha256sum "$out/bench" "$out/bench.hip" "$out/kernels.hpp" "$out/check-codegen.py" "$out/check-fused.py" "$out/summarize.py"
} > "$out/metadata.txt"
llvm-objdump --offloading "$out/bench" > "$out/offloading.txt"
llvm-objdump -d --demangle "$out/bench.0.hipv4-amdgcn-amd-amdhsa--gfx90a" > "$out/disassembly.txt"
python3 "$out/check-codegen.py" "$out" > "$out/inspection.txt"
python3 "$out/check-fused.py" "$out" > "$out/fused-inspection.txt"
rocm-smi --showbus --showclocks --showtemp --showpower > "$out/smi-before.txt"
numactl --cpunodebind=0 --membind=0 "$out/bench" "$out" "$suite" "$rounds" "$iters" > "$out/stdout.txt" 2> "$out/run.log" &
pid=$!
python3 "$here/../vram-ceiling/monitor.py" "$pid" "$out" &
monitor=$!
status=0
wait "$pid" || status=$?
wait "$monitor" || true
rocm-smi --showbus --showclocks --showtemp --showpower > "$out/smi-after.txt"
[ "$status" -eq 0 ] || exit "$status"
if [ "$suite" != test ]; then python3 "$out/summarize.py" "$out"; fi
echo "saved $out"
