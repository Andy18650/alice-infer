#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
root=$(CDPATH= cd "$here/../.." && pwd)
out="${1:?output directory}"
mode="${2:-controlled}"
phase="${3:-quick}"
rounds="${4:-9}"
barriers="${5:-off}"
clock_mode="${6:-rich}"
timed_iters="${7:-12}"
window_phase=0
case "$phase" in windows|body-control) window_phase=1;; esac
case "$clock_mode" in rich|basic) ;; *) echo "clock monitor: rich|basic" >&2; exit 1;; esac
case "$mode" in controlled|default) ;; *) echo "mode: controlled|default" >&2; exit 1;; esac
suffix=""
case "$barriers" in off) ;; on) suffix="-barriers";; phase) suffix="-phase-barriers";; *) echo "barriers: off|on|phase" >&2; exit 1;; esac
binary="$root/temp/vram-coalescing/$mode$suffix"
if [ ! -x "$binary" ] || [ "$binary" -ot "$here/probe.hip" ] || [ "$binary" -ot "$here/readers.hpp" ] || [ "$binary" -ot "$here/wave-control.hpp" ]; then
  echo "Missing/stale probe: run $here/build.sh first" >&2; exit 1
fi
mkdir -p "$out"
cp "$here/probe.hip" "$out/probe.hip"
cp "$here/readers.hpp" "$here/wave-control.hpp" "$here/check-loads.py" "$out/"
cp "$binary" "$out/probe"
{
  echo "O3 HIP C++20 gfx90a; HIP device 0; CPU+memory NUMA node 0; no model/context"
  echo "mode=$mode phase=$phase rounds=$rounds scheduling_barriers=$barriers; $timed_iters iterations/batch; no clock overrides"
  if [ "$window_phase" -eq 1 ]; then echo "clock_monitor=$clock_mode; brief + 30s sustained warmup; split4 has four kernel launches/iteration"; fi
  if [ "$mode" = controlled ]; then
    echo "-fno-vectorize -fno-slp-vectorize -mllvm -amdgpu-load-store-vectorizer=false"
  fi
  git -C "$root" rev-parse HEAD
  hipcc --version
  sha256sum "$out/probe.hip" "$out/readers.hpp" "$out/wave-control.hpp" "$out/probe" "$out/check-loads.py"
} > "$out/metadata.txt"
llvm-objdump --offloading "$out/probe" > "$out/offloading.txt"
llvm-objdump -d --demangle "$out/probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" > "$out/disassembly.txt"
python3 "$out/check-loads.py" "$out" "$mode" "$barriers" > "$out/inspection.txt"
rocm-smi --showbus --showuse --showclocks --showtemp --showpower > "$out/smi-before.txt"
rich_pid=""
if [ "$window_phase" -eq 1 ] && [ "$clock_mode" = rich ]; then
  clock_binary="$root/temp/vram-coalescing/clock-monitor"
  if [ ! -x "$clock_binary" ] || [ "$clock_binary" -ot "$here/clock-monitor.cpp" ]; then
    echo "Missing/stale clock monitor: run $here/build.sh first" >&2; exit 1
  fi
  cp "$here/clock-monitor.cpp" "$clock_binary" "$out/"
  sha256sum "$out/clock-monitor.cpp" "$out/clock-monitor" >> "$out/metadata.txt"
  ldd "$out/clock-monitor" > "$out/clock-libraries.txt"
  rm -f "$out/clock-ready" "$out/clock-stop" "$out/device.txt" "$out/status.txt"
  trap 'touch "$out/clock-stop"' EXIT
  "$out/clock-monitor" $$ "$out" auto > "$out/clock-stdout.txt" 2> "$out/clock-stderr.txt" &
  rich_pid=$!
  while [ ! -f "$out/clock-ready" ]; do
    if ! kill -0 "$rich_pid" 2>/dev/null; then echo "Clock monitor initialization failed" >&2; exit 1; fi
    sleep 0.05
  done
fi
numactl --cpunodebind=0 --membind=0 "$out/probe" "$out" "$phase" "$rounds" "$timed_iters" > "$out/stdout.txt" 2> "$out/run.log" &
pid=$!
python3 "$here/../vram-ceiling/monitor.py" "$pid" "$out" &
monitor=$!
status=0
wait "$pid" || status=$?
wait "$monitor" || true
if [ -n "$rich_pid" ]; then
  touch "$out/clock-stop"
  wait "$rich_pid" || status=1
fi
rocm-smi --showbus --showuse --showclocks --showtemp --showpower > "$out/smi-after.txt"
[ "$status" -eq 0 ] || exit "$status"
if [ "$window_phase" -eq 1 ]; then
  cp "$here/windows-summary.py" "$out/"
  python3 "$out/windows-summary.py" "$out"
else
  python3 "$here/summarize.py" "$out"
fi
echo "saved $out"
