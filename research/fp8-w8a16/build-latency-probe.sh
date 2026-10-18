#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/fp8-w8a16/latency"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"; printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/latency_probe.hip" -o "$out/latency-probe"
llvm-objdump-21 --offloading "$out/latency-probe" > "$out/offloading.txt"
cp "$out/latency-probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/latency-probe.hsaco"
llvm-objdump-21 -d --demangle "$out/latency-probe.hsaco" > "$out/latency-probe.isa"
llvm-readobj-21 --notes "$out/latency-probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/latency_probe.hip" "$out/latency-probe" "$out/latency-probe.hsaco" > "$out/hashes.txt"
echo "Built $out/latency-probe"
