#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/fp8-w8a16/nop-sweep"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"; printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/nop_sweep.hip" -o "$out/nop-sweep"
llvm-objdump-21 --offloading "$out/nop-sweep" > "$out/offloading.txt"
cp "$out/nop-sweep.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/nop-sweep.hsaco"
llvm-objdump-21 -d --demangle "$out/nop-sweep.hsaco" > "$out/nop-sweep.isa"
llvm-readobj-21 --notes "$out/nop-sweep.hsaco" > "$out/metadata.txt"
hipcc "${flags[@]}" "$here/nop_clock.hip" -o "$out/nop-clock"
llvm-objdump-21 --offloading "$out/nop-clock" > "$out/nop-clock-offloading.txt"
cp "$out/nop-clock.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/nop-clock.hsaco"
llvm-objdump-21 -d --demangle "$out/nop-clock.hsaco" > "$out/nop-clock.isa"
sha256sum "$here/nop_sweep.hip" "$here/nop_clock.hip" "$out/nop-sweep" "$out/nop-sweep.hsaco" "$out/nop-clock" "$out/nop-clock.hsaco" > "$out/hashes.txt"
echo "Built $out/nop-sweep"
