#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/fp8-w8a16/paired"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"; printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/paired_probe.hip" -o "$out/paired-probe"
llvm-objdump-21 --offloading "$out/paired-probe" > "$out/offloading.txt"
cp "$out/paired-probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/paired-probe.hsaco"
llvm-objdump-21 -d --demangle "$out/paired-probe.hsaco" > "$out/paired-probe.isa"
llvm-readobj-21 --notes "$out/paired-probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/paired_probe.hip" "$out/paired-probe" "$out/paired-probe.hsaco" > "$out/hashes.txt"
echo "Built $out/paired-probe"
