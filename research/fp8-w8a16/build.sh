#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/fp8-w8a16"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"
printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/probe.hip" -o "$out/probe"
objdump=${OBJDUMP:-llvm-objdump-21}
"$objdump" --offloading "$out/probe" > "$out/offloading.txt"
cp "$out/probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/probe.hsaco"
"$objdump" -d --demangle "$out/probe.hsaco" > "$out/probe.isa"
llvm-readobj-21 --notes "$out/probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/probe.hip" "$out/probe" "$out/probe.hsaco" > "$out/hashes.txt"
echo "Built $out/probe and actual code-object disassembly"
