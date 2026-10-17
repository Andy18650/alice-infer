#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:-"$root/temp/fp8-w8a16/order"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra)
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; } > "$out/compiler-flags.txt"; printf '\n' >> "$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/order_probe.hip" -o "$out/order-probe"
llvm-objdump-21 --offloading "$out/order-probe" > "$out/offloading.txt"
cp "$out/order-probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/order-probe.hsaco"
llvm-objdump-21 -d --demangle "$out/order-probe.hsaco" > "$out/order-probe.isa"
llvm-readobj-21 --notes "$out/order-probe.hsaco" > "$out/metadata.txt"
sha256sum "$here/order_probe.hip" "$out/order-probe" "$out/order-probe.hsaco" > "$out/hashes.txt"
echo "Built $out/order-probe"
