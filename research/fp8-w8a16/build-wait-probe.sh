#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
base=${1:-"$root/temp/fp8-w8a16/wait"}
for mode in waited unwaited; do
  omit=0; [[ "$mode" == unwaited ]] && omit=1
  out="$base/$mode"; mkdir -p "$out/tmp"; out=$(cd "$out" && pwd)
  export TMPDIR="$out/tmp"
  flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra "-DOMIT_PAIR_WAIT=$omit")
  { printf '%q' hipcc; printf ' %q' "${flags[@]}"; printf '\n'; } > "$out/compiler-flags.txt"
  hipcc "${flags[@]}" "$here/wait_probe.hip" -o "$out/wait-probe"
  llvm-objdump-21 --offloading "$out/wait-probe" > "$out/offloading.txt"
  cp "$out/wait-probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/wait-probe.hsaco"
  llvm-objdump-21 -d --demangle "$out/wait-probe.hsaco" > "$out/wait-probe.isa"
  llvm-readobj-21 --notes "$out/wait-probe.hsaco" > "$out/metadata.txt"
  sha256sum "$here/wait_probe.hip" "$out/wait-probe" "$out/wait-probe.hsaco" > "$out/hashes.txt"
  echo "Built $out/wait-probe"
done
