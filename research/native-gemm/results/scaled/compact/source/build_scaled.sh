#!/usr/bin/env bash
# Explicit fresh namespaces; never rebuild a previously measured kernel.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
mode=${1:-scaled}
case "$mode" in scaled|compact|heavy) ;; *) exit 2;; esac
out=${2:-"$root/temp/native-gemm/scaled/$mode"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
python3 "$here/generate_scaled.py" --mode "$mode" --output "$out/compute.hpp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee
       -Wall -Wextra -DNATIVE_FP8_B=1 -DNATIVE_SCALED_B=1 -I"$out")
{ printf '%q' hipcc; printf ' %q' "${flags[@]}"; printf '\n'; } >"$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/probe.hip" -o "$out/probe"
llvm-objdump-21 --offloading "$out/probe" >"$out/offloading.txt"
cp "$out/probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/probe.hsaco"
llvm-objdump-21 -d --demangle "$out/probe.hsaco" >"$out/probe.isa"
llvm-readobj-21 --notes "$out/probe.hsaco" >"$out/metadata.txt"
python3 - "$here" "$out" "$(hipconfig -l)/clang++" "${flags[@]}" <<'PY'
import json,sys
from pathlib import Path
here,out,compiler=sys.argv[1:4]
source=str(Path(here)/'probe.hip')
Path(out,'compile_commands.json').write_text(json.dumps([dict(
    directory=out,file=source,arguments=[compiler,*sys.argv[4:],'-x','hip','-c',source],
)],indent=2)+'\n')
PY
python3 "$here/inspect_scaled.py" "$out"
sha256sum "$here/"{probe.hip,data.hpp,data_fp8.hpp,data_scaled.hpp,scaled_kernel.hpp,generate.py,generate_fp8.py,generate_scaled.py} \
    "$out/"{compute.hpp,probe,probe.hsaco} >"$out/hashes.txt"
