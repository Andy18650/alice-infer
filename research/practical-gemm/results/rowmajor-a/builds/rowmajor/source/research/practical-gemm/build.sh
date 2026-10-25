#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
layout=${1:?packed or rowmajor}
case "$layout" in packed|rowmajor) ;; *) exit 2;; esac
out=${2:?fresh build directory required}
if [[ -e $out && -n $(find "$out" -mindepth 1 -print -quit) ]]; then
    echo "refusing nonempty build path: $out" >&2; exit 2
fi
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"
python3 "$here/generate.py" --layout "$layout" --output "$out/compute.hpp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra -I"$out")
if [[ $layout == rowmajor ]]; then flags+=(-DSTUDY_ROW_MAJOR_A=1); fi
{ printf '%q' hipcc; printf ' %q' "${flags[@]}" "$here/probe.hip" -o "$out/probe"; printf '\n'; } >"$out/compiler-flags.txt"
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
sha256sum "$here/"{probe.hip,rowmajor_kernel.hpp,generate.py,build.sh} \
    "$root/research/native-gemm/"{data.hpp,data_fp8.hpp,data_scaled.hpp,scaled_kernel.hpp,generate.py,generate_fp8.py,generate_scaled.py,generate_scale_latency.py,generate_q0_schedule.py} \
    "$out/"{compute.hpp,probe,probe.hsaco} >"$out/hashes.txt"
