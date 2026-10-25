#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
format=${1:?native,plain,block128,block256}
case "$format" in native) id=0;;plain) id=1;;block128) id=2;;block256) id=3;;*) exit 2;;esac
out=${2:?fresh build directory required}
if [[ -e $out && -n $(find "$out" -mindepth 1 -print -quit) ]];then
    echo "refusing nonempty build path: $out" >&2;exit 2
fi
mkdir -p "$out/tmp";out=$(cd "$out" && pwd);export TMPDIR="$out/tmp"
python3 "$here/common_b.py" --format "$format" --output "$out/compute.hpp"
flags=(-O3 -std=c++20 --offload-arch=gfx90a -fdenormal-fp-math=ieee -Wall -Wextra -DCOMMON_B_FORMAT="$id" -I"$out")
{ printf '%q' hipcc;printf ' %q' "${flags[@]}" "$here/common_probe.hip" -o "$out/probe";printf '\n'; } >"$out/compiler-flags.txt"
hipcc "${flags[@]}" "$here/common_probe.hip" -o "$out/probe"
hipcc --version >"$out/compiler-version.txt"
llvm-objdump-21 --offloading "$out/probe" >"$out/offloading.txt"
cp "$out/probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/probe.hsaco"
llvm-objdump-21 -d --demangle "$out/probe.hsaco" >"$out/probe.isa"
llvm-readobj-21 --notes "$out/probe.hsaco" >"$out/metadata.txt"
ldd "$out/probe" >"$out/dynamic-dependencies.txt"
python3 - "$here" "$out" "$(hipconfig -l)/clang++" "${flags[@]}" <<'PY'
import json,sys
from pathlib import Path
here,out,compiler=sys.argv[1:4];source=str(Path(here)/'common_probe.hip')
Path(out,'compile_commands.json').write_text(json.dumps([dict(directory=out,file=source,
    arguments=[compiler,*sys.argv[4:],'-x','hip','-c',source])],indent=2)+'\n')
PY
sha256sum "$here/"{common_probe.hip,common_gpu.hpp,common_host.hpp,common_b.py,generate.py,rowmajor_kernel.hpp,build_common.sh} \
    "$root/research/native-gemm/"{data.hpp,data_fp8.hpp,data_scaled.hpp,generate.py,generate_fp8.py,generate_scaled.py,generate_scale_latency.py,generate_q0_schedule.py} \
    "$out/"{compute.hpp,kernel.hpp,probe,probe.hsaco} >"$out/hashes.txt"
