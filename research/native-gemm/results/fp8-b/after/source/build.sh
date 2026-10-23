#!/usr/bin/env bash
# Standalone HIP build. Fresh MFMA-first defaults preserve branch-clean and
# older measured binaries.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)

if (($# > 4)); then
    echo 'usage: build.sh [128|256 tile] [output-directory] [gemm|resident] [32|64 StageK]' >&2
    exit 2
fi

tile=${1:-128}
case "$tile" in
    128|256) ;;
    *) echo 'tile must be128 or256' >&2; exit 2 ;;
esac

kind=${3:-gemm}
generator_options=()
case "$kind" in
    gemm) ;;
    resident) generator_options=(--resident) ;;
    *) echo 'kind must be gemm or resident' >&2; exit 2 ;;
esac

stage=${4:-32}
case "$stage" in
    32|64) ;;
    *) echo 'StageK must be32 or64' >&2; exit 2 ;;
esac

generator=generate.py
namespace=mfma-first
format_flags=()
extra_sources=()
if [[ ${NATIVE_FP8_B:-0} == 1 ]]; then
    [[ $tile == 128 && $stage == 32 && $kind == gemm ]] || {
        echo 'FP8 variant supports only128x128x32 ordinary GEMM' >&2; exit 2;
    }
    generator=generate_fp8.py
    namespace=fp8-b
    format_flags=(-DNATIVE_FP8_B=1)
    extra_sources=("$here/generate_fp8.py" "$here/data_fp8.hpp" "$here/build_fp8.sh")
fi
out=${2:-"$root/temp/native-gemm/$namespace/s$stage-t$tile"}
mkdir -p "$out/tmp"
out=$(cd "$out" && pwd)
export TMPDIR="$out/tmp"

if [[ $generator == generate_fp8.py ]]; then
    python3 "$here/$generator" --output "$out/compute.hpp"
else
    python3 "$here/$generator" --tile "$tile" --stage-k "$stage" \
        "${generator_options[@]}" --output "$out/compute.hpp"
fi

flags=(
    -O3
    -std=c++20
    --offload-arch=gfx90a
    -fdenormal-fp-math=ieee
    -Wall
    -Wextra
    -I"$out"
    "${format_flags[@]}"
)
{
    printf '%q' hipcc
    printf ' %q' "${flags[@]}"
    printf '\n'
} > "$out/compiler-flags.txt"

hipcc "${flags[@]}" "$here/probe.hip" -o "$out/probe"

# Inspect the actual code object embedded in the executable, not a side build.
llvm-objdump-21 --offloading "$out/probe" > "$out/offloading.txt"
cp "$out/probe.0.hipv4-amdgcn-amd-amdhsa--gfx90a" "$out/probe.hsaco"
llvm-objdump-21 -d --demangle "$out/probe.hsaco" > "$out/probe.isa"
llvm-readobj-21 --notes "$out/probe.hsaco" > "$out/metadata.txt"

# The same compilation flags are used by the local language-server database.
python3 - "$here" "$out" "$(hipconfig -l)/clang++" "${flags[@]}" <<'PY'
import json
import sys
from pathlib import Path

source_directory, output_directory, compiler = sys.argv[1:4]
source = str(Path(source_directory) / 'probe.hip')
command = {
    'directory': output_directory,
    'file': source,
    'arguments': [compiler, *sys.argv[4:], '-x', 'hip', '-c', source],
}
Path(output_directory, 'compile_commands.json').write_text(
    json.dumps([command], indent=2) + '\n'
)
PY

python3 "$here/inspect_code.py" "$out"
sha256sum \
    "$here/probe.hip" \
    "$here/data.hpp" \
    "$here/generate.py" \
    "${extra_sources[@]}" \
    "$out/compute.hpp" \
    "$out/probe" \
    "$out/probe.hsaco" > "$out/hashes.txt"

echo "Built $out/probe"
