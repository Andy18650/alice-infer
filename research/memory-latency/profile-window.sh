#!/usr/bin/env bash
# Existing installed components only. One counter pass, one selected launch/case.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:?output directory required};binary=${2:?binary required}
mkdir -p "$out";out=$(cd "$out" && pwd);binary=$(realpath "$binary")
if [[ -n $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]]; then echo "Refusing nonempty output: $out" >&2;exit 1;fi
manifest="$(dirname "$binary")/hashes.txt"
[[ -f "$manifest" ]] || { echo 'Use a build-window.sh build with its hash manifest' >&2;exit 1; }
sha256sum -c "$manifest" > /dev/null
read -r expected _ < "$manifest"
[[ $(sha256sum "$binary" | cut -d' ' -f1) == "$expected" ]] || { echo 'Executable does not match build manifest' >&2;exit 1; }
sdk=/home/andyhu/.cache/uv/archive-v0/2EEHNuXZame80Ad1/_rocm_sdk_core
export PATH="/home/andyhu/.local/rocm-prof-venv/bin:$sdk/bin:$PATH"
export PYTHONPATH="$sdk/lib/python3/site-packages"
export LD_LIBRARY_PATH=/home/andyhu/.local/rocm-prof/lib
export HIP_VISIBLE_DEVICES=0
cp "$binary" "$out/measured-window"
cp "$here/window-counters.yaml" "$out/input.yaml"
command=("$sdk/bin/rocprofv3" -i "$out/input.yaml" -d "$out/capture" --kernel-trace --output-format csv --kernel-include-regex window_entry --kernel-iteration-range '[4,8,12,16,20,24,28,32]' -- numactl --cpunodebind=0 --membind=0 "$out/measured-window" --mib 1040 --rounds 1 --repeats 1 --seed 12345)
printf '%q ' "${command[@]}" > "$out/command.txt";printf '\n' >> "$out/command.txt"
"${command[@]}" > "$out/run.log" 2>&1
sha256sum "$out/measured-window" "$out/input.yaml" > "$out/hashes.txt"
touch "$out/complete"
