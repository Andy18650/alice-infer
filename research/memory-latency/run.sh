#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=${1:?output directory required}
binary=${2:?binary required}
shift 2
mkdir -p "$out";out=$(cd "$out" && pwd)
binary=$(realpath "$binary")
if [[ -n $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  echo "Refusing nonempty output: $out" >&2;exit 1
fi
manifest="$(dirname "$binary")/hashes.txt"
[[ -f "$manifest" ]] || { echo 'Use a build.sh build with its hash manifest' >&2;exit 1; }
sha256sum -c "$manifest" > /dev/null
read -r expected _ < "$manifest"
[[ $(sha256sum "$binary" | cut -d' ' -f1) == "$expected" ]] || { echo 'Executable does not match build manifest' >&2;exit 1; }
cp "$binary" "$out/measured-probe"
mkdir -p "$out/source"
cp "$here"/*.hip "$here"/*.sh "$out/source/"
hipcc --version > "$out/compiler.txt"
sha256sum "$out/measured-probe" "$out/source/probe.hip" > "$out/hashes.txt"
printf '%q ' env HIP_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 "$out/measured-probe" "$@" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
HIP_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 "$out/measured-probe" "$@" > "$out/results.csv" 2> "$out/run.log"
touch "$out/complete"
