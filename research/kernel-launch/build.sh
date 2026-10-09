#!/bin/sh
# Single-GCD native HIP C++20 probe, independent of the production engine.
set -eu
cd "$(dirname "$0")"
root=$(CDPATH= cd ../.. && pwd)
mkdir -p "$root/temp/kernel-launch"
hipcc -O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra probe.hip \
  -o "$root/temp/kernel-launch/probe"
echo "built $root/temp/kernel-launch/probe"
