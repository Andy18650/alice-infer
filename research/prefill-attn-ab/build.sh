#!/bin/sh
# Standalone build (research harness, deliberately outside the engine CMake).
set -e
cd "$(dirname "$0")"
hipcc -O3 -std=c++17 --offload-arch=gfx90a ab.hip -o ab
echo "built ./ab"
