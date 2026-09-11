#!/bin/sh
# Host-only g++ build (no hipcc): RCCL + HIP C APIs, same style as the
# earlier rocBLAS scout benches.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++20 -D__HIP_PLATFORM_AMD__ -I/usr/include/rccl \
    study.cpp -o study -lrccl -lamdhip64 -lpthread
echo "built ./study"
