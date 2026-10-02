#!/usr/bin/env bash
# ============================================================================
# RETIRED ARTIFACT (2026-09) — preserved in git history, not maintained.
#
# This was the run-script for the smoke cross-check harness
# (research/model-format/smoke_load.cpp, deleted in the follow-up commit as
# the manifest path retired). It loaded a packed .model via init_engine AND
# the same model via the manifest + safetensors path (gpu::Model::load), ran
# both forward_logits() on identical tokens, and compared bit-exactly. The
# committed result was PASS (128256 logits bit-exact), confirming the
# model-format path == the manifest path. The harness is obsolete now that
# the two paths are unified on gpu::Model and the manifest path is retiring;
# this script is kept only as a record of how the cross-check was run. It
# will NOT run as-is (smoke_load.cpp is gone).
# ============================================================================
#
# Original doc follows:
# Smoke test for the alice model format loader (research/model-format path).
#
# Loads a packed .model via init_engine (the new loader: envelope -> metadata
# + tokenizer/chat_template -> device model) and compares every block's
# forward() output bit-exactly against the manifest + safetensors path. This
# is the cross-check that packer layout == factory layout == old-path bytes,
# and that the arena/scratch (NaN-poisoned) is written before read.
#
# Run:   ./temp/smoke_test.sh
# From:  anywhere (resolves the repo root from this script's own location).
#
# Prereq: a GPU (the engine is device-resident), a built libalice.a.
# Builds the smoke binary in a tmpdir; cleans up on exit. Re-packs
# temp/packed1b if it's missing. Prefers build-release/libalice.a (faster),
# falls back to build/libalice.a.
set -euo pipefail

# --- resolve repo root from this script's location (works from any CWD) -----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- pick the built libalice.a (release preferred, debug fallback) ----------
LIB=""
BUILD_TREE=""
for cand in build-release build; do
  if [[ -f "$cand/libalice.a" ]]; then
    LIB="$cand/libalice.a"
    BUILD_TREE="$cand"
    break
  fi
done
if [[ -z "$LIB" ]]; then
  echo "error: no libalice.a found — build first:  cmake -S . -B build && cmake --build build -j" >&2
  exit 2
fi
echo "[smoke_test] using $LIB"

# --- nlohmann_json include dir (FetchContent) -------------------------------
JSON_INC="$BUILD_TREE/_deps/nlohmann_json-src/include"
if [[ ! -d "$JSON_INC" ]]; then
  # Debug/release share the same _deps layout; try the other tree.
  for cand in build build-release; do
    if [[ -d "$cand/_deps/nlohmann_json-src/include" ]]; then
      JSON_INC="$cand/_deps/nlohmann_json-src/include"; break
    fi
  done
fi

# --- ensure the packed model exists (re-pack if missing) --------------------
PACKED="temp/packed1b"
MODEL_DIR="/opt/models/Llama-3.2-1B-Instruct"
MANIFEST="models/llama-3.2-1b-instruct.manifest.json"
if [[ ! -f "$PACKED/model.model" || ! -f "$PACKED/weights.bin" ]]; then
  echo "[smoke_test] $PACKED missing — re-packing from $MODEL_DIR"
  if [[ ! -x ".venv/bin/python" ]]; then
    echo "error: .venv/bin/python not found — needed to run tools/pack_model.py" >&2
    exit 2
  fi
  .venv/bin/python tools/pack_model.py "$MODEL_DIR" "$PACKED"
fi

# --- build the smoke binary in a tmpdir, clean up on exit -------------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

INC="-Iinclude -Isrc -I$JSON_INC"
DEFS="-DJSON_DIAGNOSTICS=1 -D__HIP_PLATFORM_AMD__ -DALICE_SOURCE_DIR=\"$REPO_ROOT\""
LIBS="-lamdhip64 -lrocblas -lblake3 -lpcre2-8 -lpthread -lgomp"

echo "[smoke_test] compiling smoke_load.cpp (hipcc, c++20)..."
hipcc -std=c++20 $INC $DEFS \
  -c research/model-format/smoke_load.cpp -o "$TMP/sl.o"

echo "[smoke_test] linking against $LIB (--whole-archive: factories self-register)..."
hipcc --hip-link "$TMP/sl.o" \
  -Wl,--whole-archive "$LIB" -Wl,--no-whole-archive \
  $LIBS -o "$TMP/sl"

echo "[smoke_test] running (4096 ctx, 512 mpc)..."
echo
"$TMP/sl" "$PACKED" "$MODEL_DIR" "$MANIFEST" 4096 512
