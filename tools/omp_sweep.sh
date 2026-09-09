#!/usr/bin/env bash
# OpenMP thread-count sweep (design §10 knobs: thread count first).
# Usage: tools/omp_sweep.sh [decode_tokens=32]
# Reports prefill/decode rates for OMP_NUM_THREADS = 16..96 step 16.
set -u
cd "$(dirname "$0")/.."

REPL=./build-release/alice_repl
MODEL=/opt/models/Llama-3.2-1B-Instruct
MANIFEST=models/llama-3.2-1b-instruct.manifest.json
TOKENS=${1:-32}

# A long-ish prompt gives the prefill measurement some signal (one chunk).
PROMPT="The history of computing began with mechanical calculators in the seventeenth century, when Blaise Pascal and Gottfried Wilhelm Leibniz built machines that could add and multiply. Two centuries later, Charles Babbage designed the Analytical Engine, a programmable computer that was never completed in his lifetime, while Ada Lovelace wrote what is often called the first algorithm intended for a machine. The electronic age arrived with vacuum tubes, then transistors, then integrated circuits, each step making computers smaller, cheaper, and vastly faster, until a machine that once filled a room fit into a pocket."

printf "%-9s | %-22s | %-22s\n" threads prefill decode
for t in 16 32 48 64 80 96; do
  out=$(OMP_NUM_THREADS=$t "$REPL" "$MODEL" "$MANIFEST" "$TOKENS" 8192 "$PROMPT" 2>&1)
  pre=$(echo "$out" | grep -oP 'prefill: .*tok/s' | grep -oP '[\d.]+ tok/s, [\d.]+' | head -1)
  pre_rate=$(echo "$out" | grep -oP '\[prefill[^\]]*' | grep -oP '[\d.]+(?= tok/s)')
  dec_rate=$(echo "$out" | grep -oP '\[decode[^\]]*' | grep -oP '[\d.]+(?= tok/s)')
  pre_ms=$(echo "$out" | grep -oP '\[prefill[^\]]*' | grep -oP '[\d.]+(?= ms/token)')
  dec_ms=$(echo "$out" | grep -oP '\[decode[^\]]*' | grep -oP '[\d.]+(?= ms/token)')
  printf "%-9s | %6s ms/t %6s t/s | %6s ms/t %6s t/s\n" "$t" "$pre_ms" "$pre_rate" "$dec_ms" "$dec_rate"
done
