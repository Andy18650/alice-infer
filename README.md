# alice-infer

A from-scratch C++20 LLM inference engine (single-user, single-context).
Weights load directly from safetensors against an externally-authored
manifest that acts as the total contract. Runs Llama-family models on CPU
(Zen3-tuned AVX2 kernels) and, for the FFN blocks, on AMD GPUs via rocBLAS
(auto-selected when a ROCm GPU is present).

The full design rationale lives in `docs/engine-design.md`.

## Requirements

- CMake, a C++20 compiler (developed with GCC 15)
- Optional: OpenMP (kernel parallelism; sequential fallback without it)
- Optional: ROCm (rocBLAS + HIP runtime + hipRTC) for the GPU FFN path
- Tests: GoogleTest (fetched automatically), models under `/opt/models`

## Build

```sh
cmake -S . -B build                                  # Debug (tests)
cmake --build build -j

cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release   # optimized
cmake --build build-release -j
```

## Test

```sh
ctest --test-dir build          # 54 tests; model-backed ones skip if absent
```

## Run

```sh
./build-release/alice_repl <model_dir> <manifest.json> \
    [max_new=64] [max_seq=8192] [prompt] [hardware]
# hardware: auto (default) | gpu_rocblas | cpu_avx2 | cpu_generic
# no prompt -> interactive REPL; with prompt -> single non-interactive round
```

Examples:

```sh
./build-release/alice_repl /opt/models/Llama-3.2-1B-Instruct \
    models/llama-3.2-1b-instruct.manifest.json 32 8192 "The capital of France is"

./build-release/alice_repl /opt/models/deepseek-ai/deepseek-r1-distill-llama-8b \
    models/deepseek-r1-distill-llama-8b.manifest.json 32 8192 "The capital of France is"

./build-release/alice_repl /opt/models/deepseek-ai/deepseek-r1-distill-llama-70b \
    models/deepseek-r1-distill-llama-70b.manifest.json 8 2048 "The capital of France is"
```

Multi-shard models are handled automatically (all `.safetensors` in the
model directory are probed; no index file is read). Manifests for
llama-family models can be authored with `tools/gen_manifest.py` and
pre-flight-checked with `tools/check_manifest.py`.

## Benchmarks & tools

```sh
./build-release/alice_bench_ffn            # per-block FFN microbench
./build-release/alice_bench_lmhead_gpu     # rocBLAS lm_head scout
tools/omp_sweep.sh                         # thread/NUMA sweep helper
```

## Layout

- `include/alice/` — public headers (typed tensor views, block interfaces, manifest, model)
- `src/blocks/<hw>/` — block implementations per hardware class (cpu_generic oracle, cpu_avx2, gpu_rocblas)
- `src/frontend/` — REPL, tokenizer (PCRE2 via dlopen), benches
- `models/` — manifests (the contract), one per bring-up model
- `tests/` — GoogleTest suite + canonical teacher-forced fixtures
- `tools/` — manifest generator/auditor, sweep scripts
- `docs/` — engine-design.md (the design), session-api.md (draft)
