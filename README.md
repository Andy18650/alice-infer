# alice-infer

A from-scratch C++20 LLM inference engine for this machine (4× MI250X =
8 GCDs, gfx90a; Zen3 host). **GPU-mandatory**: model state is
device-resident and the block loop runs entirely on device — the host
loads weights, tokenizes, and reads back token ids. Weights load directly
from safetensors against an externally-authored manifest that acts as the
total contract. Single-user, single-context.

The full design rationale lives in `docs/engine-design.md`.

## Requirements

- CMake, a C++20 compiler (developed with GCC 15), **ROCm (rocBLAS + HIP
  runtime) and hipcc** — required, the engine does not build without them
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
ctest --test-dir build          # 60 tests; model-backed ones skip if absent
```

Long-sequence canonical fixtures (~1500-token WikiText prompts, fp32
reference) are NOT committed (777 MB each). Regenerate with:

```sh
.venv/bin/python tests/python/dump_canonical.py \
    --model /opt/models/Llama-3.2-1B-Instruct \
    --out tests/canonical_long --skip-battery \
    --prompt-file <raw_text_file> --prompt-tokens 1500 --dtype fp32
```

## Run

```sh
./build-release/alice_repl <model_dir> <manifest.json> \
    [max_new=64] [max_seq=8192] [prompt]
# no prompt -> interactive REPL; with prompt -> single non-interactive round
```

Examples:

```sh
./build-release/alice_repl /opt/models/Llama-3.2-1B-Instruct \
    models/llama-3.2-1b-instruct.manifest.json 32 8192 "The capital of France is"

./build-release/alice_repl /opt/models/deepseek-ai/deepseek-r1-distill-llama-8b \
    models/deepseek-r1-distill-llama-8b.manifest.json 32 8192 "The capital of France is"
```

Multi-shard models are handled automatically (all `.safetensors` in the
model directory are probed; no index file is read). Manifests for
llama-family models can be authored with `tools/gen_manifest.py` and
pre-flight-checked with `tools/check_manifest.py`.

Development-only fault localization: `ALICE_CPU_REF_BLOCKS=0,5,17` swaps
the listed blocks to naive CPU reference implementations (D2H → compute →
H2D inside the block). Never for production.

## Layout

- `include/alice/` — public headers (typed tensor views, manifest, gpu model)
- `src/gpu/` — GPU-resident engine (SIMT kernels via hipcc, blocks, trunk, model)
- `src/blocks/cpu_generic/` — naive CPU reference implementations (test oracle)
- `src/blocks/gpu_rocblas/` — platform plumbing (device + rocBLAS handle)
- `src/frontend/` — REPL, tokenizer (PCRE2 via dlopen), benches
- `models/` — manifests (the contract), one per bring-up model
- `tests/` — GoogleTest suite + canonical teacher-forced fixtures
- `tools/` — manifest generator/auditor, sweep scripts
- `research/` — decision-support studies (RCCL/P2P multi-GPU, ROCm overlays)
- `docs/` — engine-design.md (the design), session-api.md (draft)
