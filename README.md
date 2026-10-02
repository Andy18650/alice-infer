# alice-infer

A from-scratch C++20 LLM inference engine for this machine (4× MI250X =
8 GCDs, gfx90a; Zen3 host). **GPU-mandatory**: model state is
device-resident and the block loop runs entirely on device — the host
loads weights, tokenizes, and reads back token ids. Models load via the
alice model format (a packed `.model` descriptor + `weights.bin`, packed
from HuggingFace safetensors by `tools/pack_model.py`). Single-user,
single-context.

The full design rationale lives in `docs/engine-design.md`.

## Requirements

- CMake, a C++20 compiler (developed with GCC 15), **ROCm (rocBLAS + HIP
  runtime) and hipcc** — required, the engine does not build without them
- **PCRE2** (`libpcre2-dev`; discovered via pkg-config) — required, for the
  tokenizer's pre-tokenizer regex
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
./build-release/alice_repl <model.model> [max_new=64] [max_seq=8192] [prompt]
# no prompt -> interactive REPL; with prompt -> single non-interactive round
```

Examples:

```sh
./build-release/alice_repl /opt/models/alice/Llama-3.2-1B-Instruct/model.model \
    32 8192 "The capital of France is"

./build-release/alice_repl /opt/models/alice/deepseek-r1-distill-llama-8b/model.model \
    32 8192 "The capital of France is"
```

Models are loaded via the alice model format (a packed `.model` descriptor +
`weights.bin`); pack a HuggingFace llama-family model with
`tools/pack_model.py`:

```sh
.venv/bin/python tools/pack_model.py /opt/models/Llama-3.2-1B-Instruct /opt/models/alice/Llama-3.2-1B-Instruct
```

Development-only fault localization: `ALICE_CPU_REF_BLOCKS=0,5,17` swaps
the listed blocks to naive CPU reference implementations (D2H → compute →
H2D inside the block). Never for production.

## Layout

- `include/alice/` — public headers (typed tensor, engine config)
- `src/gpu/` — GPU-resident engine (trunk/state SIMT kernels via hipcc, trunk, model)
- `src/blocks/cpu_generic/` — naive CPU reference implementations (test oracle)
- `src/blocks/gpu_rocm/` — ROCm block impls (rocBLAS GEMMs + private SIMT kernels) + platform plumbing (device + rocBLAS handle)
- `src/frontend/` — REPL, tokenizer (system PCRE2)
- `src/model_format.{hpp,cpp}` — the alice model format loader (init_engine)
- `tests/` — GoogleTest suite + canonical teacher-forced fixtures
- `tools/` — pack_model.py (HF -> .model), sweep scripts
- `research/` — decision-support studies (RCCL/P2P multi-GPU, ROCm overlays)
- `docs/` — engine-design.md (the design), session-api.md (draft)
