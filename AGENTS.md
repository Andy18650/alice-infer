# AGENTS.md — working agreement for coding agents

This file is the fast on-ramp for agents working in this repo. The design
rationale and decision log live in `docs/engine-design.md` — read it
before non-trivial changes. The user dictates design; agents implement,
document, and test. When design and code disagree, flag it — don't
silently pick one.

## What this is

`alice-infer`: a from-scratch C++20 LLM inference engine for this machine
(4× MI250X = 8 GCDs × 64 GB HBM2e, gfx90a, ROCm 7.1; Zen3 host).
GPU-mandatory: model state is device-resident and the block loop runs
entirely on device. Safetensors weights only (GGUF rejected); an
externally authored manifest is the TOTAL contract. Bring-up models:
Llama-3.2-1B-Instruct, DeepSeek-R1-Distill-Llama-8B,
DeepSeek-R1-Distill-Llama-70B (under `/opt/models`, manifests in
`models/`).

**Migration state (2026-09):** the docs describe the committed
GPU-resident architecture; the tree is pre-migration (mixed CPU/GPU:
rocBLAS FFN, naive single-threaded CPU attention, host fp32 io). Migrate
in green-test steps; the canonical gate must stay green at every step.

## How constraints work here (read first)

- **Constraints come from the user, not from past agent improvisations.**
  When the right solution needs something outside the current sandbox — a
  package install, a new compiler/toolchain, a system config change —
  STOP and ask. Do not improvise around it: no /tmp-extracted .debs, no
  vendored library copies, no avoiding a standard compiler because the
  build file is currently simple. Standard approaches (system packages,
  hipcc, system libraries) are PREFERRED over hermetic workarounds.
- Nothing in this file or docs/engine-design.md is a constraint unless
  the user said it. If you find yourself treating an earlier agent
  decision as law, flag it and ask instead of building on it.

## Hard invariants (do not break)

- **Device-resident fp32 stream**: residual/io/KV live on the device;
  blocks operate on the stream in place; no host↔device transfers inside
  the block loop. fp32 end to end — only GEMM operands are quantized.
- **No vendor shim inside Blocks**: each block implementation is native
  to its architecture (hipcc/HIP + rocBLAS here). Portability = multiple
  per-arch impls of one abstract block class, selected at load; no impl
  for the present device = loud load error, never a silent CPU fallback.
- **Trunk/state ops are plain SIMT kernels** (no matrix intrinsics)
  behind the one macro header that maps kernel spelling to HIP/CUDA/MUSA.
  That header never crosses a Block boundary.
- **CPU blocks are development-only reference implementations** (D2H →
  naive host compute → H2D, all inside the block). Never production
  compute, never a fallback terminus.
- **Typed tensor system is the loading vocabulary**: `TView<T>`,
  `File::view_as<T>()` as the one checked runtime→static conversion.
  Device buffers are plain pointers, not views.
- **Optimized impls are self-contained**: private weight storage, private
  math, no shared kernel toolbox across impls. Private copies at
  construction; staged views freed on return.
- **No mmap anywhere**: pread → pinned staging → H2D → free, per block.
- **Manifest is the contract**: logical names/shapes only; probe shard
  headers, never HF index.json; two-way audit before compute.
- **Numerics policy** (user-signed): cross-impl equivalence is
  TOLERANCE-based; within-impl determinism per (impl, S, schedule); the
  teacher-forced canonical test (gate 0.75) is the equivalence gate.
- **Threading**: model loop strictly sequential; intra-op parallelism is
  each impl's private business; host OpenMP static only (user decision).
- **C++20 pinned**. Minimal core deps: nlohmann/json private in .cpp
  only; GoogleTest via FetchContent; PCRE2 via dlopen. Device TUs
  compile with hipcc.
- **Don't touch `docs/session-api.md`** (user-owned draft).

## Workflow

```sh
cmake -S . -B build && cmake --build build -j      # Debug: iterate here
ctest --test-dir build                             # must stay 100% green
cmake --build build-release -j                     # Release: measure here
```

- Tests run pinned at `OMP_NUM_THREADS=48` (gtest_discover_tests
  ENVIRONMENT in CMakeLists — the single discoverable place) and require
  a GPU, as the engine does.
- Perf numbers only from `build-release`; state thread count, numactl
  placement, and context length with every number — each flips winners.
- **Stale-binary trap**: after editing CMake/source, rebuild BOTH trees
  before benchmarking (build-release does not track build).
- Canonical fixtures (`tests/canonical*`) are committed; regenerate only
  deliberately via `tests/python/dump_canonical.py` (needs `.venv`,
  `HF_HUB_OFFLINE=1`). Ad-hoc model validation: `CanonicalEnv` gtest with
  `ALICE_CANONICAL_{MODEL_DIR,MANIFEST,FIXTURES[,CHUNK]}` — no recompile.
- Manifests are generated, not hand-written: `tools/gen_manifest.py`;
  pre-flight audit with `tools/check_manifest.py`.
- Commits: detailed messages (what + why + measured numbers); update
  `docs/engine-design.md` status paragraphs with every feature.
- Questions are questions: never turn a user's design question into a
  code change unless asked.

## State of play (2026-09)

- 54/54 tests. Validated end-to-end vs HF on all three models
  (teacher-forced canonical, gate 0.75; max drift 0.22/0.28/0.68 for
  1B/8B/70B — watch item: drift grows with depth).
- Pre-migration perf (48T): 1B decode 52.9 tok/s short-context (GPU FFN)
  → 26 tok/s at 630 ctx (the single-threaded CPU attention KV pass,
  36 µs/context-token, dominates); 8B decode 14.5 short / 6.5 @630 ctx;
  prefill attention-bound everywhere. Attention on device is the single
  biggest lever the migration unlocks.
- Migration sequence (per design §3/§8): device stream + Trunk SIMT
  kernels → FFN blocks flipped to device-io → attention device kernel →
  GPU lm_head + on-device sampler → retire CPU paths (AVX2 kernels to
  git history; CPU_GENERIC to dev-reference role) → 70B multi-die
  placement → §11 quantization.
- Kernel lineage (pre-migration): FFN CPU_AVX2 = tinyBLAS-style
  register-tile GEMM (matched OpenBLAS sgemm, 1.2 TFLOP/s, so NO CPU
  BLAS dependency); GPU FFN = rocBLAS gemm_ex bf16→fp32-HPA + fused
  gate/up + silu-mul.

## Hardware context

AMD EPYC 7413 (Zen3, AVX2 no AVX-512), 2×24 physical cores SMT-2,
NUMA node0 = cpus 0-23,48-71 (257 GB), node1 = 24-47,72-95 (254 GB).
4× MI250X = 8 GCDs × 64 GB HBM2e (~1.6 TB/s each, gfx90a:sramecc+:xnack-),
ROCm 7.1 system packages (rocBLAS 5.1, hipcc, HIP runtime). Resizable
BAR on (64 GB aperture/GCD); xnack off. Pinned host buffers mandatory
for DMA speed (pageable runs a staged path: measured 2.2 GB/s vs 28.5
GB/s pinned). No sudo for agents (sudo needs interactive auth) — ASK the
user to run package installs. /tmp is a 250 GB tmpfs and NOT durable
storage (machine reboots wipe it). Fresh configures fetch googletest +
nlohmann_json over the network (FetchContent) unless system packages
provide them.
