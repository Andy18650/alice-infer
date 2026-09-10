# AGENTS.md — working agreement for coding agents

This file is the fast on-ramp for agents working in this repo. The design
rationale and decision log live in `docs/engine-design.md` (14 sections) —
read it before non-trivial changes. The user dictates design; agents
implement, document, and test. When design and code disagree, flag it —
don't silently pick one.

## What this is

`alice-infer`: a from-scratch C++20 LLM inference engine. Single user, ONE
live append-only context, no locking, static memory planning at load.
Safetensors weights only (GGUF rejected), assembled per an externally
authored manifest = the TOTAL contract. CPU-first with hand-rolled AVX2
kernels; AMD GPU (rocBLAS) for FFN blocks, auto-selected when present.
Bring-up models: Llama-3.2-1B-Instruct, DeepSeek-R1-Distill-Llama-8B,
DeepSeek-R1-Distill-Llama-70B (all under `/opt/models`, manifests in
`models/`).

## Hard invariants (do not break)

- **Typed tensor system**: `TView<T>` non-owning rank-2 view, element type
  in the type — no `void*`, no runtime dtype field, no `as<T>()`. The ONE
  checked runtime→static conversion is `File::view_as<T>()` at the load
  boundary. `DType` survives only in file vocabulary (TensorInfo,
  WeightSpec).
- **Block contract**: `Block::forward(F32View io)` — fp32 host [S, d]
  in-place. `StatefulBlock` adds `reset_state()` (+ default-trap snapshots).
  This boundary is why mixed CPU/GPU models compose; keep it fp32 and host.
- **Class structure**: one abstract class per (block type × weight dtype);
  one `final` impl per hardware class; factories resolve with fallback
  chains per block type (AUTO → GPU_ROCBLAS if present → CPU_AVX2 if cpuid
  → CPU_GENERIC). CPU_GENERIC is the permanent naive oracle — never
  "optimize" it.
- **Optimized impls are self-contained**: private weight storage, private
  math, no shared kernel toolbox across impls. "Dispatch on schedule,
  never on arithmetic order." Private copies at construction; staged views
  freed on return.
- **No mmap anywhere**: pread into staging, construct-then-free per block.
- **Manifest is the contract**: logical names/shapes only; sharding is
  physical packaging (probe shards' resident headers; never read HF
  index.json). Two-way audit before compute.
- **Numerics policy** (user-signed): cross-impl equivalence is
  TOLERANCE-based; within-impl determinism per (impl, S, thread count);
  the teacher-forced canonical test (gate 0.75) is the equivalence gate.
- **Threading**: model loop strictly sequential; intra-op parallelism is
  each impl's private business; OpenMP **static only** — no dynamic
  scheduling (user decision).
- **C++20 pinned** (C++23 unavailable). Minimal core deps: nlohmann/json
  private in .cpp only; GoogleTest via FetchContent; PCRE2 via dlopen.
- **Don't touch `docs/session-api.md`** (user-owned draft).

## Workflow

```sh
cmake -S . -B build && cmake --build build -j      # Debug: iterate here
ctest --test-dir build                             # must stay 100% green
cmake --build build-release -j                     # Release: measure here
```

- Tests run pinned at `OMP_NUM_THREADS=48` (set via gtest_discover_tests
  ENVIRONMENT in CMakeLists — the single discoverable place).
- Perf numbers only from `build-release` with `OMP_NUM_THREADS=48`;
  state numactl placement (spread/node0/interleave) with every number —
  placement flips winners (prefill loves node0 locality, decode seeks
  both controllers).
- **Stale-binary trap**: after editing CMake/source, rebuild BOTH trees
  before benchmarking (build-release does not track build).
- Canonical fixtures (`tests/canonical*`) are committed; regenerate only
  deliberately via `tests/python/dump_canonical.py` (needs `.venv`,
  `HF_HUB_OFFLINE=1`). Ad-hoc model validation: `CanonicalEnv` gtest with
  `ALICE_CANONICAL_{MODEL_DIR,MANIFEST,FIXTURES[,CHUNK]}` — no recompile.
- Manifests are generated, not hand-written: `tools/gen_manifest.py`;
  pre-flight audit with `tools/check_manifest.py`.
- GPU work: host-only C++ vs rocBLAS/HIP C APIs (no hipcc in the build;
  custom device kernels via hipRTC, source strings inside the impl);
  `__HIP_PLATFORM_AMD__` define; pinned host buffers mandatory (pageable
  D2H runs a ~1 GB/s staged path).
- Commits: detailed messages (what + why + measured numbers); update
  `docs/engine-design.md` status paragraphs with every feature.

## State of play (2026-07)

- 54/54 tests. Validated end-to-end vs HF on all three models
  (teacher-forced canonical, gate 0.75; max drift 0.22/0.28/0.68 for
  1B/8B/70B — watch item: drift grows with depth).
- Perf (1B, 48T): decode 43.6 (CPU) → 52.9 tok/s (GPU FFN); prefill
  ~157 tok/s (attention-bound). 8B decode 9.5 → 14.5 tok/s. 70B loads
  (17 shards, zero code changes) and validates; untuned ~1 tok/s.
- Kernel lineage: FFN CPU_AVX2 = tinyBLAS-style register-tile GEMM
  (S>1) + multi-accumulator dot GEMV (S=1); matched OpenBLAS sgemm
  (1.2 TFLOP/s) in the lm_head scout, so NO BLAS dependency. GPU FFN =
  rocBLAS gemm_ex bf16→fp32-HPA + hipRTC silu-mul.
- Known next steps: attention optimization (the bottleneck everywhere
  now — CPU-naive, forces GPU idle), GPU lm_head + on-device sampler,
  NUMA-partitioned decode GEMV, 70B shape retuning + multi-die sharding,
  §11 weight quantization (decode byte-reduction), loader overhaul
  (user-owned plan).
- Reference material: `~/projects/inference/reference/sgemm.cpp`
  (llamafile tinyBLAS), Zhihu research notes in session history.

## Hardware context

AMD EPYC 7413 (Zen3, AVX2 no AVX-512), 2×24 physical cores SMT-2,
NUMA node0 = cpus 0-23,48-71 (257 GB), node1 = 24-47,72-95 (254 GB).
4× MI250X = 8 GCDs × 64 GB HBM2e (~1.6 TB/s each, gfx90a), ROCm 7.1.
No sudo for agents (sudo needs interactive auth); system ROCm + Ubuntu
OpenBLAS (.deb extracted at /tmp/openblas-local) available.
