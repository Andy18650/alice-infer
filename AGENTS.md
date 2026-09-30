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

**Migration state (2026-09):** the GPU-resident path (design §3) is the
ONLY execution path — the CPU production branch is RETIRED (host
Model/Trunk, CPU_AVX2, host-io GPU FFN + hipRTC, HardwareClass
factories all deleted to git history). cpu_generic survives as the
dev-reference oracle + `gpu::CpuReferenceBlock` compute core (swap-in
via `ALICE_CPU_REF_BLOCKS=<indices>`). The 70B is intentionally dark
until multi-die (it never fit one GCD).

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
- **Typed tensor is the host-side vocabulary**: move-only owning
  `Tensor<T>` (the former TView/Staging pair merged); `File::check_as<T>()`
  as the one checked runtime→static conversion. Device buffers are plain
  pointers, not Tensors; borrowed host access is ptr + dims.
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
- **C++20 pinned**. Use industry-standard dependencies when appropriate;
  the CMake script is the source of truth for what the project links.
  (Rationale, user 2026-09: compile time is not a concern at this scale.
  The engine will be library-like in the GGML/libcurl sense — consumed by
  projects building ON it (language servers, chat frontends), not as a
  general-purpose utility — so API/ABI churn between versions is
  acceptable; consumers recompile. Do not re-implement what a standard
  library provides.) Device TUs compile with the HIP toolchain.
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

## State of play (2026-09, GPU-resident v1)

- Tests (post-retirement: tensor/safetensors/manifest/tokenizer,
  cpu_generic oracle kernel+block suites, GPU-resident kernel/block/
  model/canonical suites incl. long-case). Canonical: short fixtures
  1B 0.33 / 8B 0.47 (bf16 ref, gate 0.75); long fixtures (WikiText
  ~1500 tokens, fp32-ideal reference, user-signed 2026-09) mean 0.018 /
  max 0.76-1.52 (gates 0.05 / 2.0), greedy with coin-flip rule. Long
  fixtures regenerable (`dump_canonical.py --prompt-file <raw>
  --dtype fp32`), gitignored. The semantic-exactness proof (CPU engine
  0.0002 vs fp32 ideal) is recorded in design §13.
- Perf (Release, single GCD, greedy, default host placement): 1B prefill
  7826 tok/s at 602 ctx / 8715 at 3640 ctx (pre-v4: 3652 / 885); 8B prefill
  2182 at 3640 ctx (was 398). 1B decode 261.7 tok/s short-ctx / 159.6 at
  3770 ctx (was 114.6 / 86.1 pre-GEMV); 8B decode 64.9 at ~135 ctx / 47.1
  at 3770 (was 35.1 / 28.2). Decode now runs the S=1 split-K GEMV
  (gpu_rocm family kernel, dispatched at S==1; 933-1276 GB/s per shape vs
  the 1290 GB/s measured read ceiling — the gemm_ex path it replaced ran
  313-1143). Remaining decode budget: lm_head still on gemm_ex (1.1
  ms/token), the SIMT tail (~2 ms), launch/sync overhead — the launch-diet
  half of the campaign; prefill is GEMM-bound again.
- Attention kernels (gpu_rocm private TU; v1 RETIRED — every dispatch
  decision is measured in research/prefill-attn-ab): S>4 → v4 split-T
  prefill (thread-per-query, key-major LDS score tile, deferred exp,
  partials in a loader-allocated SHARED scratch — first §14 scratch-pool
  instance — plus combine); S≤4 → v2 flash-decode at ALL T (split-T,
  in-warp shuffle reductions + combine; beats v1 even at T=8). The v1
  one-block-per-(query,head) kernel is deleted.
- S=1 GEMV (2026-09, research/decode-roofline): both gpu_rocm blocks
  dispatch S==1 to a family-private split-K GEMV (warp-per-row, uint64
  bf16 loads, partials + combine, in-register operand quantization so the
  separate f32_to_bf16 launch disappears from decode); gemm_ex stays for
  S>1. CTA target 4096, swept (1024/2048/4096/8192 -> 13.37/13.37/12.99/
  14.44 ms/token floor). Same kernel family is the model for the lm_head
  site (still gemm_ex, ~1.1 ms/token) when that is worth taking.
- Next: 70B multi-die (needs ≥3 GCDs; the TP/PP research is in
  research/rccl-single-process/), then §11 quantization; on the horizon:
  MFMA variant of the v4 tile structure (hd=128), fp16/bf16 KV cache,
  decode-overhead campaign (GEMV + launch diet).
- RCCL on this box: Canonical build broken (any ≥2-rank collective
  segfaults — vendor unit tests confirm); AMD-overlay RCCL validated as
  replacement; hand-rolled P2P validated (134.7 GB/s intra-card). See
  research/rccl-single-process/README.md and research/rocm-overlay-method.md.

## Hardware context

AMD EPYC 7413 (Zen3, AVX2 no AVX-512), 2×24 physical cores SMT-2,
NUMA node0 = cpus 0-23,48-71 (257 GB), node1 = 24-47,72-95 (254 GB).
4× MI250X = 8 GCDs × 64 GB HBM2e (~1.6 TB/s each, gfx90a:sramecc+:xnack-),
ROCm 7.1 system packages (rocBLAS 5.1, hipcc, HIP runtime). Resizable
BAR on (64 GB aperture/GCD); xnack off. Staging is THP-backed mmap
(2026-09 revision: pageable H2D measures ~20 GB/s on this stack and
hipHostMalloc pin/unpin costs more than it saves; the real win is
MADV_HUGEPAGE first-touch faults — see design §9). No sudo for agents (sudo needs interactive auth) — ASK the
user to run package installs. /tmp is a 250 GB tmpfs and NOT durable
storage (machine reboots wipe it). For transient-but-should-survive
artifacts (packed smoke models, benchmark outputs), use the repo-local
`temp/` — gitignored, durable on the repo's disk, wiped only by
explicit choice. Fresh configures fetch googletest +
nlohmann_json over the network (FetchContent) unless system packages
provide them.
