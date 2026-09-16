# ExLlamaV3 study — architecture comparison vs alice-infer

2026-09, based on a master snapshot (repo mirrored at
`~/projects/inference/exllamav3`, tarball from codeload; github.com:443 is
blocked from this box, raw/codeload are not). ExLlamaV3 is turboderp-org's
single-user local-inference engine: Python + pybind CUDA extension,
NVIDIA-only (CC 8.0+; JIT-builds the extension at first import unless a
prebuilt per-torch+CUDA wheel fits), quantization-first (EXL3), broad HF
model support (73 architecture files), continuous batching + TabbyAPI server
integration. It is the closest philosophical sibling to alice: own hot-path
kernels, single-user focus, consumer hardware. The differences are as
instructive as the agreements.

## How it does things, by subsystem

### Substrate & packaging
Python/torch is the tensor runtime; the extension (`exllamav3_ext`, ~13.6k
lines of .cu) holds the hot kernels. torch >= 2.6, CUDA >= 12.4; PyPI
package JIT-compiles at first import per torch version ("a few minutes
once"); release wheels are published per (CUDA × torch × python) combo —
the same packaging-matrix strategy as FlashInfer's cubins, same "rack of
servers + IT squad" smell.

### Quantization (the project's center of gravity)
EXL3 = QTIP variant (arxiv 2406.11235): trellis-coded weight-only quant,
procedural codebook, ~2–8 bpw, quality benchmarks in the README are the
project's identity. The GEMM is hand-written CUDA
(`exllamav3_ext/quant/exl3_gemm*.cuh`) with the QTIP Hadamard rotation
**fused into the kernel** via cooperative-groups `grid.sync()` between
in-kernel phases (persistent/cooperative launch). Two regime paths,
dispatched on rows (their S): `rows <= AUTO_RECONSTRUCT_THRESHOLD (144)` →
direct quantized GEMM (weights stay packed, activations fp16); above →
reconstruct (dequantize to fp16 with Hadamard folded in) then cuBLAS-class
`hgemm`. Same shape as our `S > 4` attention clause, measured threshold.
There is a dedicated `exl3_gemv_int8` path for decode.

### Kernel selection — validates design §15 almost feature-for-feature
- `exl3_kernel_map.cu: select_gemm_shape(cc, size_m, size_k, size_n, K,
  multi, ...)` — a hand-written cascade of per-compute-capability clause
  sets (CC_OLD/AMPERE/ADA/...) over mod-256/mod-512 alignment and shape
  ratios, returning a tile-shape index. Exactly our "code-time measured
  clauses" pattern.
- `quant/coop_autotune.cu` — a **disk-persistent autotune cache** for
  cooperative-launch parameters: `EX3ATUNE` magic, format version,
  `COOP_AUTOTUNE_VERSION` folded into a salted hash, per-shape records of
  {kernel, block_dim, num_sms, concurrency, latency samples}. This is the
  measured-decision cache we designed in §15, implemented independently.
  Read its format before building ours.

### Attention
Own **Triton** kernels are the preferred path (`attention_fn/triton_paged.py`:
paged decode/prefill/varlen, any head_dim ≤ 512 zero-padded to pow2,
quantized-cache-direct variants with online dequant). Then an ordered
**fallback chain**: older Triton → bighead-scalar → xformers → torch SDPA
(`attention_fn/dispatch.py`), first match wins, with an elaborate
no-match report function as the debugging cost. Notably: **no FA2/FA3, no
FlashInfer** — a (mostly) single-author project writes its own attention
rather than pay the libraries' dependency weight. Triton kernels JIT at
runtime per specialization (triton package required).

### KV cache
Paged (`generator/pagetable.py`, block tables — needed for continuous
batching), fp16 default, optional **quantized cache** (2–8 bit K and V
separately, `cache/quant.py`) attended either directly (online dequant in
the Triton kernel) or via dequant staging (`EXL3_QC_STAGING` env levels —
an A/B knob in the same spirit as our retired ALICE_FORCE_ATTN_V1).
Recurrent-state caches for GDN/Mamba hybrids.

### Tensor parallelism
**Multiprocess**: one Python process per GPU, pipe/pseudo-connection
orchestration, a CPU worker slot for helpers (`model/model_tp.py`).
Modules declare splits (`tp_export`/`tp_import_split*`), an allocation
planner balances VRAM (`model_tp_alloc.py`). Two collective backends
(`model_tp_backend.py`): NCCL, and **native** = GPU-mappable shared pinned
host memory + spin-wait sync (`parallel/ll.cuh`, `ldg_acquire_sys_u64` +
`__nanosleep` backoff) + **CPU-assisted allreduce with AVX2/AVX512**
(`parallel/all_reduce_cpu*.cpp`) — deliberately paying PCIe traffic to
avoid NCCL for small latency-bound decode messages on consumer rigs.
Direct validation of our hand-rolled-P2P-first stance (our single-process
threads will be simpler than their cross-process sync).

### CUDA graphs
`graph.cu` + `libtorch/*_bc.h` "BC" wrappers: decode-time module graph
capture (stream capture → cudaGraphExec replay). The constraint propagates
visibly — e.g. the exl3.py contiguity assert "a silent copy here would
hide a hot-path inefficiency and break CUDA-graph address stability".
Their motivation is torch's per-op overhead; our static-allocation design
already guarantees address stability, so capture should be cheap when we
reach the launch-latency floor (queued item).

### Loading
`stloader.cpp`: planned sequential **runs** over safetensors files —
contiguous reads spanning multiple tensors, small gaps (< 512 KB)
read-through-and-discarded because "a few KB of dead read is much cheaper
than losing sequential access", multithreaded, deferred loading for
cross-device scatter. Same philosophy as our pread → pinned → H2D, more
engineering; no evidence of mmap reliance on the hot path.

### Scope beyond alice
Continuous dynamic batching, speculative decoding (draft models + ngram),
multimodal, fused GPU sampling stack (DRY, rep-pen, gumbel tricks in
`generator/`), 73 architecture files behind a registry, HF-config-driven,
OpenAI server via TabbyAPI. `doc/env_vars.md` alone is 693 lines — the
configurability bill for serving arbitrary models/hardware.

## Where it validates alice

1. **Own kernels for the hot path** — attention and quantized GEMM are
   hand-written; libraries are fallback/reconstruct-path only.
2. **Measured, versioned, disk-cached autotuning** (§15 cache) —
   independently invented, down to the version-salted hash.
3. **Hand-written per-arch dispatch clauses** (§15 code-time culling) —
   `select_gemm_shape` is our v2/v4 clause discipline at larger scale.
4. **Regime split on S with a measured threshold** — their 144 rows ≈ our
   S>4.
5. **Dodge the vendor collective library for small messages** — their
   native TP backend vs NCCL ≈ our P2P-first vs broken-Canonical-RCCL.
6. **Graphs at the decode launch floor** — already on our queue.
7. **Sequential file access, explicit staging** — matches our no-mmap
   invariant.

## Where we differ, deliberately

- **Fallback chain vs loud failure.** Their first-match-wins chain with
  silent degradation is the price of arbitrary hardware × arbitrary
  models; our no-impl-for-device = load error stance stands. Their
  no-match *report* function shows the debugging debt fallback chains
  accrue.
- **Python/torch substrate.** They pay: per-torch-version JIT builds, ABI
  fragility (see the `unpack_bf` workaround comments), graph
  address-stability contortions, and torch's op-dispatch overhead that
  forces graph capture in the first place. We pay: everything ourselves.
- **Quantization-first vs fp32-stream-first.** Their whole engine assumes
  packed weights; ours keeps an fp32 stream with bf16 GEMM operands, §11
  quantization ahead. When we get there, EXL3/QTIP (fused-Hadamard
  cooperative GEMM, trellis codebook, gemv_int8 decode path) is the
  reference implementation to study — and their 144-row threshold says
  our quantized regime split will need its own measurement.
- **Paged + batched vs contiguous + single-sequence.** PageTable exists
  for continuous batching, a non-goal for us; our append-only contiguous
  KV stays. Their quantized-KV-direct attention is the pattern to copy
  when bf16/fp8 KV lands here.
- **Multiprocess TP vs single-process.** Theirs fits Python (GIL, per-GPU
  CUDA contexts); ours is threads over hipSetDevice — but their
  CPU-assisted AVX allreduce over GPU-mapped pinned host memory is a
  candidate small-message path for our 8-GCD box if P2P latency
  disappoints at decode sizes.

## Actionable takeaways

1. §11 quantization: read the QTIP paper (2406.11235) +
   `exl3_gemm_kernel.cuh` / `exl3_gemv_int8*` before designing ours.
2. §15 measured-decision cache: crib the on-disk format from
   `coop_autotune.cu` (magic, format version, version-salted shape hash,
   launch-param record).
3. TP (70B bring-up): study `model_tp_backend.py` native backend +
   `parallel/ll.cuh` sync and `all_reduce_cpu*`; decide P2P vs
   host-pinned-CPU-reduce per message size, measured.
4. CUDA/HIP graphs: our static allocation already satisfies address
   stability; capture is a small step when decode overhead campaign
   reaches the launch floor.
5. Nothing here changes a committed alice decision.

## Caveats

Single-lead-author project, fast-moving (dev branch ahead); snapshot is
one point in time. NVIDIA-only: none of its kernel *performance* numbers
transfer to gfx90a — only structures and strategies do. Findings from
source reading, not from running it (no NVIDIA GPU here; building it on
ROCm is not supported upstream).
