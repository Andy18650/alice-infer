# Compute-bound synthetic GEMM and clock audit

User direction: establish a strong compute-bound GEMM before further layout
or fine-grained shape searches. BF16 activations/weights, FP32 accumulation/
output; actual full non-repeating synthetic matrices, one kernel, no split-K.
Standalone, no production changes or clock/power configuration changes.

## Main finding: the raw reference and GEMM have different effective clocks

The best observed real GEMM reaches ~144 TFLOP/s, versus ~167 TFLOP/s for a
register-only 32x32x8 BF16 MFMA reference. That apparent ~14% utilization gap
is **mostly a cycle-supply-rate difference**, not unaccounted MFMA idle time.

A diagnostic build timestamps every wave with HIP `clock64()` (core cycle
counter) and `wall_clock64()` (fixed-rate steady counter). Rates below are
median core-cycle/steady-time ratios, corroborated by firmware's average GFX
clock, not the nominal/current field that remains 1700 MHz.

Native O3 HIP/C++20, gfx90a GCD 0, 104 CUs, CPU+memory NUMA node 0,
no model/attention context, automatic clocks. Nine measured rounds:

| Workload | TFLOP/s | Effective GFX MHz | Busy SMI average GFX MHz |
|---|---:|---:|---:|
| Register MFMA, low-magnitude constant operands | 166.50 | 1570.82 | 1570 |
| Same four-chain MFMA loop, normal signed BF16 operands | 151.05 | 1425.56 | 1425 |
| Working V3 GEMM, full independent matrices | 144.13 | 1385.22 | 1384 |

The random reference changes only operand values, not the four-MFMA inner
loop structure; input generation is a tiny prologue. Its core-cycle count
is nearly unchanged while TFLOP/s and effective clock both fall ~9.3%.
This directly demonstrates that the original reference is not a workload-
independent sustained rate. It is not proof of a specific power-management
mechanism, power cap, channel mapping, or a guaranteed fixed-frequency limit.
Throttle masks are zero in the captured samples; this does not establish
absence of normal clock/power regulation.

Scaling the measured reference by the GEMM clock ratio gives **146.83
TFLOP/s**, versus 144.13 achieved: **98.17% of that clock-adjusted reference**.
Using the normal-signed-data reference independently gives 146.78 TFLOP/s
and 98.20%, so this conclusion does not depend on the low-magnitude operand
choice after the effective rate is accounted for.
Per-wave FLOPs/core-cycle also agree: ~255.30 for the reference versus ~250.33
for GEMM. The diagnostic working kernel retains 508 reported registers,
64 KiB LDS, zero private segment and one resident CTA/CU; its ~144.13 rate
matches the uninstrumented ~144.2 result closely. Timestamp stores are outside
the inner loop. This supports near-arithmetic-bound operation in the supplied
core cycles, not a claim that general GEMM sustains the raw 167 TFLOP/s under
all inputs/clock states.

A short arithmetic probe immediately after GEMM rapidly returns to ~1571 MHz
inside its wave lifetime. Its whole-call rate is ~150 TFLOP/s because launch/
admission/timestamp overhead is significant for the short probe. It does NOT
hold the preceding GEMM's clock state and is not substituted as that ceiling.
No clock locking, power-cap changes, package installs or system-policy changes
were attempted.

## The working kernel

Best path: CK Tile compute-optimized V3, 256x256x64 output/input block tile,
four waves/CTA, 32x32x16 warp GEMM (native gfx90a 32x32x8 instructions),
register prefetch plus one LDS buffer set. Both operands are panel-packed:

```
A: [K_panel][M_tile][M_inside][K_inside=64]
B: [K_panel][N_tile][N_inside][K_inside=64]
```

For these aligned shapes each input panel read by a CTA is contiguous. This
is a best-case synthetic storage experiment whose panel sizes match the
compute geometry, NOT promotion of a production activation format or the
previous shared 32x128 weight format. Affine and B-only-panel controls remain.

Problem dimensions are M=grid_m*tile_m, N=grid_n*tile_n. Grid 10x10 means 100
CTAs in ONE launch, not 100 launches. Tested grids include 100,104,208,416
CTAs; K=4096..32768. The register MFMA reference is measured separately for
native 16x16x16 and 32x32x8 instructions. Repetitions, eager/graph and isolated/
queued protocols remain distinct. All actual GEMM outputs pass FP64 sampled
oracles, small all-output tests, poisoned graph output, guards and within-
implementation repeatability. Initialization/packing is outside timing.

Initial screening: basic CK V1 ~125 TFLOP/s, optimized V3 ~141, long-K V3
~144, classic CK V3 ~137, double-LDS V4 ~106--122. Native-K=8 warp variants
were worse; the wide one has 124 private bytes/thread. Warp16 wide variants
spill 332 bytes/thread. The upstream scheduling flag control did not improve
the tested cases. The no-LDS direct four/eight-stage register-ring real GEMM
passes correctness but only reaches ~108--110 TFLOP/s, showing that removing
LDS also sacrifices operand feeding/reuse. Those exploratory outcomes are
preserved, not used to assert a unique bottleneck.

## Affine versus packed operands: confirmed ordering, not a layout law

The repeated control rows support **AB-panel > affine > B-panel** at all nine
confirmed problems when each storage variant uses its better measured CTA
order. For example, M=4096,N=6656,K=8192 with the same n-fast order gives
**137.88 / 131.12 / 140.78 TFLOP/s** for affine / B-panel / AB-panel (warm graph
batches of 16, two processes x nine rounds, same GPU/NUMA placement as above).
The original result CSV contained these controls; the explicit table and
interpretation are now in [layout-comparison.md](layout-comparison.md).

This is not a clean memory-layout-only comparison: the wide B-panel variant
has **16 private bytes/thread**, with a spill reload inside its K loop;
affine and AB-panel have zero. Their reported register counts are 512/512/508
respectively, all with 64 KiB LDS and one CTA/CU. Descriptor/address codegen,
physical access patterns and CTA order all change; no control isolates their
individual contributions or measures clocks for every layout.

There is also a fixed-order exception: at M=4096,N=6656,K=32768, n-fast affine
falls to 130.64 versus B-panel 134.56 and AB-panel 144.32; m-fast affine reaches
142.90. Do not promote the better-order ranking to a universal fixed-order or
asymmetric-tiling rule. These panels differ from the shared 32x128 B format.

## Audit of what works, not another variant search

The HSA code object extracted from the measured native-11 executable is
byte-matched to its object-file bundle. The latest build's device instruction
`.text` is byte-identical even though ELF metadata differs. Audit focuses on
the actual AB-panel K-step backedge, not a larger backward control-flow edge
that includes prologue/tail/output code.

Per wave, the hot loop has:

- 128 `v_mfma_f32_32x32x8bf16_1k` instructions;
- 16 wide `buffer_load_dwordx4` operations;
- 16 `ds_write_b128` and 32 `ds_read_b128` operations;
- 16 `vmcnt(15)` waits, no full global drain in the hot loop;
- two `lgkmcnt(0)` waits and two CTA barriers.

Global data staging, fresh global loads and current register-operand MFMA
are interleaved. 106 MFMA instructions precede the second barrier; remaining
math overlaps next LDS prefetch. The working pipeline is therefore not a
load-all/wait-zero/compute-all serialization. Static issue order is NOT a
measurement of stall cycles; wait thresholds don't tell how many requests
were actually outstanding at runtime.

ELF resources: 64 KiB LDS, zero private bytes, 25 SGPRs, 256 AGPRs and reported
VGPR count 508. HIP reports 508 too; these resource fields must not be blindly
summed into another register count. LDS alone limits residency to one CTA/CU
(four waves). More queued CTAs do not provide concurrent second-CTA latency
hiding for this kernel. The register reference already reaches ~167 with
104 four-wave CTAs, so residency alone is not an explanation for the raw gap.

Analytic requested traffic per CTA K step: 64 KiB global loads, 64 KiB LDS
writes, 128 KiB LDS reads for 8,388,608 FLOPs. These are instruction-payload
counts, NOT physical HBM traffic. At grid 8x13 the unique current A/B panel
set is only 672 KiB (about 1.3 MiB for two steps), consistent with the intended
outer-product L2 reuse; no hit-rate counter is claimed.

A controlled same-full-problem diagnostic compares earlier kernels at
M=4096,N=6656,K=32768: basic V1 128x128 is ~120 TFLOP/s, ~83% of its own
clock-adjusted reference; V3 is much closer. The V3-128 timestamp probe raises
register usage from 204 to 220, so its diagnostic rate is not substituted for
the uninstrumented benchmark. The strongest clock conclusion uses the wide
V3 probe whose resources and rate remain unchanged. Clock audit findings must
not be extrapolated to every earlier shape/kernel without measuring it.

## Independent clock-verification probes

`operand_clock.hip`, `int8_clock.hip`, `mixed_clock.hip` and `gemv_clock.hip`
independently probe operand-dependent rates, integer instructions, concurrent
loads and SIMT GEMV. Fresh native builds/runs reproduce the clock observations;
assembly confirms live instructions/loads and zero private segments in these
standalone probes. Phase-aligned firmware average GFX corroborates the core /
steady counter ratios while current GFX remains 1700 MHz.

Important scope: mixed-load phases reach ~243-244 rather than the >=250
FLOPs/core-cycle goal, so their loss is not only clock rate. GEMV timing covers
only the partial producer, not combine; its historical GFLOP/s header actually
contains TFLOP/s values (already corrected in source). Its repeated integer
inputs and y[0] check do not establish general GEMV correctness. The detailed
review, reproduction commands, additional uninstrumented `raw_mfma.hip`
reference and retained original/fresh evidence are in
[clock-verification.md](clock-verification.md). These observations do not
identify a particular clock/power policy or measure physical HBM traffic.

## Actual GEMM counter profiling

The existing profiler now collects hardware counters successfully with its
installed Python/SDK/AQL components explicitly configured. Two 12-pass actual
GEMM captures plus three same-pass core confirmations per shape/layout show
long-K AB-panel at **97.70% counter-derived MFMA utilization** (the AMD formula
uses an emulated busy-cycle counter), consistent with the earlier clock audit.
LDS bank-conflict counts are zero; fixed n-fast L2-to-HBM read payload rates are
only ~147-163 GB/s, not near the streaming ceiling. B-panel executes ~6.24%
more VMEM instructions at long K and has more dependency-wait wave cycles.
A same-pass m-fast control restores spill-free affine from 86.88% to 96.57%
utilization: zero spills alone is not sufficient. See
[profiling.md](profiling.md) for exact protocols, raw-counter checks, modelled
reuse, profiler limitations and reproduction commands. Profiled timings remain
separate from the earlier unprofiled performance confirmations.

## Reproduce and retained data

```sh
bash research/gemm-compute-bound/build.sh temp/gemm-compute-bound/new native
numactl --cpunodebind=0 --membind=0 temp/gemm-compute-bound/new/bench \
  --suite test --rounds 1 --repeats 1
python3 research/gemm-compute-bound/confirm.py temp/gemm-compute-bound/new-confirm \
  --binary temp/gemm-compute-bound/new/bench --rounds 9
python3 research/gemm-compute-bound/audit.py \
  temp/gemm-compute-bound/audit/working.o.0.hipv4-amdgcn-amd-amdhsa--gfx90a \
  temp/gemm-compute-bound/new-audit
```

`build.sh` also accepts `ckflags` to test the upstream post-misched/noalias
settings separately. `--impl`, explicit `--gm/--gn/--k`, rounds and repetitions
are supported. `run.sh` snapshots exact executable/source/command/placement.
`CLOCK_AUDIT` is a separate diagnostic compile of only the selected kernel;
normal benchmark builds contain no timestamp instructions.

Full raw data/code objects/executables are under durable repo-local
`temp/gemm-compute-bound/`: `first-six`, `native-9`, `ckflags-9`, `native-11`,
`wk8`, `direct`, `confirmation/{a,b}`, and `audit/{verified,timed,timed-hot,
final-clock,profile128-v1,profile128-v3,clock-run}`. The first broad clock
monitor attach missed the process and produced only a header; it is not
claimed as evidence. Valid monitored runs launched sampling before workload.
Curated summaries are under `results/`; reviewed fresh-build artifacts are
under `temp/clock-review/`. No production or system configuration changes were
made. This checkpoint retains the research, its controls and its limitations.
