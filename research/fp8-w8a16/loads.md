# Real loads, phase 1: deliberately unhidden wait baseline

The register-only checkpoint was committed as **`8e1c120`** before adding this
experiment. The user accepts the expected lower clock under additional math
load and requests this sequence: **add loads and report with bubbles first;
then address the per-MFMA gap; only later tune the backedge**.

This note covers only the first step. No prefetch-depth search, wait
relaxation, address-generation tuning or boundary redistribution is performed.

## What is loaded / exact schedule

This is real device global-memory traffic over **synthetic, changing weight
data**, not real model weights or a general tiled GEMM. Activations are still
FP16 values already in registers, and accumulation/output is FP32.

Each wave computes a 32x32 output tile using `32x32x8f16` MFMA. Each consumed
weight fragment is four FP8 values per lane, packed in one dword; all 64 lanes
read adjacent records (256 B useful wave payload). The FP16 control reads the
same decoded values from two-dword records (512 B/wave).

Two ready FP16 operand banks are primed from steps 0/1 before the core-clock
body. For steady-state logical step s:

```text
MFMA(activation, ready_bank[s % 2])
form address of weight fragment s+2
issue global load of that fragment
s_waitcnt vmcnt(0)                    // intentionally immediate full drain
convert loaded FP8 into ready_bank[s % 2]
```

The FP16 stream has the same schedule but no conversion. A real load and
wait are inside a single volatile assembly block, and machine scheduling
barriers retain MFMA -> load/wait -> conversion order. There is **at most one
outstanding weight load per wave**. The second ready bank protects operand
lifetime; it does not hide memory latency, because the wave waits before
issuing its next MFMA even though that next operand is already ready.

The register-only controls use the first logical weight fragment repeatedly:

| Mode | Data path |
|---:|---|
| 0 | Resident FP16, raw MFMA |
| 1 | Resident FP8, ordinary after-MFMA 8/8 conversion |
| 2 | Streamed FP16, one load + immediate wait per MFMA |
| 3 | Streamed FP8, one load + immediate wait + eight conversion VALUs per MFMA |

All modes explicitly establish initial accumulator/operand readiness before
the core timestamp. **The new streamed path intentionally does not carry
over the checkpoint's 10/6 boundary split**: backedge tuning is deferred by
request. The resident FP8 control is therefore labeled 8/8, not falsely
presented as the committed optimized checkpoint.

The older bring-up `probe.hip` also has a distinct eight-load batch/drain
control (~46 TFLOP/s in its retained runs). This new one-load/immediate-wait
baseline is intentionally weaker and is not claimed as an improvement over
that batching experiment. Original programs and measurements are preserved.

## Data coverage, footprints and correctness

There are 17 distinct signed weight tiles. Per-wave/per-step phase is:

```text
phase = (step + floor(step/17) + 7*wave + 3*epoch) % 17
```

The resulting 289-step pattern and wave phase make constant-weight reuse,
common offset/stride mistakes, and stale data detectable. Device initialization
writes both formats outside timing. The CPU independently counts the phases
of the **consumed first `steps` fragments**, maps every MFMA output register
to its row/column, and computes an FP64 reference. No timed repacking occurs.

Weights include FP8 zeros, subnormals and normals, with byte magnitudes 0..31;
activations are multiples of 1/8 in [-3/8,3/8]. Products have denominator 4096.
At the supported lengths <=8192, an absolute bound on the sum's integer
numerator is `8 * 8192 * 180 = 11796480 < 2^24`, so FP32 accumulation is exact
for these inputs. **Any nonzero output error is rejected.** This is a bounded
correctness workload, not coverage of arbitrary long-K accumulation or all
possible model weights. The existing conversion-primitive tests cover the
finite E4M3FN encodings separately.

Each wave owns disjoint physical addresses. Allocate **steps+2** fragments
per wave: the first two prime the operand banks, and the last two steady-state
loads prepare unused future operands. This avoids out-of-bounds lookahead
without an in-loop tail branch.

| MFMAs/wave | FP8 allocation | FP16 allocation |
|---:|---:|---:|
| 4096 | 416.203125 MiB | 832.40625 MiB |
| 8192 | 832.203125 MiB | 1664.40625 MiB |

The larger sets are well beyond cache capacity, but no physical-HBM traffic
counter or memory-latency attribution is claimed. Reported bandwidth is useful
shader-requested weight payload divided by event time. A second field includes
the two priming reads: requested/useful ratio `(steps+2)/steps`. It still does
not count cacheline amplification, output traffic or physical HBM transfers.
Resident controls report zero **steady-state weight-stream** bandwidth, not
zero kernel memory traffic.

Before timing each process:

- Run all 12 configurations with epoch 1, then epoch 0, without recompiling.
- Poison all output values before every validation launch.
- Check every output and 16 guard floats on each side.

All **72** changed-input/poisoned-output validation cases across the retained
three processes pass. Every output of the final launch in all **252 timed
samples** also matches FP64 exactly; guards are unchanged. No timing result
is excluded for incorrect numerics.

## ISA / resources

`audit_load_probe.py` inspects all 12 entries from the **actual executable**:

- Exactly U MFMAs per backedge at U=4/8/32.
- Streamed FP8: exactly U `global_load_dword` instructions, each immediately
  followed by `s_waitcnt vmcnt(0)`, then the eight-VALU conversion.
- Streamed FP16: exactly U `global_load_dwordx2` instructions, each immediately
  followed by the same full wait; no FP8 conversion.
- Resident controls have no in-loop weight loads.
- Zero private/scratch bytes, zero LDS, no SGPR lane-spill transfers.
- Address instructions and other vector work are counted rather than hidden
  in the conversion total. At U=8, each loaded fragment requires **two address
  VALUs**, in addition to FP8's eight conversion VALUs.

U=8 reported resources:

| Mode | VGPRs | AGPRs | SGPRs |
|---|---:|---:|---:|
| Resident FP16 | 28 | 16 | 18 |
| Resident FP8 8/8 | 28 | 16 | 19 |
| Streamed FP16 | 32 | 16 | 18 |
| Streamed FP8 | 28 | 16 | 19 |

Streamed FP8 U=32 reports 42 SGPRs instead of 19, still no private storage or
lane spills. This compiled-addressing difference is retained, not tuned away.
These resource fields are not an observed occupancy/assignment trace.

## First measurements — no optimization yet

Native HIP O3/C++20, gfx90a, system ROCm 7.1 / Clang 21.1.8. HIP logical
device 0, 104-CU MI250X GCD, CPU and memory bound to NUMA node 0,
**no model or attention context**, no clock/power overrides.

104 CTAs x 256 threads, 416 waves (nominal four waves/CU). Two independent
nine-round processes at 4096 MFMAs/wave, seeds 1729/2718; three rounds at
8192, seed 31415. Warm all entries and condition with all four U=8 modes for
>=2 seconds. Randomized case order; eight event-timed launches/sample;
per-wave core/steady intervals from the final launch. End timestamps do not
explicitly drain the final MFMA. Small endpoint effects explain raw-resident
intervals slightly below 64; this is not faster-than-nominal arithmetic.

Median results, **U=8**, 4096 MFMAs/wave:

| Mode | Core cycles/MFMA, run 1 / run 2 | TFLOP/s, run 1 / run 2 | Useful weight GB/s, run 1 / run 2 |
|---|---:|---:|---:|
| Resident FP16 | 63.991 / 63.991 | 151.461 / 151.363 | — |
| Resident FP8, 8/8 | 64.502 / 64.502 | 146.072 / 146.026 | — |
| Streamed FP16 | 612.697 / 612.165 | 17.630 / 17.667 | 550.928 / 552.085 |
| Streamed FP8 | 584.754 / 586.015 | 18.514 / 18.433 | 289.274 / 288.017 |

Streamed paths run at effective core rates ~1628–1630 MHz versus resident
~1433/1475 MHz; these are measured core/steady-counter ratios, not clock
settings. Neither the old ~145 TFLOP/s checkpoint nor the new resident rate
is used to infer memory-stall cycles by assuming a fixed clock.

U=4/32 streamed FP8 gives ~594/~580 cycles/MFMA in run 1, ~593/~580 in run 2.
The 8192 control is similar: U=8 FP8 ~583.579 cycles/MFMA, 18.505 TFLOP/s,
289.136 useful GB/s; FP16 ~613.473, 17.672 TFLOP/s, 552.257 useful GB/s.
All cases, min/max intervals and both footprint sizes are retained in CSVs.

We do **not** fit these memory-stalled results to a precise fixed backedge
constant: per-sample/per-wave memory variation and compiled address generation
are still present. U variation is diagnostic, not a selected optimized dispatch.

## Interpretation / required stopping point

The dominant new gap is **per MFMA**: roughly 585 core cycles versus ~64
for resident arithmetic. The immediate wait visibly serializes each future
load with the next instruction stream. Some current-MFMA work may overlap
the load, but the ready next operand is not used while the wave is waiting.
The measurements do not separate physical DRAM latency from cache, request
fragmentation, translation, addressing or wave scheduling effects.

This one-use stream has **64 FLOPs/useful weight byte** for FP8 and 32 for
FP16. Consequently it is not a promise that arithmetic-peak TFLOP/s can be
reached with no weight reuse, even after latency hiding; a full GEMM's reuse
strategy is a separate question. The present payload bandwidth is not a
measured memory roofline.

**Stop here as requested.** Next phase is the per-MFMA/load gap; only after
reporting that phase should the algorithm be adjusted for the backedge. No
read-ahead depth tuning, wait relaxation, partial conversion movement,
address-mode optimization, LDS sharing or production change is included here.
The accepted register-only checkpoint remains `8e1c120`; the load baseline is
separate working-tree research for review.

Following this baseline report, the user asks for 2/4/8-deep raw-weight
queues while continuing to idealize A availability. Those separate results
are in [pipeline.md](pipeline.md); this immediate-wait source/data remain
unchanged as the preserved first phase.

## Reproduce / artifacts

```sh
bash research/fp8-w8a16/build-load-probe.sh
python3 research/fp8-w8a16/audit_load_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/load/load-probe 4096 9 1729
# Repeat: 4096 9 2718. Larger footprint/control: 8192 3 31415.
python3 research/fp8-w8a16/analyze_load_probe.py
```

Stdout is CSV, stderr records footprints/protocol and validation. Flags,
source/executable/object hashes, audit, raw data and summaries are in
`results/load/`; full ISA, hot loops, metadata and binaries are under
`temp/fp8-w8a16/load/`. The analyzer reads the retained runs, verifies exact
correctness/completeness and payload accounting, and intentionally performs
no backedge fit. Engine sources/CMake are unchanged; existing engine tests
were not rerun for this isolated research extension.
