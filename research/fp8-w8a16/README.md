# FP8 E4M3FN W8A16 on gfx90a — first prototype

User-directed standalone research. Goal: consume byte weights at the native
FP16 MFMA arithmetic rate, with FP16 activations already in registers and FP32
accumulation. No engine integration, model format, production packing, LDS,
clock/power configuration changes, or end-to-end speedup claim.

**Current result:** exact finite-E4M3FN conversion, including every subnormal,
is validated on device. The original register probe reaches ~95% of matched
FP16 throughput. The later user-directed [preheader + explicit 10/6 split](splitting.md)
removes the fitted boundary term: **~64.009 core cycles/MFMA**, matching raw
FP16 per-cycle issue rate. Net event throughput is **96.54% / 96.53%** of raw
FP16 (~145 versus ~150 TFLOP/s) at lower effective clocks, not identical
wall-time throughput. The first global-load path remains basic and far from
arithmetic saturation. Original measured sources/binaries are preserved.
The earlier [MFMA-first/conversion-after ordering](ordering.md) alone was
correct but slightly worse (~20 versus ~12 fitted cycles/backedge).
The [four-bank and staged-VALU diagnostic](dependency.md) finds that four
banks do not fix the bubble, and seven vector NOPs already reproduce one
without any MFMA-source dependency. Compiler copies/waits and numerical
hazards in source-write variants are recorded explicitly.

The register-only checkpoint is committed as **`8e1c120`**. The new
[real-load phase-1 baseline](loads.md) deliberately drains each weight load
immediately and leaves ordinary 8/8/backedge bubbles intact. It is a separate
untuned baseline, not a replacement for the accepted register arithmetic.
The user requests per-MFMA/load work before revisiting the backedge.
The next [load-to-convert ring experiment](pipeline.md) tests depths 1/2/4/8
with A treated as freely available: FP8 depth 8 reaches ~86.3/~85.8 TFLOP/s,
about 4.56x/4.55x its co-run depth-1 baseline. No A loading, weight-reuse,
grid or backedge tuning is included.

The [paired real-A experiment](paired.md) now materializes shared packed
A[32,32768] and uses ordered A/B pairs with `vmcnt(2D-2)`. At depth 8 it
retains ~95% of co-run free-A event throughput, with similar ~117 core
cycles/MFMA; all timed and independent A/B-change checks pass exactly.
No reuse, grid or backedge tuning is added.

The latest [13x8 CTA-grid round](grid.md) distributes eight A row streams
across M, retains four waves entirely along N, and naturally shares B across
eight M groups. At equal FLOPs it improves FP8 to ~91–92 TFLOP/s versus the
co-run 104x1 ~82. Aggregate request bandwidth includes cache reuse, not HBM
traffic. No 2x2 wave layout has been implemented.

The subsequent [single-CTA diagnostic](latency.md) pursues only the latency-
coverage route, not 2x2 wave tiling. Its resident anchor is ~64 cycles/MFMA;
the FP8 body is ~100 after in-dispatch pre-read and ~117 after eviction at
D8. D16 does not close the gap; cache-sized warm input still leaves ~96.
No unique CU-local/cache/issue mechanism is assigned.

The real-A paired/grid paths are **fixed-shape packed W8A16 GEMMs** with
full K-varying A/B loads and checked outputs. They are not yet a general
GEMM interface or performance-tuned kernels; the diagnostic resident and
repeated-fragment paths are separately labeled and not counted as such.

## Reproduce

Latest single-CTA diagnostic (full protocol in [latency.md](latency.md)):

```sh
bash research/fp8-w8a16/build-latency-probe.sh
python3 research/fp8-w8a16/audit_latency_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/latency/latency-probe 4096 9 1729
```

Earlier CTA-grid experiment (full protocol in [grid.md](grid.md)):

```sh
bash research/fp8-w8a16/build-grid-probe.sh
python3 research/fp8-w8a16/audit_grid_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/grid/grid-probe 4096 9 1729
```

Earlier real-A paired experiment (full protocol in [paired.md](paired.md)):

```sh
bash research/fp8-w8a16/build-paired-probe.sh
python3 research/fp8-w8a16/audit_paired_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/paired/paired-probe 4096 9 1729
```

Earlier free-A queue-depth experiment (full protocol in [pipeline.md](pipeline.md)):

```sh
bash research/fp8-w8a16/build-pipeline-probe.sh
python3 research/fp8-w8a16/audit_pipeline_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/pipeline/pipeline-probe 4096 9 1729
```

Earlier immediate-wait baseline (full protocol in [loads.md](loads.md)):

```sh
bash research/fp8-w8a16/build-load-probe.sh
python3 research/fp8-w8a16/audit_load_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/load/load-probe 4096 9 1729
```

Register-only preheader/split experiment (details and full protocol in
[splitting.md](splitting.md)):

```sh
bash research/fp8-w8a16/build-split-probe.sh
python3 research/fp8-w8a16/audit_split_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/split/split-probe 4096 9 1729
```

Original bring-up/calculator tests follow. From the repository root, with the
existing system HIP compiler and user's calculator checkout/venv (no install):

```sh
python3 research/fp8-w8a16/check_calculator.py
bash research/fp8-w8a16/build.sh
python3 research/fp8-w8a16/audit.py temp/fp8-w8a16
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/probe 4096 9
# Or all steps:
bash research/fp8-w8a16/run.sh
# Short correctness/codegen smoke:
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/probe 64 1
```

`STEPS`, `ROUNDS`, and `OUT` configure `run.sh`. Steps must be a positive
multiple of eight. Defaults require about 1.22 GiB of weight-stream storage.
The standalone native O3 binary is under `temp/`, not either engine CMake tree;
this does not benchmark an unrecompiled engine binary. No attention context
or model is involved. Device is HIP logical 0; CPU and memory NUMA node 0.

## Calculator verification

`tools/external/amd_matrix_instruction_calculator/.venv/bin/python` runs the
local calculator successfully. Version **1.3.2**, upstream revision
`2ef91896bcdc4d26624f952e5c905c787cd9bc9e`.

`check_calculator.py` exercises the real CLI, checks gfx90a/MI250X aliases,
both instruction detail reports, and every A/B/D element mapping for the
32x32x8 instruction against independently written formulas. Its 1536 mapping
checks are also the basis for the C++ dense-output oracle. This is a focused
smoke/consistency test, not the calculator's full upstream test suite.

| FP16 MFMA | Execution cycles | Possible VALU coexecution cycles | A/B dwords per lane | FP32 accumulator elements per lane |
|---|---:|---:|---:|---:|
| 16x16x16 | 32 | 28 | 2 / 2 | 4 |
| 32x32x8 | 64 | 60 | 2 / 2 | 16 |

Reports and CSV mappings are retained in `results/`. The register mapping is:

- A: row = lane % 32; K = 4*(lane/32) + element_in_half4.
- B: column = lane % 32; K = 4*(lane/32) + element_in_half4.
- D: row = 8*(register/4) + 4*(lane/32) + register%4;
  column = lane % 32.

## Conversion and numerical contract

This prototype implements **E4M3FN, bias 7**, not E4M3FNUZ. The two NaN byte
encodings (0x7f/0xff) are excluded from the finite numerical contract; they are
not preserved as NaNs. Negative zero is preserved in the conversion test.
There are no quantization scales in this first experiment.

Four FP8 bytes in a dword become four FP16 values in two dwords:

1. Two `v_perm_b32` place pairs of bytes at halfword bits 15:8, zeroing the
   lower bytes. Permutation selectors are `0x01040004` and `0x03040204`,
   with src0=zero, src1=packed weights.
2. Two `v_pk_ashrrev_i16` shift each signed halfword right by one.
   **`op_sel_hi:[0,1]` is explicit**: both halves use shift amount one,
   not the high half (zero) of the immediate.
3. Two ANDs with `0xbfffbfff` clear bits 30 and 14. The result encodes w/256.
4. Two `v_pk_mul_f16` multiply by FP16 256 (`0x5c005c00` packed).

The build explicitly selects `-fdenormal-fp-math=ieee`; on-device tests,
not just that flag, establish subnormal survival in the packed multiply.
Every finite result is exact and every nonzero restored weight is FP16 normal.
MFMA still flushes FP16 **activation** subnormals; that is not changed here.
No x256 epilogue compensation is needed after the explicit multiplies.

Eight VALU instructions per four weights nominally fit within the fifteen
wave64 VALU slots of the 32x32x8 instruction. This issue-budget argument is
not by itself a proof of dependency-safe scheduling or actual overlap.

### Checks that run before and during timing

- 1024 mixed dwords: every byte encoding in every position, independently
  varying neighbors; compare all raw halfword bits, and all 4064 finite
  rescaled elements against independently decoded E4M3FN -> FP16 bits.
- Arbitrary signed dense 32x32x8 tile with all 254 finite encodings represented
  in B: all 1024 outputs versus a host FP64 dot-product oracle.
- All seven positive and seven negative subnormal magnitudes separately,
  in every operand half: restored path gives the exact expected MFMA result;
  unscaled w/256 control demonstrably gives zero.
- Every throughput output (all lanes/registers of every wave) against FP64.
  Streaming uses three distinct signed matrices, rotating with K step and
  wave, rather than one invariant input; references independently count the
  three phases. Gate is absolute error <= 0.001*(1+abs(reference)); observed
  error is **zero** in the retained 4096-step runs.

Additional short runs at 8/16/24/72 steps pass all checks, exercising every
three-matrix stream phase and different loop lengths. The existing Debug
engine suite also passes: 35 passed, one optional CanonicalEnv skip. It was
run in two parts (the first command timed out after tests 1..27 passed;
`ctest -I 28,36` completed the remaining tests). Those are unchanged engine
binaries, not FP8 integration tests. Logs are under `temp/fp8-w8a16/`.

These checks do not replace random/long-K numerical coverage, independent
per-wave random matrices, a real multi-tile GEMM, or canonical model tests.

## Scheduling lesson and four controls

The first repeated consumer test caught a failure that the one-MFMA dense
check did not: an immediate inline-asm multiply -> MFMA consumer sequence
produced wrong repeated results without separation. Adding conservative
`s_nop 3` separation made the checks pass. This is empirical evidence for a
scheduling hazard, not a minimum-spacing measurement. The follow-up
[NOP sweep](nop-sweep.md) calibrates each encoded wait-state unit at about
four core cycles here: `s_nop 3` is roughly 16 core cycles, not four.
Inline asm conceals instruction-specific dependencies from compiler hazard
handling; a correct isolated conversion is insufficient validation.

The current modes are:

1. **register-fp16:** invariant pre-expanded register weight operand.
2. **register-fp8-convert:** eight conversions and eight MFMAs per backedge,
   with two operand banks. Prepare the **next** operand, then issue MFMA on
   the **current** operand. The next conversion separates each packed
   multiply from its consuming MFMA. The initial current operand is primed
   from the pre-expanded seed outside the loop; there is still one conversion
   per MFMA, with the final prepared operand unused. Earlyclobber plus a
   compiler-only previous-operand input keep conversion destinations distinct;
   machine scheduling barriers preserve the intended order. There are **no
   explicit NOPs in this hot loop**.
3. **stream-fp16-batch8:** eight independent global FP16 weight loads before
   eight MFMAs; matching logical weights/activations.
4. **stream-fp8-batch8:** eight global FP8 dword loads, conversion and MFMA;
   currently retains the conservative immediate-consumer delay. This is a
   batch/drain control, **not a cross-batch rotating prefetch pipeline**.

Activations stay in registers in all modes. Each wave computes a separate
32x32 tile; there is no cross-wave weight sharing or LDS. Eight MFMAs per
backedge avoid conflating the separate lone-MFMA loop experiment with this
conversion study. The compiler-only batching constraints do not prove that
memory latency is hidden.

`audit.py` checks the **actual executable-extracted** gfx90a object: eight
MFMAs/backedge, exactly sixteen each of perm/shift/AND/multiply in conversion
loops, eight loads in streaming loops, and no hot-loop NOPs in register FP8.
ELF metadata verifies zero private bytes and zero LDS for all four kernels:

| Mode | Reported VGPRs | AGPRs | SGPRs |
|---|---:|---:|---:|
| register FP16 | 24 | 16 | 20 |
| register FP8 | 28 | 16 | 20 |
| stream FP16 | 40 | 16 | 18 |
| stream FP8 | 36 | 16 | 18 |

Raw register counts are metadata fields, not a measured occupancy claim.
ISA, metadata and extracted loops are under `temp/fp8-w8a16/`; the small audit
summary and source/executable/object hashes are retained in `results/`.

## First measurements

Native HIP O3/C++20, gfx90a, ROCm 7.1/system Clang 21.1.8. One MI250X GCD,
HIP 0, 104 CUs; 104 CTAs x 256 threads (416 waves, nominal four waves/CU,
not a measured assignment trace). CPU+memory NUMA node 0, **no model or
attention context**, no clock overrides. 4096 MFMAs per wave; nine rotating-
order rounds, eight event-timed launches per sample, one warm launch/mode.
No sustained thermal-conditioning phase; these are initial short-process
confirmations rather than a final power-controlled campaign.

FP8 streaming footprint is 416 MiB, FP16 832 MiB; both are allocated and
initialized before timing. Every wave has disjoint stream addresses. Packing,
reference checks, copies and allocation are outside GPU-event timing.

| Mode | Process 1 median TFLOP/s | Process 2 median TFLOP/s |
|---|---:|---:|
| register FP16 | 149.467 | 149.402 |
| register FP8 + conversion | 141.885 | 141.799 |
| stream FP16 batch8 | 35.758 | 35.811 |
| stream FP8 batch8 | 46.091 | 45.957 |

Register FP8 is **94.93% / 94.91%** of matched FP16 wall-time throughput.
Typical per-wave intervals are ~64.13 core cycles/MFMA for FP16 versus
~66.0-66.1 for FP8, with effective core/steady rates ~1463 versus ~1420 MHz.
Thus both per-cycle issue and operand-dependent effective clock contribute;
this is not evidence that conversion costs only clocks, nor a full-speed claim.
Per-wave timestamps surround the body, not a drained last-result timestamp;
4096 steps amortize the endpoints. The 64-step smoke is for correctness only,
not a reliable throughput/effective-clock measurement.

The conservative immediate-consumer register version (`results/initial.log`)
was ~138.438 TFLOP/s, ~68 cycles/MFMA. It is historical, not the current
source. The current streamed FP8 path still has this conservative schedule.
The streaming comparison changes payload, load instruction width and register
use together; do not interpret it as isolated conversion cost, physical HBM
traffic, or a tuned FP8-vs-FP16 GEMM comparison.

## Next experiments

1. **NOP-window diagnostic completed; arithmetic rate accepted by the user.**
   The [delay sweep](nop-sweep.md) confirms the predicted 1/U initial slope
   followed by a one-for-one exposed-delay slope, with a knee around 24–28
   added core cycles. This is scalar issue-gap tolerance, not guaranteed useful
   VALU/VMEM capacity. No further chase of the accepted ~95% rate is planned.
2. Give streaming FP8 the same two-bank schedule and a real rotating future-load
   window. Sweep window depth and verify waits/load ordering in actual ISA.
3. Add general multi-tile M/N/K GEMM validation, tails and adversarial repeated
   inputs before claiming a useful GEMM prototype or full finite-domain support
   beyond this conversion primitive.
4. Only then evaluate conversion-once/LDS reuse across waves, as a separate
   experiment with synchronization and traffic accounted for.
