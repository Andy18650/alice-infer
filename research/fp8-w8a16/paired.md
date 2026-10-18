# Real shared A: paired A/B load pipelines

User-approved next step: remove the free-A assumption, keep A **MFMA-packed**,
and issue ordered A/B load pairs with **`vmcnt(2D-2)`** before consuming the
oldest pair. The prior register-only and B-only kernels remain unchanged.

**Result:** depth 8 largely hides the added A-loading latency in this specific
shared-A workload. At U=8, real-A FP8 reaches **80.793 / 80.812 TFLOP/s** versus
co-run free-A **84.865 / 85.411**. Core intervals remain about **117 cycles/MFMA**
for both; real A runs at a lower effective clock (~1421 versus ~1475 MHz).
Event throughput retains **95.20% / 94.62%** of the free-A control. This is not
a proof that all remaining wall-time difference is clock-only, nor a claim of
compute-bound GEMM or hidden activation latency for arbitrary matrix shapes.

## Logical shape and actual VRAM data

For the primary 4096-step run:

```text
A: 32 x 32768, FP16, one shared allocation
B: 32768 x 13312, FP8 or matching FP16 control
C: 32 x 13312, FP32
```

Each wave still owns a 32x32 output tile and traverses full K; there are 416
waves (104 CTAs x 256 threads). Each CTA's four waves cover 32x128 logical
output elements. No split-K, new M/N tiling, explicit cross-wave sharing or
LDS is added. At 8192 steps, K doubles to 65536.

A is now physically populated across all K tiles, not one 32x8 fragment
reused at every step. Its storage is:

```text
[K-tile][K-half:2][M-row:32][K-within-half:4] of FP16
```

Per-lane record = four FP16 values / two dwords. An entire A tile is 512 B,
coalesced across wave64 and already in MFMA operand order. Step s accesses
`activation[s*64 + lane]`: **there is no wave/block term in that address**.
Thus every wave needs the same A tile at a given logical step, although they
need not reach it simultaneously. Four waves/CTA request it independently;
there is no software broadcast or cooperative block barrier.

B keeps the prior private per-wave stream and MFMA-packed layout. All depth
and free/real-A cases use the same B allocations and `steps+10` row pitch.

| Steps | Logical A | Padded shared A | FP8 B allocation | FP16 B allocation |
|---:|---:|---:|---:|---:|
| 4096 | 2 MiB | 2.0048828125 MiB | 417.015625 MiB | 834.03125 MiB |
| 8192 | 4 MiB | 4.0048828125 MiB | 833.015625 MiB | 1666.03125 MiB |

The contents are synthetic and K-varying: A uses 13 distinct signed fragments
with phase `(5*s + floor(s/13) + 7*A_epoch) % 13`; B retains its independent
17-fragment phase `(s + floor(s/17) + 7*wave + 3*B_epoch) % 17`. Full arrays
are materialized before timing; kernels load the actual K-indexed addresses,
not the small pattern tables. A's phase period is 169 tiles, B's is 289, and
their joint pattern is much longer than a single repeated MFMA fragment.
The A0 baseline preserves the earlier free-A values, while streamed A varies
at each logical step. Comparison includes this intended change in activation
values as well as their loads; it is not an isolated fixed-operand latency
measurement.

## Paired queue / readiness

D is the number of **pairs** at consumption, not the number of total load
instructions. Two ready A operands and two ready B operands are primed for
steps 0/1, accumulator initialization and initial loads are completed, then
D-1 raw pairs for steps 2..D are issued before the core timestamp.

```text
step s:
    MFMA(ready_A[s%2], ready_B[s%2])
    issue A(s+D+1)                 // global_load_dwordx2
    issue B(s+D+1)                 // dword FP8, dwordx2 FP16
    s_waitcnt vmcnt(2D-2)
    copy oldest A into its freed ready bank
    convert/copy oldest B into its freed ready bank
```

A-first/B-second issue is a literal two-instruction assembly block. At the
wait, up to 2D global-read credits have been issued; retiring the oldest two
makes the matching A/B pair ready. The next pair is left outstanding:

| D | Primed global loads | Steady wait | Remaining credits |
|---:|---:|---|---:|
| 1 | 0 | `vmcnt(0)` | 0 |
| 2 | 2 | `vmcnt(2)` | 2 |
| 4 | 6 | `vmcnt(6)` | 6 |
| 8 | 14 | `vmcnt(14)` | 14 |

The steady-state instructions are the gfx90a global-read class under the
user's in-order-return assumption; these experiments verify their observed
behavior with both widths. Initial mixed/seed reads are fully drained first.
This is not a portable guarantee for arbitrary memory classes or devices.

Wait and **first reads of both raw results** are fused in one assembly block,
continuing the previous asynchronous-register safety rule. For FP8 the block
has two A dword copies and eight B-conversion VALUs. The FP16 B control uses
four dword copies. Other emitted address work is also recorded: U=8 real-A
FP8 includes an additional address-related MOV and scalar address arithmetic,
not just those two data copies. No total-instruction-cost claim hides that
work.

Queues continue across backedges without batch drains. A final full drain
keeps trailing unused results live until complete, before output/timestamp
register reuse. The core-clock interval includes that drain; event time also
includes priming/initialization. No backedge split or other tuning is added.

Co-run free-A controls perform the same B streaming but load only A0 in the
prologue and retain it in registers, using `vmcnt(D-1)` for the B queue. They
are the matched comparison, not the earlier executable's absolute TFLOP/s.

## Correctness and code-object evidence

The reference independently maps every lane/register to the 32x32 output
coordinate. CPU FP64 dot products are precomputed for each A/B fragment pair;
for each wave, joint phase counts reconstruct the full K reduction. Free-A
uses A0 and separate B counts. Bounded dyadic values retain the previous
integer-numerator bound below 2^24 at 4096/8192 steps, so **any nonzero output
error is rejected**, not hidden by a growing tolerance.

Every process tests four independent fixtures without recompilation:

1. A changed, B unchanged.
2. B changed, A unchanged.
3. Both changed.
4. Baseline restored.

All outputs are poisoned before each fixture/case, and both output guards are
checked. All **384** independent A/B-change/poison/guard cases across three
processes pass, as do all **672 timed samples**, with zero observed absolute
error. Short 32/96-step controls additionally cover priming/wraparound/drain;
they are correctness checks, not steady-state measurements.

`audit_paired_probe.py` inspects all **32 entries** in the actual executable:

- Exactly U A-first/B-second load pairs and U `vmcnt(2D-2)` waits in real-A
  loops; free-A has U B loads and U `vmcnt(D-1)` waits.
- The 16-credit D8 queue is not silently drained to zero inside the loop.
- Pessimistic FIFO simulation over three cyclic bodies forbids reads or
  writes of unresolved load destinations, including compiler-inserted copies.
  It verifies initial 2(D-1), peak 2D and surviving 2(D-1) credits; free-A uses
  the corresponding one-stream counts. Thirty-six deliberate early-read
  negative controls, covering both streams, are rejected.
- Exact FP8 conversion counts and final full drain before end timestamps.
- Zero private/scratch bytes, zero LDS, no SGPR lane-spill transfers.

Real-A U=8 resource metadata (16 AGPRs in all cases):

| B format | D | VGPRs | SGPRs |
|---|---:|---:|---:|
| FP8 | 1 | 36 | 27 |
| FP8 | 2 | 40 | 27 |
| FP8 | 4 | 44 | 27 |
| FP8 | 8 | 56 | 27 |
| FP16 | 1 | 36 | 30 |
| FP16 | 2 | 40 | 30 |
| FP16 | 4 | 48 | 30 |
| FP16 | 8 | 64 | 30 |

U=32 also remains spill/LDS-free; its larger SGPR/VGPR counts are retained in
the audit rather than optimized away. Metadata is not an occupancy trace.

## Measurements

Native HIP O3/C++20, gfx90a, system ROCm 7.1 / Clang 21.1.8. HIP logical
device 0, 104-CU MI250X GCD, CPU+memory NUMA node 0, **no model or attention
context**, no clock/power overrides. 104 CTAs x 256 threads = 416 waves
(nominal four waves/CU, not a measured placement trace).

Warm all entries and condition U=8 cases for >=2 seconds. Two independent
nine-round processes at 4096 MFMAs/wave (seeds 1729/2718), plus three rounds
at 8192 (31415). Randomized case order, eight event-timed launches/sample;
per-wave core/steady intervals from the final launch. Runs are sequential on
the GPU. No output/reference work is included in event timing. The endpoint
is a load-queue drain, not an explicit final-MFMA completion fence.

### FP8 B, U=8, K=32768

| D | Free-A TFLOP/s, run 1 / run 2 | Real-A TFLOP/s, run 1 / run 2 | Real-A cycles/MFMA, run 1 / run 2 |
|---:|---:|---:|---:|
| 1 | 18.906 / 18.865 | 14.647 / 14.672 | 764.266 / 763.333 |
| 2 | 34.944 / 34.914 | 29.306 / 29.336 | 368.970 / 369.552 |
| 4 | 62.679 / 62.693 | 53.710 / 53.753 | 189.216 / 189.313 |
| 8 | 84.865 / 85.411 | **80.793 / 80.812** | **117.110 / 116.775** |

Depth 8 retains 95.20% / 94.62% of co-run free-A event throughput. Free-A
core intervals at D8 are 117.893 / 116.332 cycles/MFMA, very close to real A.
Effective core rates are ~1422/~1421 MHz for real A versus ~1476/~1474 MHz
for free A, consistent with the chip's accepted response to extra work and
changed input activity. That accompanies much of the event-rate difference;
we do not assign every remaining percentage point to clocks or claim an
isolated A-load latency from the subtraction.

### Request bandwidth is not HBM bandwidth

At D8, real-A FP8 has:

| Payload | Run 1 | Run 2 |
|---|---:|---:|
| B useful GB/s | 1262.396 | 1262.688 |
| A wave-request GB/s | 2524.792 | 2525.376 |

The A field counts **all 416 waves' reads to shared addresses**, not unique
matrix bytes or HBM transfers. One logical A pass is only 2 MiB, versus
832 MiB of aggregate useful A requests over the grid. Both cache reuse and
concurrent miss merging can reduce external traffic; no hardware counters
were collected to measure hit rates, physical HBM bytes or actual broadcast.
Do **not** describe the combined ~3.79 TB/s shader payload as HBM bandwidth.

Requested (including priming/lookahead) A/B byte counts use
`waves * (steps+D+1)` fragments. Useful counts use `waves * steps`; each A
fragment is 512 B, B is 256 B FP8 or 512 B FP16. Free-A's steady A-request
field is zero by convention, not a claim that it has no prologue A load.

### Larger A matrix and other controls

At K=65536 (4 MiB logical A), real-A FP8 D8/U8 reaches **81.541 TFLOP/s**,
117.630 cycles/MFMA, versus free-A 84.391 TFLOP/s and 118.598 cycles/MFMA.
This control does not show an additional large per-MFMA stall. It does not
prove cache behavior for larger M or arbitrary activation working sets.

Real-A FP8 U=32/D8 gives ~81.382/~81.240 TFLOP/s at K=32768. These are
diagnostic unroll comparisons, not a tuned dispatch. Real-A FP16 B at U=8/D8
is ~41.399/~41.431 TFLOP/s, 1293.734/1294.732 useful B GB/s, versus free-A
44.201/44.256 TFLOP/s and 1381.280/1382.988 useful B GB/s.

## Scope / next decision

The requested real shared-A implementation works, with matching A/B steps
and the approved paired counter arithmetic. Depth 8 substantially hides the
added A-loading stall at this shape; shallower queues show a clear cost.
The remaining event-rate difference is recorded, not tuned away. No new
M/N tiling, explicit cache/broadcast strategy, LDS sharing, deeper queue,
backedge redistribution, production integration or compute-bound claim is
introduced. Previous sources/binaries and both research checkpoints remain
unchanged. The next algorithmic step remains the user's decision.

The following user-directed [13x8 CTA-grid test](grid.md) distributes eight
A row streams and shares B across those M groups, still with all four CTA
waves along N. It is a separate equal-work shape/mapping experiment; the
paired source and measurements above remain unchanged.

## Reproduce

```sh
bash research/fp8-w8a16/build-paired-probe.sh
python3 research/fp8-w8a16/audit_paired_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/paired/paired-probe 4096 9 1729
# Repeat: 4096 9 2718. Larger shared A: 8192 3 31415.
python3 research/fp8-w8a16/analyze_paired_probe.py
```

Raw runs/logs, flags, source/executable/object hashes, audit, medians and
comparisons are retained in `results/paired/`; full ISA, loops, metadata and
binaries are in `temp/fp8-w8a16/paired/`. The analyzer verifies completeness,
exact correctness, independent fixtures and all request-byte accounting.
Engine/CMake files are unchanged; engine regressions were not rerun for this
isolated research extension.
