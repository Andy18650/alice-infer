# Single-CTA latency-coverage diagnostic — 2x2 not pursued

The user approved the **single-CTA route only**. Every timed probe launches
**one CTA, 256 threads, four wave64s still along N**, with real shared A and
four private B column tiles. No 2x2 wave layout, new reuse strategy, LDS,
address/backedge tuning or production change is included.

**Finding:** the pure resident MFMA anchor reaches ~64 cycles/MFMA, but the
loaded FP8 body does not approach it merely by increasing queue depth.
At the primary 4096-step size, D8 is ~100 cycles with an in-kernel input
pre-read, ~117 after eviction. D16 is slightly worse, ~104/~120–121.
A smaller, cache-sized workload reaches ~96 warm at D8, but D16 still does
not close the gap. Thus **“above 64 implies depth 8 is insufficient” is not
supported as a complete explanation**. Latency coverage helps up to D8;
other CU-local delivery/issue/dependency/cache effects remain candidates.

## Shape, pipeline and depth

One CTA covers M=32,N=128, with K=8*steps. Primary K=32768; small K=8192;
larger K=65536. The four waves share A addresses and have distinct B columns.
Packed A/B layouts and the real paired load/convert body are retained from
the grid probe. Only this CTA runs during the clocked interval; the other
103 CUs have no probe CTAs. No CU/SIMD affinity or physical routing trace is
claimed, although a CTA necessarily resides on one CU.

D=1/2/4/8/**16**, all at **32 MFMAs per backedge**, avoiding dynamic ring
phases for D16. The raw pair ring primes D-1 A/B pairs, issues one new pair
per step, waits `vmcnt(2D-2)`, and prepares the oldest pair into two ready
operand banks. D16 therefore reaches 32 issued global-read credits and waits
at 30; there is no overflow/clipped-depth assumption. Final unused requests
are fully drained before endpoint register reuse.

Common padding is now 18 records per stream for all depths, safely covering
D16. It differs from the earlier ten-record pitch, so do not interpret this
as an instruction/address-identical comparison against earlier full-grid
numbers. The resident anchor uses the same CTA/unroll and repeats A0/B0
from registers through one dependent FP32 accumulator chain, with no timed
memory or conversion. Its host oracle is separate from the real K-varying
stream reference.

## Cache controls — why the warm protocol changed

An initial implementation warmed by launching the probe repeatedly before
the timed launch. It showed almost no warm/evicted separation, even at the
small size. That does **not** establish the cache state of a later dispatch:
cache retention, dispatch effects and placement were not isolated. Those
preliminary files are kept under `temp/`, not used in the primary results;
no specific runtime invalidation mechanism is asserted.

The final two protocols use the **same compiled entry**, selected by a runtime
flag, and exactly one timed probe dispatch per sample:

- **Pre-read / warm:** the single CTA reads its full A/B input footprint
  **inside the same dispatch**, before the core timestamps, then initializes
  operands/accumulator and executes the measured body. There is no dispatch
  boundary between pre-read and measurement. A checksum dependency and
  volatile wide reads retain the pre-read; code-object inspection verifies
  its backward loop precedes the first core timestamp.
- **Eviction-controlled:** a disjoint **256 MiB** reader over 832 CTAs is
  completed and synchronized **before launching** the probe. The probe has no
  internal pre-read. This is a conventional cache-eviction control, not a
  proven architectural cache flush or an isolated physical-DRAM latency test.
  Initial operand/queue priming necessarily warms a few leading records.

The eviction reader explicitly loads every 16-byte record and consumes all
four dwords. Three grid-stride checksum controls (first/middle/last thread)
pass in every process. The reader is outside the measured launch and core
interval; it may still influence prior power/fabric/translation state. Its
broad grid is setup, not a second timed tiling route.

**Use body clocks, not event throughput, for the cache comparison.** GPU-event
`launch_ms` includes the warm pre-read, whereas `cycles_per_mfma` and
`body_us_mean` start afterward. There is intentionally no misleading warm
TFLOP/s column computed from those unequal amounts of launch work.

Runtime reports **8 MiB L2**. Active input footprints, including padding:

| Steps | A + FP8 B | A + FP16 B |
|---:|---:|---:|
| 1024 | 1.5264 MiB | 2.5439 MiB |
| 4096 | 6.0264 MiB | 10.0439 MiB |
| 8192 | 12.0264 MiB | 20.0439 MiB |

The small case fits the runtime-reported aggregate L2 for both formats. The
primary FP16 and larger cases do not. The user subsequently recalls a
CDNA2 allocation-locality effect: one shader engine may allocate into only
a preferred ~2 MiB quarter despite being able to read the full L2. This round
did not verify that hardware policy. Under that model the small FP8 set
(1.53 MiB) fits but the small FP16 set (2.54 MiB) does not. Therefore the
cross-format warm payload-rate observation below is NOT a capacity-matched
proof of a common CU-local throughput limit. The FP8 residual remains an
observation below that nominal 2 MiB budget. Even a below-capacity footprint
does not guarantee all hits: associativity, distribution and traversal matter.
“Warm” describes the pre-read protocol, not a measured hit rate.

## Primary results: one CTA, 4096 steps

Native HIP O3/C++20, gfx90a, system ROCm 7.1 / Clang 21.1.8. HIP device 0,
CPU+memory NUMA node 0, **no model or attention context**, no clock/power
settings changed. Two independent nine-round processes (seeds 1729/2718),
randomized kernel/cache order. One timed dispatch/sample, four per-wave
clock/steady intervals; table reports median mean-wave core cycles/MFMA.

| FP8 depth | Pre-read, run 1 / run 2 | Eviction-controlled, run 1 / run 2 |
|---:|---:|---:|
| 1 | 518.689 / 517.007 | 672.894 / 673.966 |
| 2 | 274.900 / 273.217 | 366.801 / 370.623 |
| 4 | 155.377 / 152.981 | 193.229 / 194.782 |
| 8 | **99.903 / 100.533** | **117.504 / 117.222** |
| 16 | 104.291 / 104.597 | 121.061 / 120.196 |
| Resident FP16 MFMA anchor | **64.024 / 64.048** | **64.080 / 64.070** |

Effective clocks are ~1683–1685 MHz in these cases. This differs from the
full-grid ~1400–1500 MHz operating points; equal core cycles across those
experiments do not imply equal nanosecond intervals. The tests are lightly
loaded references under these addresses/cache/power states, **not a universal
minimum memory latency** or a direct measurement of NoC contention.

FP16 streamed B, D8: warm 178.087/174.271 versus evicted 170.909/175.718.
D16 is 174.640/178.477 versus 176.909/173.444, again no systematic recovery
to 64. Its ~10 MiB working set exceeds reported L2, so the cache-small
control is more useful for its warm-path interpretation.

## Cache-small and larger controls

Three rounds at 1024 steps (seed 31415), median cycles/MFMA:

| Format / depth | Pre-read | Eviction-controlled |
|---|---:|---:|
| FP8 D1 | 345.915 | 684.500 |
| FP8 D4 | 113.517 | 197.917 |
| FP8 D8 | **96.442** | 122.039 |
| FP8 D16 | 114.118 | 118.017 |
| FP16 D1 | 327.655 | 669.205 |
| FP16 D8 | **134.064** | 176.167 |
| FP16 D16 | 133.463 | 174.979 |

The clear small-case separation validates the usefulness of the in-dispatch
pre-read control, without claiming exact cache levels or hit counts. The
remaining warm gap above the resident anchor is real in this implementation;
more buffering is not a monotonic cure.

For context only, four waves request 3072 useful A+B bytes per logical K step
with FP8 B, 4096 with FP16 B. At small-case warm D8, these correspond to
**31.85 / 30.55 logical request bytes per core cycle per CTA**. The similar
payload rates are consistent with a CU-local delivery/throughput limit, but
are **not physical L1/L2 bandwidth counters or proof of a particular unit**.
Instruction issue, dependencies, register-bank placement and cache behavior
have not been separated. D16 also changes register allocation/cache working
window despite identical steady-state opcode counts; no spilling is present.

At 8192 steps (three rounds, seed 16180), FP8 D8 is 120.833 warm / 117.983
evicted and D16 124.085 / 119.801. The larger footprint exceeds reported L2;
a pre-read is not expected to retain the complete stream. Resident remains
~64.01/~64.02. All raw per-wave minima/maxima and sample ranges are retained.

## Correctness / actual-code-object checks

All **528 timed samples** and **352 independent A-only/B-only/both-change/
restore, warm/cold-path, poisoned-output/guard cases** pass exactly against
FP64. The bounded dyadic data/exact FP32 bound is unchanged. Short 32/96-step
controls pass as well. Eviction checksums validate the reader separately.

`audit_latency_probe.py` checks ten streaming entries, the resident anchor
and eviction reader in the executable-extracted object:

- D1..16 pair issue order and exact waits, with no accidental inner full drain.
- Three-lap pending-register/FIFO safety, including 16 rejected injected early
  reads, and explicit endpoint drain.
- In-dispatch warm pre-read loops before clocks; resident body has **only 32
  MFMA vector instructions/backedge** and no timed loads/conversion.
- Explicit eviction dwordx4 loads/checksummed consumption.
- Zero private/scratch/LDS bytes and no lane spills. At FP8 D8/D16 the metadata
  is 56/80 VGPRs, 16 AGPRs, 68 SGPRs; FP16 is 68/100 VGPRs, 16 AGPRs,
  53 SGPRs. Higher register counts are not declared harmless: placement can
  affect issue behavior even without spilling.

Core endpoints do not explicitly fence completion of the final MFMA, as in
prior probes; long loops amortize that small endpoint effect. A launch uses
one CTA, but no specific CU/SIMD assignment is enforced or traced.

## Conclusion and stopping point

Depth helps substantially through D8, and cold input adds measurable cost.
But **the entire residual >64 interval cannot simply be assigned to missing
pipeline depth**: D16 does not recover 64, and a cache-small pre-read case
still has a ~96-cycle FP8 floor. This narrows the next question toward local
memory delivery/issue/dependency/cache effects; it does not uniquely identify
one or rule out a latency component masked by other limits.

Only the selected single-CTA route was pursued. **No 2x2 wave layout** or
further corrective optimization is added. Previous research kernels and
production code remain unchanged. This diagnostic round is concluded.

This real-K state is checkpointed at `21c735b`. The subsequent user-requested
[fixed-K-address diagnostic](fixed.md) deliberately repeats fragment zero
with all load/conversion instructions retained. Both formats then fit in a
few KiB; this preserved real-K source/data are not overwritten.

## Reproduce

```sh
bash research/fp8-w8a16/build-latency-probe.sh
python3 research/fp8-w8a16/audit_latency_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/latency/latency-probe 4096 9 1729
# Repeat: 4096 9 2718. Small: 1024 3 31415. Larger: 8192 3 16180.
# Optional fourth argument: eviction MiB, default 256 (32..1024 accepted).
python3 research/fp8-w8a16/analyze_latency_probe.py
```

Raw runs/logs, flags, source/executable/object hashes, audit and summary are
under `results/latency/`; full ISA, loops, metadata, binaries and preliminary
dispatch-warm observations are under `temp/fp8-w8a16/latency/`. No engine/CMake
changes or engine-regression rerun are claimed for this standalone experiment.
