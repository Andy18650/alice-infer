# One direct-to-LDS load / VM0: individual latency intervals

Follow-up [grid sweep](GRID_LATENCY.md) replays this frozen binary and shows
that the original64-batch aggregate mixes slower first source traversal
(batches0–15,~580–590 cycles) with faster revisits (~276–288). The prior
pretraversal/warmup does not guarantee uniformly hot capture after record
initialization. Original observations and raw data below remain unchanged.

User-directed second test, separate source/binary from the no-periodic-wait
throughput baseline. Preserve that baseline unchanged. Capture many individual
core-cycle deltas in registers; do not turn this into a whole-loop average.

## Exact timed interval

Each of64 unrolled samples/wave uses:

```asm
s_memtime s[16:17]                    # begin core timestamp
s_waitcnt lgkmcnt(0)                  # begin-clock read completion
  global_load_dword v[8:9], off ... lds
  s_waitcnt vmcnt(0)                  # exactly one load to retire per wave
s_memtime s[18:19]                    # end core timestamp
s_waitcnt lgkmcnt(0)                  # end-clock result availability
s_sub_u32 delta, s18, s16             # low32 difference, modular arithmetic
v_mov_b32 sample_register, delta      # a distinct VGPR for each sample
```

**Timing code is included; no overhead calibration/subtraction.** Delta includes
begin-clock completion wait, load issue/retirement and end-clock sampling.
The end-clock completion wait and result capture happen after the end timestamp.
This is slightly above the load's actual latency, not an isolated instruction
latency or hardware queue-occupancy measurement. Report core cycles, not ns
converted using an assumed GPU frequency. Intervals are far below the2^32
wrap period; modulo-low32 subtraction is valid. No samples are filtered.

M0 setup/source address updates are before each timed interval. Every16
samples they advance4 KiB; immediates0/256/...3840 shift both source/LDS
addresses. There are64 distinct VGPR result registers live until the batch
ends. Native inspection checks exact load/wait/clock/capture adjacency, no
spills, stores, LDS consumers or barriers between sample endpoints.

A workgroup barrier AFTER all64 samples prevents any wave from issuing result
or LDS-check stores while another wave still samples. A batch is one kernel:
all result stores complete before the next batch on the same stream. Thus
previous result stores cannot be included in the next measured VM0 wait.
The four waves independently execute one outstanding load each; this is NOT
an otherwise idle single-wave latency measurement.

## Source and checks

Same conditions as baseline: native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0,
CPU+memory NUMA0, one CTA/256 threads,1 MiB source,64 KiB allocated LDS in
both modes. No model/context/clock-policy changes. Source consists of four
256 KiB wave ranges. Shared destination16 KiB versus private16 KiB/wave.
Every load fully coalesces64 lanes x4 B =256 B/wave.

An unmeasured volatile source pretraversal and200 ms selected-kernel warmup
precede capture. Each batch traverses16 KiB/wave; source offset cycles through
16 positions to cover the1 MiB allocation. Default64 batches give four full
traversals and **4096 samples/wave,16384 per process**. The global source is
intended L2-hot; no cache counters certify residency.

Source bits repeat by lane, so overlapping/reordered LDS writes agree. Final
full LDS contents, inactive shared-mode sentinel area, record/check guards,
all expected sample keys and positive non-sentinel cycle values are checked.
LDS contents are checked after the LAST measured batch, not all preceding
batches; all individual cycle records are retained/guard-checked. The periodic
payload is not a source-stride/last-writer oracle (same caveat as baseline).

## Unfiltered distributions

Two counterbalanced process repetitions: shared/private, then private/shared.
Each row contains16384 individual observations, not medians of event times.
Core cycles, timing overhead included:

| Destination | Run | Min | Max | Median | Mean | Population SD | P95 | P99 | IQR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Shared16 KiB | 1 | 232 | 1456 | 296 | 362.34 | 134.59 | 604 | 624 | 104 |
| Shared16 KiB | 2 | 236 | 2192 | 296 | 362.71 | 136.54 | 604 | 624 | 109 |
| Private4x16 KiB | 1 | 236 | 1704 | 296 | 365.39 | 137.44 | 612 | 632 | 102 |
| Private4x16 KiB | 2 | 236 | 1800 | 296 | 361.80 | 137.63 | 612 | 628 | 106 |

All four medians are296 cycles; means361.8–365.4 and P95~604–612 show a broad
right tail. P05 is256–260 cycles. Min/max ranges are1224/1956 shared and
1468/1564 private. There is no systematic shared/private central-latency gap.
The four maxima occur in batch0 (sample16–19); they remain in the statistics.
We do not attribute these extrema uniquely to cache miss, clocks, instruction
fetch, launch transition or another hardware effect. Per-wave and sample-
position summaries are supplied to inspect this without another experiment.

**65536 measured intervals +512 smoke intervals** captured; final LDS and all
record/check guards pass. Percentiles use inclusive linear interpolation,
SD is population SD, IQR=P75-P25, range=max-min. The per-wave/sample observations
are correlated; counts are not a claim of statistically independent draws.

The prior streaming17.4-cycle throughput period is not a single-load latency.
This serialized timing probe measures hundreds of cycles while the streaming
probe overlaps requests. Their ratio is not an exact outstanding queue depth:
this timing includes instrumentation, and the two schedules/register footprints
are different. No empty-clock control or extra tuning is added this round.

## Files / replay

Raw per-interval CSV retains mode,batch,source offset,wave,sample index,cycles.
`results/latency/summary.csv` gives complete stats; `per-wave.csv` and
`per-position.csv` give stratified stats. Source snapshots, commands, source/
native SHA256, actual timed ISA and metadata accompany them. Measured binary
and full native object/ISA: `temp/lds-staging/latency/`.

```sh
bash research/lds-staging/build-latency.sh              # optional fresh build path
python3 research/lds-staging/inspect_latency.py temp/lds-staging/latency
numactl --cpunodebind=0 --membind=0 temp/lds-staging/latency/probe \
  --mode private --batches 64 --warmup-ms 200
python3 research/lds-staging/run_latency.py --output <fresh-results-directory>
```

This changes no no-wait baseline source/binary, GEMM, production/CMake/model,
system packages or clock policy. Only explicit one-load VM0 retirement is
introduced in this new latency probe, as requested.
