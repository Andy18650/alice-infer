# Disjoint-source CTA-count latency sweep

User-directed extension of the one-load/VM0 experiment. Preserve both earlier
binaries/source. The same generated64-sample assembly is byte-identical in
source to the single-CTA probe and passes the same native instruction audit.

## Setup

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, block256,
no model/context/clock-policy overrides. Device reports **104 CUs,8 MiB L2**.
CTAs=1/8/16/32/64/128/256; both shared16 KiB and private16 KiB/wave LDS
layouts,64 KiB allocated per CTA. One load outstanding per wave, four waves/
CTA; intervals include the original clock code, no overhead subtraction.

CTA c accesses only `[c*1MiB,(c+1)*1MiB)`, with the original four256 KiB
wave ranges and16 KiB/wave slice per batch. Source offset advances between
16 positions, then repeats. **64 batches** give four full source traversals:
4096 measurements/wave,16384/CTA/process. Two counterbalanced repetitions
(counts ascending/shared-first, then descending/private-first).

| CTAs | Total distinct source | Source bytes touched/launch |
|---:|---:|---:|
| 1 | 1 MiB | 64 KiB |
| 8 | 8 MiB | 512 KiB |
| 16 | 16 MiB | 1 MiB |
| 32 | 32 MiB | 2 MiB |
| 64 | 64 MiB | 4 MiB |
| 128 | 128 MiB | 8 MiB |
| 256 | 256 MiB | 16 MiB |

Thus16+ CTAs exceed L2 in total source footprint;256 exceeds it in source
bytes touched per launch too. Each launch is one source pass over its slice,
not repeated reads of one address hidden inside a huge unused allocation.
These are footprints, not counter-certified miss rates or cache partitions.

64 KiB/CTA limits concurrent residency; CTAs above104 cannot all be resident
simultaneously. They execute in scheduling waves. We do not assume a precise
CTA-to-CU mapping or a globally synchronized sample phase.

## Keep instrumentation from becoming a large competing stream

Result clocks stay in64 VGPRs/wave until the post-sampling workgroup barrier,
then1 KiB of cycle records/CTA is written. Final LDS contents are checked
locally after sampling, returning256 error counts (1 KiB)/CTA rather than
copying64 KiB/CTA to global memory. This checks all active and inactive LDS
words while avoiding a64 KiB global checker-write stream competing with other
CTAs still sampling. Record and status guards are checked on CPU.

Unlike the one-CTA case, other CTAs can be in epilogues while a CTA samples;
its small record/status traffic is not eliminated or presented as nonexistent.
The measured load/wait/clock/register-capture instructions are unchanged; no
barriers or result stores are inserted within samples. No spills/lane copies.
Only FINAL batch LDS status is returned, as in the preceding probe.

## Main matched64-batch result

Medians of individual core-cycle intervals, run1/run2. Every observation is
one wave's256 B coalesced load plus VM0 and timing overhead, not a CTA-span
or throughput period. All observations retained.

| CTAs | Shared median | Private median | Private mean | Private P95 |
|---:|---:|---:|---:|---:|
| 1 | 292 /296 | 296 /296 | 363.54 /364.77 | 612 /608 |
| 8 | 580 /580 | 580 /580 | 568.31 /564.92 | 752 /748 |
| 16 | 584 /584 | 584 /584 | 611.13 /612.98 | 820 /828 |
| 32 | 592 /592 | 592 /592 | 620.26 /620.10 | 836 /836 |
| 64 | 684 /684 | 684 /684 | 714.99 /712.51 | 976 /972 |
| 128 | 972 /972 | 972 /972 | 949.04 /946.87 | 1364 /1352 |
| 256 | 952 /952 | 952 /952 | 950.55 /951.45 | 1388 /1388 |

Both layouts remain similar. There is a large shift between the1 MiB and
8 MiB nominal source cases, then further latency growth with device load.
8 MiB source already consumes the entire reported L2 before counting records,
status, arguments/other accesses. Not all latency growth is established as
cache-capacity effect: memory-system load, wave/CTA scheduling and phase skew
also change. At128/256 CTAs residency saturates and the median is not strictly
monotonic. No exact HBM latency, bottleneck queue/count or bandwidth ceiling
is inferred from this instrumented per-wave latency test.

**33,095,680 individual intervals**, all final per-CTA LDS checks and all
record/status guards pass. Raw compressed CSV keeps mode,grid size,batch,CTA,
source offset,wave,sample,cycles. `summary.csv` includes min/max/median/mean,
population SD, range, P05/P25/P75/P95/P99 and IQR; `per-batch.csv` preserves
the source-pass evolution. No sample filtering or clock-overhead correction.

## Important protocol correction: first traversal versus repeated source

The initial pilot used16 batches (one source traversal), reducing data volume.
Its1-CTA median was~590 rather than the earlier296. We disclosed the discrepancy
and replayed the **preserved old single-CTA binary** under both protocols:

-16 batches:4096 intervals, median588, mean591.13 cycles.
-64 batches:16384 intervals, median292, mean359.01 cycles.
-In the64-batch replay, batches0–15 stayed around580–590; batches16 onward
 dropped to~276–288. Addresses are first revisited at batch16.

The main sweep was therefore rerun with the **original64-batch protocol**,
not compared to a relabeled16-batch baseline. The16-batch pilot remains in
`results/grid-latency/` (8,273,920 intervals), main data in
`results/grid-latency64/`, original-binary replay CSV/logs/commands/fingerprints
in the latter's `controls/`. No measured artifact was overwritten.

Main-grid per-batch results reproduce this: private1-CTA first-pass batch
medians572–604, later passes268–304. At8+ CTAs the large first-pass/revisit
median collapse is absent. This is consistent with source cache warming and
working-set capacity, but does not uniquely identify a prefetcher or prove
an exact cache hit/miss split.

Pretraversal and selected-kernel200 ms warmup occur BEFORE H2D initialization
of the record buffer, matching the old protocol. Thus they did **not guarantee
uniformly hot source accesses during capture**. We have not separately isolated
which intervening operation/dispatch/coherency state is responsible. The old
296-cycle aggregate is a mixture of slower first traversal and faster revisits,
not a single uniform hot-latency constant. First-batch extrema are a narrower
observation than this whole-first-source-pass effect.

## Replay

```sh
bash research/lds-staging/build-grid-latency.sh       # optional fresh build path
numactl --cpunodebind=0 --membind=0 temp/lds-staging/grid-latency/probe \
  --ctas 256 --mode private --batches 64 --warmup-ms 200
python3 research/lds-staging/run_grid_latency.py --output <fresh-results-directory>
```

For main results use **explicit --batches64** (the CLI's pilot default is16;
the main runner explicitly selects64). `build-grid-latency.sh` reuses the
unchanged sample generator/native audit; its own build path preserves old
executables. Source/native manifests, raw clocks, metadata, commands and
per-batch distributions accompany both datasets. No production/CMake/model,
package/system/clock-policy or unrelated research changes.
