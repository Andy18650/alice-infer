# Coarse CTA-count knees: DWORDx4 VGPR ring

User-directed quick survey, not a variance-controlled optimum search. Keep
one/two/four waves per CTA and the same32-slot x4-VGPR ring; sweep CTAs
1/2/4/8/16/32/64/104/208. Stop after identifying broad knees/regressions.
No new wait policy, ring-size, width or address-layout optimization.

## What is held fixed

Identical generated volatile-asm header to the preceding one-CTA study:
32 DWORDx4 loads,16 address ops,5 control ops per53-instruction loop,128
payload VGPRs/lane, zero loop waits/consumers/copies/barriers. Only terminal
VM0 outside issue interval. Native metadata136 VGPRs,28 SGPRs, zero LDS/
scratch/lane spills. Source adds blockIdx.x x1 MiB so CTAs do not alias.

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, block64/128/256,
no model/context/clock overrides. Reported device104 CUs/8 MiB L2.
Fixed1 MiB source/CTA means total1–208 MiB: CTA count and cache footprint
change together. The early curve bend cannot be uniquely assigned to L2
capacity, and requested bandwidth is not measured physical HBM/L2 traffic.

To keep this survey proportional, use **three checked samples/process**, two
counterbalanced ascending/reversed process repetitions, at least100 ms
selected-kernel warmup. Each sample requests **256 MiB/CTA** (256 source
traversals), not the prior detailed study's1 GiB/CTA. Default groups8192/waves
keep bytes/CTA fixed across wave counts. Initial full-source traversal and
subsequent H2D output initialization follow the preceding protocol; initial
cache transients are included in the long stream, no counter-based hot claim.

Each CTA records raw core+steady clocks for all its active waves. Core spans
are calculated per CTA only; aggregate requested GB/s uses the **global
steady-clock makespan** max(issue-end)-min(begin) over the grid. Do not combine
core clocks from different CUs. Completion and full HIP-event rates are
retained as cross-checks, no subtraction/correction. Ring dumps happen after
each CTA's own issue/completion intervals; epilogues can overlap other CTAs.

## Broad knees

1. **1–2 CTAs:** close to linear aggregate scaling; one/two/four-wave CTAs
   behave alike, as in the preceding single-CTA result.
2. **4–8 CTAs:** first sublinear/flat region. Four-CTA results vary across
   repeats, and going to8 gives little gain. This is near the nominal8 MiB
   L2 footprint at8 CTAs, but the bend starts earlier and no cache/partition
   counters isolate the cause.
3. **16–32 CTAs:** scaling resumes in the larger-source regime. **32–64 CTAs
   is the broad best-throughput region**; little useful improvement beyond.
4. **104–208 CTAs:** not just a benign plateau—throughput regresses. One-
   and two-wave CTAs hold up better; four-wave CTAs lose substantially more.
   We retain/report that regression and do not try a different optimization.

The curves do not describe one monotonic universal saturation threshold.
They couple source footprint, concurrent load and mapping/scheduling. No exact
cache/HBM/return-port bottleneck or general register-bandwidth ceiling claim.

## Numeric record (not a precise tuning recommendation)

Requested GB/s from grid issue makespan, run1/run2; rounded for readability:

| CTAs | 1 wave/CTA | 2 waves/CTA | 4 waves/CTA |
|---:|---:|---:|---:|
| 1 | 107 /107 | 107 /107 | 107 /107 |
| 2 | 213 /214 | 214 /214 | 214 /214 |
| 4 | 340 /361 | 425 /345 | 375 /364 |
| 8 | 407 /384 | 413 /382 | 373 /373 |
| 16 | 727 /726 | 722 /723 | 723 /710 |
| 32 | 1322 /1336 | 1307 /1321 | 1257 /1244 |
| 64 | 1411 /1363 | 1376 /1455 | 1349 /1374 |
| 104 | 1227 /1166 | 1243 /1240 | 770 /753 |
| 208 | 1185 /1184 | 1304 /1109 | 858 /835 |

These are only three-sample process medians; early4-CTA and late2-wave cases
have material variation. Do not rank tiny differences or treat64 CTAs as a
precisely selected winner.

HIP occupancy queries report12/6/3 CTAs/CU at1/2/4 waves, respectively, for
this actual kernel. Every tested grid is below the corresponding theoretical
whole-device residency capacity, including208<312 for four-wave CTAs.
**All measured CTA issue intervals overlap in every sample.** Thus the large-
grid regression is not explained by a forced oversize-grid second admission
wave. This is capacity/overlap evidence, not a claimed CTA-to-CU placement.

Measured core/steady ratios fall modestly from~1682 MHz at1 CTA to~1630–1646
at large grids. That alone is too small to explain the four-wave collapse;
the104/208 four-wave cases do not even have lower measured ratios than64.
Full-event bandwidth closely tracks body bandwidth (within~0.6% throughout),
so it is not merely an event-epilogue accounting artifact. This does not
eliminate all scheduling, cache-mapping, memory-clock or interference effects.

## Validation / replay

**162 checked timing launches /7902 CTA records**, plus three multi-CTA fill
smokes. Every final ring slot/lane/DWORD and all32 global output guards pass.
Same periodic-payload/last-writer-oracle limitations as VGPR_RING.md apply.
No intermediate data consumers or payload checks are inside timing.

Raw per-CTA/per-wave clocks, process medians, occupancy queries, commands,
source/native manifests and extracted loops: `results/grid-vgpr/`.
Actual distinct executable/full ISA/CDB: `temp/lds-staging/grid-vgpr/`.
The previous single-CTA executable/source remains unchanged.

```sh
bash research/lds-staging/build-grid-vgpr.sh
numactl --cpunodebind=0 --membind=0 temp/lds-staging/grid-vgpr/probe \
  --ctas 64 --waves 4 --rounds 3 --warmup-ms 100
python3 research/lds-staging/run_grid_vgpr.py --output <fresh-results-directory>
```

No per-load wait added, no follow-on refinement/tuning, and no production/
CMake/model/system/package/clock-policy changes. Research status belongs in
`research/research-log.md`, not engine-design.md.
