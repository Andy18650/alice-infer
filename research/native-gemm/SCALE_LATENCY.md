# Scale-load latency, pair queue and packed-drain/read experiments

All three user-approved proposals implemented and measured once in a matched
four-way run. No search for a large win, no follow-up tile/interleave sweep.
**Keep the simple late load:** it gives a clear104-CTA gain. Pair batching and
moving q1 reads are mixed/near noise and are not promoted as improvements.

## Variants

### Baseline

Corrected packed-FP32 kernel from `scaled-packed-fixed`, with current-scale
SMEM load immediately before group-head LGKM0. That wait unnecessarily exposes
a load whose value is only needed for the next handoff or final scale flush.

### Late: defer the current-scale request

At group start retain `previous_scale=current_scale` but issue no scalar load.
After q1 has drained the previous raw bank, issue the new current-scale load.
q2/q3 provide eight independent MFMA issues before the next stage's existing
retirement LGKM0. The new scale is not read by these MFMAs.

That existing wait establishes readiness for the later group handoff. A
single-stage final group has no next-stage head wait, so an explicit final
LGKM0 precedes its current-scale consumption. Zero K bypasses all metadata.
The steady loop changes ONLY the location of the scale request/pointer update;
all other native hot-loop instructions/order match after projecting out those
three instructions per request. No raw-bank, payload, layout or FMA move.

### Batch: deferred two-scale SGPR queue

Repack only the scale table from logical `[K128][N128]` to physical
`[N128][K128 padded-to-even]`. Load two contiguous scales with one
`s_load_dwordx2 s[20:21]` after the first group's q1 drain. The next existing
head LGKM0 completes the queue. Copy low to current after that stage's first
q0 MFMA; copy high at the second group's handoff. Final first-bank flush waits
and selects queue low explicitly, covering single-stage/partial endings.

The logical table/oracle stays K-major and independent of physical packing.
Odd scale-group counts allocate a zero-padded final word. It may be fetched
as part of a pair but is never applied to an output. This is not an unguarded
out-of-bounds read. Actual metadata requests are ceil(groups/2),physical words
2*ceil(groups/2), versus groups single-word requests. No weight overfetch.

Two additional SGPRs (51->53),no additional VGPRs. This measures queue/layout/
request/copy changes together,not an isolated cache-latency or bandwidth limit.

### Read: late plus earlier next-q1 operands

Based on **late**, NOT batch. At first-stage q1's final MFMA, issue its next
operand-slot LDS reads before the last eight packed scaling FMAs, rather than
after the drain/metadata request. Suppress their original later issue; final
stages do not prefetch another q1. No extra read or wait is added.

These destination registers are raw SrcA/B operands v8:15,not MFMA results or
SrcC banks v128:255. This restores the immediate post-MFMA q1 refill point used
by the original unscaled pipeline. gfx90a's relevant MFMA->VALU destination
and SrcC hazards do not describe this SrcA/B refill; LLVM21's hazard checks
agree. The matrix-result-to-scaling handoff remains unchanged after the new
q0 round; no assumption of automatic MFMA-result RAW protection is made.

Exact source executed flows at29 stage counts and whole native instructions
show ONLY the planned q1 read-pair movements relative to late. Same resources.

## Resources and fixed work

All variants:256 architected VGPR,0 AGPR,private0,16 KiB LDS,reportedcap2CTA/CU.
SGPR51 for baseline/late/read,53 for batch. Same128 MFMAs/64 packed scaleFMAs,
24 globalx4/32LDSwrites/64LDSreads,16barriers/24LGKM+VM waits per eight-stage
steady loop. Late/read retain two single-word scale loads; batch uses one pair.
Two extra final-path LGKM0 sites (one executes for nonemptyK) are outside the
steady loop and establish deferred-load readiness for partial last groups.

The raw-bank/scale model commits pending SMEM results ONLY at explicitLGKM0.
It rejects reads of pending scale registers,tracks the correct scale ID for
every raw partial and every output update,and checks ordered exactly-once
contributions. All requests have at least eight MFMA issues before their next
LGKM0. This is an order/readiness proof,not a hardware latency emulator.

## Matched measurements

HIP O3/C++20 gfx90a HIP0/BDF31:00.0,CPU+memoryNUMA0,block256,automatic clocks,
no model/context/clock override. Full useful GEMM FLOPs/full kernel HIP-event
time;packing,H2D,scale-table repack/upload,CPU reference/checking excluded.
Each variant uses two nine-sample process medians;variant+problem order reverses
in repeat2,200ms selected-kernel warmup. Same logical nonuniform scales/weights.

K32768 TFLOP/s,run1/run2:

|CTAs|Baseline|Late|Pair batch|Late+early q1 read|
|---:|---:|---:|---:|---:|
|104|120.331/120.619|125.390/125.454|125.277/125.544|125.345/125.477|
|208|136.927/136.978|137.277/137.328|137.183/137.100|137.250/137.288|
|312|139.194/139.098|139.565/139.590|139.574/139.646|139.640/139.636|
|1664|143.428/143.443|144.631/144.634|144.712/144.692|144.581/144.560|

K8192:

|CTAs|Baseline|Late|Pair batch|Late+early q1 read|
|---:|---:|---:|---:|---:|
|104|113.594/113.665|117.258/117.334|117.574/117.731|117.415/116.628|
|208|131.435/131.432|131.832/131.680|131.534/131.779|131.584/131.779|
|312|134.733/134.592|135.151/135.148|135.081/135.148|135.256/135.182|

M/N:1664/1024,3328/1024,4992/1024,4096/6656 respectively.104/208/312
coverage retained. Capacity2 does not establish uniform per-CU placement;
312 also tests admission beyond total208 concurrent CTA capacity.

Late improves long-K useful rate~4.11% at104,~0.26% at208,~0.31% at312,
~0.83% on large grid. Short-K104 gains~3.23%. The repeat agrees and resources/
hot instruction counts stay fixed; the request readiness/order is the change.
No unique scalar-memory latency value is inferred from those end-to-end gains.

Batch versus late: no consistent extra win (e.g.208long~0.12% worse,104short
~0.30% better); most deltas are near noise. Earlier q1 reads are also mixed:
104short~0.23% worse with noticeable repeat spread,large~0.04% worse,312tiny
positive. Neither justifies promoting added complexity on this evidence.
No claim of statistical significance for sub-percent differences from two
process repeats. No further schedule tuning was done after these results.

## Checks / records

**789 checked launches /341 hashed processes**:504 timings across four
variants and285 correctness cases across three new variants. Zero/single/
partial K128 groups,eight-stage boundaries,longK65536,M/N tails/input seeds,
ones/nonuniform dyadic/non-power-of-two/zero scales,and all-finite FP8 short
cases pass independent logical int64-partial/orderedstd::fma references.
Full/sampled reference policy unchanged. Every output has finite-after-NaN-
poison/guard checks. Corrected timestamp word selectors and independent body-
span<=HIPevent containment checks apply to every launch; raw per-wave dumps
recompute the timing metrics for all timed launches.

29-stage-count metadata readiness/bank/order/padding/no-overfetch checks for
each plan pass; source and actual native read movement isolated. Final empty
K and single-stage groups explicitly tested. Padded metadata is bounds-checked
in the model and numerical GPU cases with odd K-group counts/multipleNtiles.

```sh
bash research/native-gemm/build_scaled.sh scaled <fresh-late> all late
bash research/native-gemm/build_scaled.sh scaled <fresh-batch> all batch
bash research/native-gemm/build_scaled.sh scaled <fresh-read> all read
python3 research/native-gemm/run_scale_latency.py --late-build <fresh-late> --batch-build <fresh-batch> --read-build <fresh-read> --output <fresh-results>
python3 research/native-gemm/audit_scale_latency.py --output <fresh-audit>
```

Actual source/build/native/resource/driver/commands/results/clocks under
`results/scale-latency/`;readiness/padding/order checks in its `checks/`
subdirectory (`summary.json` is final, `initial-summary.json` retains the first run).
New builds: `temp/native-gemm/scale-latency/{late,batch,read}`. Historical
packed-FP32 driver pins its completed snapshots. All old data/binaries remain
unchanged;normal/scaled host bridges use fresh namespaces after batch-table
scaffolding. No production/CMake/model/session/system/package/clock changes.
