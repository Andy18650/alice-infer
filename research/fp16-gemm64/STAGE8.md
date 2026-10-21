# Persistent StageK=8 pipeline: two raw buffers, unrolled outer loop

**Historical retire-next schedule.** Current code instead uses
[partial LGKM waits and current-slot retirement](CURRENT_RETIREMENT.md).
The original2/4/8-slot measurements below are preserved. The later
[16/32-slot ring-size follow-up](RING_GROWTH.md) adds those choices and
counter-safe32-stage startup;16 is best observed,32 slightly regresses.

This replaces the [initial per-chunk helper experiment](STAGE8_INITIAL.md).
The user-directed change keeps **two raw register buffers across the entire
GEMM**, removes the inner compute loop from the K8 GEMM path, and fully
unrolls the outer bulk loop by the LDS stage count (2/4/8).

## Schedule and lifetimes

Each K8 slot still holds2048 unique packed FP16 bytes, shared by the four
waves in a 2x2 M/N grid. Producer roles/layout are unchanged. There is no
register-staging producer, FP8, dtype conversion, ready-bank copy or operand
VALU copy. Registers v0..3/v4..7 hold alternating A/B fragments; v8..9 hold
a persistent global pointer. One opaque asm owns all pending destinations,
so the compiler cannot copy/reallocate them between stages.

Startup publishes tile0, reads it to raw0, completes those reads, then
publishes tile1 and retires tile0's LDS slot at a CTA barrier. It can now
refill slot0 with tileN while tile0 remains available in registers.

Each steady step is:

```text
DS-read tile n+1 -> opposite raw buffer
MFMA tile n from already-complete current raw buffer
lgkmcnt(0)                    // complete NEXT prefetch, after current MFMA issue
vmcnt(2*(N-2))                // publish tile n+2; leave later copies in flight
CTA barrier                  // retire tile n+1 LDS reads AND publish n+2
refill tile n+1's retired LDS slot with tile n+N+1
```

Retirement follows LDS-read completion, not later MFMA consumption. This
lets the two-LDS-slot case use one steady barrier/step too. The post-MFMA
LGKM wait is required before overwriting the prefetched tile's LDS storage;
it is NOT a pre-MFMA wait for current operands or a raw-buffer reset.

The bulk loop explicitly expands N such steps. Each phase has **literal LDS
offsets and fixed register parity**. Global source pointers advance in place;
m0 is saved once for the whole operation and restored after each direct copy.
There is no per-record modulus, LDS address-wrap loop or compute-helper call.
Only one remaining-count update, comparison and backedge occur per N steps.

The bulk runs while remaining>2*N, making all its lookahead valid without
per-step bounds checks. Startup and a bounded tail handle K0, short fills,
odd K, ring boundaries and the last<=2*N records. Tail waits match the exact
remaining copy population. No out-of-range global lookahead is issued.

## Actual ISA checks

The inspector locates each native bulk backedge and checks:

- Exactly N MFMAs, alternating raw0/raw1, and2*N ds_read_b64 instructions.
- Read offsets2048,4096,...,2048*(N-1),0, each once for A and B.
- LGKM completion immediately AFTER MFMA issue.
- One counter subtraction and one conditional branch per N-step bulk body.
- Exactly N CTA barriers and steady VM waits0/4/12 for N2/4/8, without a
  stricter VM wait in that bulk body.
- No scratch/lane spills, dtype conversions, payload DS writes or inner
  accumulator moves. All global operand loads really target LDS.

The old compute helper remains only for larger chunks and pure-LDS diagnostic
modes; the StageK8 GEMM does not call it. `outer_unroll` in CSV explicitly
records2/4/8. The legacy `unroll=1` field describes the diagnostic helper,
not the new outer pipeline. StageK8 now requires raw depth2.

## Results

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, **CPU+memory NUMA0**, one
CTA/256 threads, actual64x64x32768 GEMM; no model/context/clock overrides.
Two randomized process-order repeats, 200 ms selected-kernel warmup, nine
exact-checked samples/process. Medians, run1/run2:

| StageK | LDS slots / outer unroll | LDS size | Cycles/four-wave MFMA step | Entire body us |
|---:|---:|---:|---:|---:|
| 8 | 2 | 4 KiB | 642.498 / 642.400 | 1562.880 / 1561.440 |
| 8 | 4 | 8 KiB | 227.601 / 227.034 | 554.080 / 552.640 |
| 8 | 8 | 16 KiB | **165.684 / 165.899** | **403.200 / 403.520** |
| 128 control | 2 / legacy U4 | 64 KiB | 143.217 / 142.946 | 347.200 / 346.880 |

The initial per-chunk version measured ~709-712/~435/~433 cycles for2/4/8
slots. The persistent implementation is substantially better, and4->8 slots
now helps. It still does NOT beat the larger-chunk control or establish the
~64-cycle compute floor. One CTA barrier/direct-producer sequence remains
per K8 step; their individual costs/hardware bottlenecks are not isolated.
The new D2/U1 pure-LDS references measure64.183/64.189 hot and82.365/82.357
sweeping cycles/step. The raw run log/snapshot inherited a D1 print-label typo;
CSV/build/assertions were D2 throughout, and the summary label is corrected.

Times cover the entire K32768 body, not one stage. Core-span divisor=4096
steps/wave, not aggregate16384 hardware MFMAs. Final accumulator guard/output
stores, CPU packing/H2D upload and verification are excluded from body timing;
full event intervals are retained as secondary fields.

## Validation and reproduction

**405 exact checked launches**:297 verification cases plus108 timed samples
(including the matched control and pure-LDS references). All4096 C values
and32 guards pass; GEMM invocations additionally check every word in every
allocated LDS slot. Coverage includes K0..65536, tails/fills/bulk thresholds,
multiple wraps and independent A-only/B-only changes. The larger-chunk
regression was also rebuilt and rerun: **340 exact checked launches** pass.
Commands now bind explicit executable paths and verify source/binary hashes.

```sh
bash research/fp16-gemm64/build.sh 2 1 8
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/d2-u1-t8-direct-only
numactl --cpunodebind=0 --membind=0 temp/fp16-gemm64/d2-u1-t8-direct-only/probe \
  --mode gemm --buffers 8 --k 32768 --rounds 9
bash research/fp16-gemm64/build.sh 2 4 128
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/d2-u4-t128-direct-only
# Use --output with a fresh directory; existing measurements are protected.
python3 research/fp16-gemm64/run_stage8.py \
  --output research/fp16-gemm64/results/stage8-replay
```

Current records, generated/native bulk bodies, source snapshots, clocks and
hashes: `results/stage8-persistent/`. Larger-chunk regression:
`results/direct-only-persistent-regression/`. Initial slow protocol:
`results/stage8/`, including its exact source and retained D1 binary. The
older misbound regression is retained and flagged in
`results/direct-only/PROVENANCE.md`; corrected pre-persistent checks are in
`results/direct-only-corrected/`. No measured binaries/results are deleted.
No production, CMake, model-format or clock-policy changes; no further tuning
or hardware-cause claim is included in this round.
