# Active matrix after the result-led cull

## Keep

Exactly one GEMM: **64x64xK, StageK8, raw2,16 LDS slots/outer-unroll16**,
paired global displacements0/256 then2048/2304, source advance4096 per pair,
post-MFMA same-buffer LDS refill, partial LGKM2, VM24, current-slot retirement,
pre-barrier M0 setup and no bulk NOP.32 KiB dynamic LDS. These are measured
implementation choices, not hardware restrictions on future designs.

Two independent LDS-only kernels remain as **opt-in diagnostics**, not GEMM
competitors/default timing controls. The fourth kernel is the all-slot copy
validator. There is no larger-chunk GEMM, special two-slot GEMM, ring selector,
raw-depth/unroll/StageK generator axis or old-protocol branch in active code.

## Scan / decisions

Latest paired-global sweep: native O3 gfx90a HIP0, CPU+memory NUMA0, one CTA /
256 threads, K32768, no model/context/clock changes, packing excluded. Two
nine-sample process medians:

| Combination | Cycles/four-wave MFMA step | Decision |
|---|---:|---|
| K8/raw2,16 slots, paired global | 93.510 / 92.646 | Keep |
| Same,8 slots | 153.557 / 154.193 | Remove: clearly slower |
| Same,32 slots | 97.312 / 95.125 | Remove: slower both repeats, twice the LDS |
| K8,2/4 slots | Earlier persistent runs ~642/~227 | Remove as stale experiments; not claimed as fresh paired-global timings |
| StageK128/two-buffer control | 143.278 / 143.309 | Retire from active code/default runs; preserve reference evidence |
| StageK32/64 or single-buffer large chunks | ~187-314 or ~147-245 in earlier sweeps | Retire superseded experiments |
| Prior raw-depth builds, early-retirement, pre-MFMA reads, NOP gap, per-record pointer updates | Historical/source-specific | Preserve history only; do not regenerate/replay by default |
| Register-staged producer | Already retired, no tested win | Remains absent |

The selection is for this one-CTA research workload, not universal GEMM
optimality or a claim about every K/cache/multi-CTA regime. The32-vs16
ranking is repeated in earlier ring/scheduling studies too. We do not infer
that unmeasured combinations can never win; stale candidates no longer get
a permanent place in the active search. Machine-readable scanned timing rows
are in `results/active-cull/matrix-scan.json`.

No historical CSV/log/assembly/source snapshot or measured binary is deleted.
Pre-cull active code/runner is recoverable at checkpointb081cef and frozen
`results/paired-global/source/`. Historical notes contain commands for their
own checkpoints; those are not current CLI instructions.

## Current interface

```sh
bash research/fp16-gemm64/build.sh                    # optional single output-directory argument
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/active
numactl --cpunodebind=0 --membind=0 temp/fp16-gemm64/active/probe --k 32768
python3 research/fp16-gemm64/run.py                    # fresh output required
```

Old build(D,U,StageK,output), generator dimension flags, probe `--buffers`
and old matrix runner options are rejected, not silently aliased. Constant
CSV dimensions remain as provenance, not runtime search axes. The old
`run_stage8.py` sweep driver is removed; its historical source snapshots stay.

## Smaller verification/timing workload

Default runner:
- 22 K cases:0,1,7,8,9,15,16,17,119,120,121,127,128,129,255,256,257,513,
  32760,32768,65535,65536.
- Two additional K257 seed cases change A only/B only.
- Two independently warmed processes x nine checked timing samples at K32768.
- **24 verification cases +18 timing samples =42 checked samples.**

This keeps short-fill, pair parity, ring wrap, bulk/tail, odd long K and
independent-input coverage without multiplying by losing implementations.
Default runs do not launch archived baselines, larger-chunk controls or LDS
reference timings. `--full` explicitly selects144 winner-only verification
cases; `--diagnostics` adds four LDS-reference checks. An archived winning
binary/source may be supplied explicitly for a matched comparison, never
by default; its legacy `--buffers16` is used only for that frozen program.
All verification remains enabled and all4096 outputs/32 guards are checked.
Separate full-word/guard validation still covers all16 LDS slots.

## Cull validation, not an optimization claim

The surviving timed GEMM has **1652/1652 identical instruction/register lines**
to b081cef after addresses/branch displacements are normalized. The direct
transfer/math algorithm is unchanged. Native metadata has four GPU entries,
zero scratch/static-LDS/lane spills, and no timed accumulator copies/dtype
conversions. Removed selectors are negatively tested.

Default 42 checked samples pass, plus four opt-in diagnostic smoke checks.
Fresh native O3/HIP0/CPU+memory NUMA0/one-CTA/K32768 medians93.280/93.116
cycles/step (227.200/226.720 us) are a recheck, NOT a pruning speedup claim.
Records, code hashes, raw clocks, diagnostics, rejections and comparison:
`results/active-cull/`. No production/CMake/model/system changes.
