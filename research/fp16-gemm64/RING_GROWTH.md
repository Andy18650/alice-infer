# StageK8 ring growth:16 stages, then32

**Historical retire-next protocol.** The subsequent user-directed
[current-slot retirement](CURRENT_RETIREMENT.md) removes the forced bulk
LGKM drain and improves16 slots from ~158 to ~109 cycles/step. The plateau
reported below belongs to this earlier schedule; its data remain unchanged.

User-requested sequential test: try16 first; proceed to32 only if16 still
trails the best StageK128 control without a regression. Stop on capacity
failure/regression. Persistent two-raw-buffer algorithm and unrolling from
[STAGE8.md](STAGE8.md) are retained; no extra optimization is mixed in.
Ring depth and outer unroll remain coupled.

## Outcome

**16 improves on8, but32 does not improve on16.** Both fit and pass checks.
32 uses all64 KiB LDS, the capacity limit of this unpadded2 KiB/slot layout.
No allocation or kernel-launch failure occurred. Nothing larger was run.

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, **CPU+memory NUMA0**, one CTA /
256 threads, real64x64x32768 GEMM, no model/context/clock overrides. Two
randomized process-order repeats, 200 ms selected-kernel warmup, nine checked
samples per process. Final matched sweep medians, run1/run2:

| StageK | Slots | LDS total | Cycles/four-wave MFMA step | Entire body us |
|---:|---:|---:|---:|---:|
| 8 | 8 | 16 KiB | 165.986 / 166.375 | 403.840 / 404.800 |
| 8 | 16 | 32 KiB | **157.934 / 157.854** | **384.320 / 384.160** |
| 8 | 32 | 64 KiB | 158.934 / 158.408 | 386.720 / 385.600 |
| 128 | 2 | 64 KiB | 143.497 / 142.887 | 348.000 / 346.720 |

The initial16-stage sweep measured157.876/157.887 versus8 stages166.188/
166.107 and StageK128 two-buffer142.898/143.247. This authorized proceeding
to32. In the final matched sweep32 is0.35-0.63% slower than16: a small
observed regression/no win, not a variance-controlled hardware-cause claim.
We stop here as requested. Sixteen slots are the best observed K8 choice;
they still do not tie the larger-chunk two-buffer control.

Body times cover all4096 K8 steps/wave, not one stage. Cycle divisor is4096,
not aggregate16384 hardware MFMAs. Packing/H2D upload, verification and final
accumulator-harvest/output epilogue are outside body timing. Full event
intervals, clocks and individual samples remain in CSV.

## 32-stage counter-safe startup

Each direct copy uses two VMEM instructions/wave. Issuing32 copies before
any wait would request64 outstanding instructions. Instead:

1. Prime at most31 slots (62 VMEM instructions).
2. Wait for tile0 with vmcnt(60), then publish at a CTA barrier.
3. Issue slot31 if it exists: no more than62 instructions remain outstanding.
4. Read tile0 into raw0, then publish tile1/retire tile0 as before.
5. Refill slot0 with tile32 if it exists and enter the normal pipeline.

Steady wait is vmcnt(60), leaving later copies in flight; each refill adds
two instructions after the oldest required pair completes. No disabled
vmcnt(63) or64-instruction pending window is used. The ISA inspector checks
the62-load prefix, intervening wait, late two-load slot31 and unrolled bulk.
These counts are shader instructions, not physical HBM transactions/queues.

LDS offset maxima are63488 plus the existing lane/operand base; the final
b64 access ends at byte65536. Static LDS allocation is zero, so no hidden
shared-state allocation pushes the64 KiB dynamic allocation over capacity.
The byte checker validates every word in all32 slots, including the last.

## Verification / preservation

- 16-stage experiment: 234 checked launches, including 144 verification cases.
- 32-stage experiment: 252 checked launches, including 144 verification cases.
- **486 total exact checked launches**; all4096 C values/32 guards pass.
  Every GEMM invocation also checks all allocated LDS slots word-for-word.
- Inputs include empty/short fills, 30/31/32-stage startup boundaries,
  ring wraps, unrolled-bulk/tail boundaries, odd K through65536, independent
  A-only/B-only seed changes and long loops.
- Native code preserves alternating raw buffers, literal LDS offsets,
  16/32 MFMAs per bulk backedge and no hidden inner loop, scratch/lane spills,
  dtype conversions, payload DS writes or inner accumulator copies.
- Current StageK8 builds have five GEMM variants2/4/8/16/32, two LDS diagnostics
  and one copy validator: eight entries. Larger-chunk builds remain five.
- Both experiment sources, generated/native bodies, program/source hashes,
  commands and raw data are preserved separately at `results/ring16/` and
  `results/ring32/`; binaries are in distinct temp/ring16 and temp/ring32
  directories. Earlier measured binaries/results are untouched.

```sh
bash research/fp16-gemm64/build.sh 2 1 8 temp/fp16-gemm64/ring32-replay
bash research/fp16-gemm64/build.sh 2 4 128 temp/fp16-gemm64/ring32-control-replay
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/ring32-replay
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/ring32-control-replay
python3 research/fp16-gemm64/run_stage8.py --stages 32 \
  --build temp/fp16-gemm64/ring32-replay \
  --control temp/fp16-gemm64/ring32-control-replay \
  --output research/fp16-gemm64/results/ring32-replay
```

The runner now accepts explicit stage/build/control/output arguments and
refuses a nonempty output directory. Source/binary fingerprints and compiled
metadata are checked before results are accepted. No production, CMake,
model-format or clock-policy changes; no further tuning/counters this round.
