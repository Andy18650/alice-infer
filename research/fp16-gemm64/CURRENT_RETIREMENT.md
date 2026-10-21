# Rolling raw buffers: retire the current LDS tile

User-directed replacement of the early next-tile retirement schedule. Two
raw buffers and the same physical LDS ring remain. The next A/B read pair
may stay pending across the boundary; no bulk LGKM drain is required.
This is the current StageK8 implementation. The previous algorithm/binary
is preserved at checkpoint706b28e and `temp/fp16-gemm64/ring32/`.

## Schedule, N>=4

Startup publishes global tiles0 and1 and primes raw0 with tile0's LDS reads,
without draining that prime or overwriting slot0. Each unrolled step n:

```text
DS-read tile n+1 into raw[(n+1)%2]
lgkmcnt(2)                    // current n complete, next pair may remain pending
MFMA raw[n%2]
vmcnt(2*(N-3))                // publish global tile n+2, leave later copies pending
CTA barrier                  // retire CURRENT n; publish n+2
copy global tile n+N into retired LDS slot n%N
```

Only current's source slot is overwritten. The next read's source slot is
left untouched until the following step's current-read completion/barrier.
Thus it is safe for those two ds_read_b64 instructions to remain pending.
They may finish early; the schedule does not force them to stay pending.
Same-class DS operations are the only LGKM operations inside the bulk.

Compared with the prior schedule, one fewer K8 global-copy stage leads the
consumer. Physical LDS allocation and raw-buffer count are unchanged.
16/32-slot VM watermarks change28/60 ->26/58. Source pointer now starts at
tileN after priming, not N+1; no early initial refill occurs. The32-slot
split-prime limit of62 outstanding VMEM instructions remains intact.

Bulk still has N fixed-offset/fixed-parity phases and one remaining-count
update/backedge per group. No inner compute loop, payload copy, ready bank,
FP8, dtype conversion or register-staging producer is introduced. LGKM wait
moves before MFMA, and the bulk contains only lgkmcnt(2), never lgkmcnt(0).
The final tail record uses lgkmcnt(0), since there is no next pair to leave
outstanding. Tail global waits and refill bounds follow current retirement.
The conservative remaining>2*N bulk bound is unchanged for comparison.

### Two-slot special case

For N2, tile n+2 cannot be copied until n's slot is retired. It needs:
current partial wait/MFMA -> retirement barrier -> copy n+2 -> vmcnt(0) ->
publication barrier. This is two steady barriers/step, not the one used by
larger rings. The two-slot case remains correct and is fully verified; no
new performance claim is made for it. N4/8/16/32 keep one barrier/step.

## Matched timing

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, one CTA /
256 threads, real64x64x32768 GEMM, no model/context/clock overrides. Two
randomized process-order repetitions, nine checked samples/process after
200 ms selected-kernel warmup. Medians, run1/run2:

| LDS slots | Previous retire-next cycles/step | New retire-current cycles/step | New body us |
|---:|---:|---:|---:|
| 8 | 163.242 / 166.476 | 138.632 / 138.858 | 337.280 / 338.080 |
| 16 | 157.906 / 157.862 | **109.474 / 109.449** | **266.560 / 266.400** |
| 32 | 158.195 / 158.298 | 110.734 / 110.989 | 269.600 / 270.080 |

Matched StageK128/two-buffer control:143.130/143.483 cycles/step,
347.200/348.000 us. The new K8 schedule now beats that control;16 slots
remain best observed. Pure-LDS D2/U1 hot reference:64.180/64.202 cycles/step.
We do not identify the remaining hardware bottleneck or claim full-GPU speed.

One step means one MFMA in each of four waves. Core CTA span divides by4096
steps/wave, not aggregate16384 instructions. Body includes startup, loop,
waits/barriers and tail; excludes final accumulator-harvest/output epilogue,
CPU packing/H2D upload and verification. Data are already packed FP16.
The measured change couples partial waiting with slot retirement/global
watermarks; it is not an isolated cost measurement of the LGKM instruction.

## Validation, native ISA and timeout recovery

**882 exact checked launches across verification and timing controls:**
720 new-schedule verification cases cover all five ring depths2/4/8/16/32,
K0..65536, short fills, split-prime/bulk/tail boundaries, multiple wraps and
independent A-only/B-only changes.162 timing samples include new, archived
old and larger-chunk controls plus LDS references. All4096 C values/32 guards
pass. Each GEMM process also checks all allocated LDS words/guards separately.

Actual-code-object checks verify:
- lgkmcnt(2) immediately before bulk MFMA, no bulk lgkmcnt(0).
- Fixed next-read offsets and alternating raw0/raw1 operands.
- Refill offsets CURRENT0,2048,..., not next2048,...,0.
- VM thresholds26/58 for16/32, no extra stricter wait in that bulk.
- One bulk backedge and one barrier/step (two for N2).
- Counter-safe32-slot priming and zero scratch/static-LDS/lane spills,
  dtype conversions, payload DS writes or inner accumulator moves.

The original harness command reached its300-second timeout late in timing
repeat2, after all720 verification cases passed. It was not a GPU error.
Recovery reused732 completed process records and executed six remaining
processes. The one interrupted CSV/log pair is retained, not overwritten.
Source/native fingerprints and the original command-line arguments were
checked before reuse. Original and recovery invocations, commands, scripts
and AMD-SMI snapshots are separate. Completed CSVs were not rerun or relabeled.
Consequently the second repetition has an explicitly recorded recovery gap;
no uninterrupted or variance-controlled timeline is claimed.

## Reproduction

```sh
bash research/fp16-gemm64/build.sh 2 1 8 temp/fp16-gemm64/current-replay
bash research/fp16-gemm64/build.sh 2 4 128 temp/fp16-gemm64/current-control-replay
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/current-replay
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/current-control-replay
python3 research/fp16-gemm64/run_stage8.py --stages 16 32 --verify-stages 2 4 8 16 32 \
  --build temp/fp16-gemm64/current-replay \
  --control temp/fp16-gemm64/current-control-replay \
  --baseline-build temp/fp16-gemm64/ring32 \
  --baseline-source research/fp16-gemm64/results/ring32/source \
  --output research/fp16-gemm64/results/current-replay
```

`--resume` requires exactly the original arguments and identical saved
source/native fingerprints; complete cases are revalidated, not relaunched.
The baseline is an explicitly selected frozen executable with verified frozen
source, not a fallback or an active old-schedule branch in the current binary.
Records/native bodies/current and old source snapshots: `results/retire-current/`.
No production, CMake, model-format or system/clock-policy changes.
