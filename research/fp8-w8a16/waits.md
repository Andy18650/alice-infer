# Remove the partial waits: performance only, deliberately unsafe reads

User request: remove `s_waitcnt` after the paired loads, use whatever is in
the source registers, and **stop numerical correctness testing** until the
performance model makes sense. This round does exactly that. No output
readback, CPU oracle, changed-input fixtures, output guards, eviction checksum
checks or numerical validity gates run in either executable.

## Change

`wait_probe.hip` preserves the fixed-K load/conversion/MFMA loop, compiled twice:

- `OMIT_PAIR_WAIT=0`: waited reference executable.
- `OMIT_PAIR_WAIT=1`: omit the `s_waitcnt vmcnt(2D-2)` string in both
  `take_pair8` and `take_pair16`. Read the raw registers immediately instead.

Input initialization, cache pre-read, two ready banks, raw-ring depth,
32-MFMA unroll, four N waves and fixed fragment-zero addresses remain.
The optional wait's unused immediate asm input remains, so the operand
constraints do not change. `inspect_wait_probe.py` extracts all ten streaming
loops from each executable. **After deleting waits and normalizing branch
displacements, every instruction AND register assignment is identical.**
No FIFO/readiness safety analysis is applied to the intentionally unsafe loop.

Blocking seed/pre-read waits outside the steady-state loop stay. The final
load drain stays before the end timestamp, preventing outstanding requests
from escaping the timed region. Thus this removes the repeated partial waits,
not all memory completion costs or hardware issue/backpressure constraints.
Neither output values nor their agreement between modes have been inspected.

## Timing protocol

Standalone native HIP O3/C++20, gfx90a, HIP device 0. One 256-thread CTA/four
wave64s along N; CPU+memory NUMA node 0; no model/context; no clock/power
changes. Two randomized nine-round runs at 4096 repetitions (1729, 2718),
three rounds at 8192 (16180), both in-dispatch pre-read and completed 256 MiB
eviction preparation. Both executables get two seconds of untimed warm-up.
Measured effective core rates are ~1.684 GHz; use core cycles/MFMA, not event
launch time (which includes any pre-read). No profiling/traffic counters.

The waited/unwaited process order is reversed for the second main run. No
kernel or numerical correctness tests accompany these timing processes.

## Results

Median **core cycles/MFMA**, pre-read, 4096 repetitions:

| B | D | Waited run1 / run2 | No partial wait run1 / run2 |
|---|---:|---:|---:|
| FP8 | 1 | 188.540 / 188.609 | 99.591 / 99.465 |
| FP8 | 2 | 103.501 / 103.072 | 99.212 / 99.644 |
| FP8 | 4 | 99.607 / 99.109 | 99.314 / 99.423 |
| FP8 | 8 | **96.100 / 96.106** | **96.109 / 96.109** |
| FP8 | 16 | 99.430 / 99.373 | 99.271 / 99.231 |
| FP16 | 1 | 173.730 / 174.086 | 128.693 / 129.101 |
| FP16 | 2 | 129.476 / 129.350 | 129.228 / 128.983 |
| FP16 | 4 | 129.318 / 128.999 | 128.921 / 128.790 |
| FP16 | 8 | **129.099 / 129.314** | **128.947 / 129.196** |
| FP16 | 16 | 129.062 / 129.279 | 128.730 / 128.917 |

Cache preparations remain close. At 8192 repetitions, waited/unwaited D8
FP8 is 96.054/96.056, FP16 128.493/128.562. Unwaited FP8 D1/2/4/16 is
~97.5–97.8; unwaited FP16 all depths ~128.5. The resident anchor remains
~64 cycles. The raw CSVs retain all cases and per-wave interval bounds.

**The explicit wait explains much of the shallow-depth penalty, but removing
it does not explain away the deep ~96/~129 floor.** This result does not
mean every individual original wait was free, or that no hidden stall exists.

## A quantitative working model

Instruction count is not clock count. MFMA itself has a ~64-cycle rate here;
the remaining instructions can overlap with it but real loads still request
and return data, even without software waiting for the results.

For a logical step across all four waves, requests are:

```text
FP8 B:  4 waves * 64 lanes * (8 A bytes + 4 B bytes) = 3072 bytes
FP16 B: 4 waves * 64 lanes * (8 A bytes + 8 B bytes) = 4096 bytes
```

Small unique input footprint does not remove these repeated shader requests.
A simple throughput model is:

```text
T_step ~= max(T_MFMA, payload / R_effective)
T_MFMA ~= 64 cycles
R_effective ~= 32 logical bytes per CU core cycle

FP8:  max(64, 3072/32) = 96 cycles
FP16: max(64, 4096/32) = 128 cycles
```

Here **R is fitted from the FP8 plateau**, not independently measured physical
cache bandwidth. The FP16 byte ratio then predicts its ~128-cycle plateau;
longer-run measurements closely follow it. This model accounts for the
observed format scaling and the unchanged deep floor with waits removed.

It does **not** identify a particular L1 port, return path, issue path,
register write path or bank conflict. All are still hypotheses; software
wait removal does not remove hardware resource limits. Do not promote a
fitted effective rate into an established physical hardware specification.
No extra optimization/ablation is included in this round.

## Artifacts and reproduction

```sh
bash research/fp8-w8a16/build-wait-probe.sh
python3 research/fp8-w8a16/inspect_wait_probe.py
for mode in waited unwaited; do
  numactl --cpunodebind=0 --membind=0 \
    temp/fp8-w8a16/wait/$mode/wait-probe 4096 9 1729
done
# Repeat with 4096 9 2718; longer control 8192 3 16180.
python3 research/fp8-w8a16/analyze_wait_probe.py
```

`results/wait/`: six timing CSVs/logs, mode-specific flags/hashes, instruction
inventories and summaries. `temp/fp8-w8a16/wait/{waited,unwaited}/`: separate
executables, code objects, ISA/metadata and extracted loops. All prior fixed-K
and correct real-K sources/binaries are preserved. This performance-only
probe is not a GEMM implementation and has no numerical correctness claim.
