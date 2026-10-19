# Idle llama server stopped: limited performance-only recheck

The user discovered a resident but reportedly idle llama.cpp server during
prior measurements, then stopped it. This recheck asks whether its presence
explains the current fixed-address ~96/~129-cycle floor. It does not claim
to revalidate all historical GEMM, bandwidth or multi-GCD measurements.

`amd-smi` snapshots before and after this recheck show no running processes
and ~10 MiB allocated on each GCD. HIP 0 maps to AMD-SMI GPU 2,
BDF `0000:31:00.0`, not AMD-SMI GPU 0. The earlier label “no model/context”
refers to the standalone probe's workload, not a prior assertion that no
other process had a model resident. The user's earlier snapshot establishes
that the llama server did have substantial VRAM residency.

## Protocol

Same waited/unwaited executables and code objects (hashes verified), one
256-thread CTA/four N waves, native HIP O3/C++20, gfx90a, HIP 0, CPU+memory
NUMA node 0, no model/context in this probe, no clock/power changes. Rerun
32 and 4096 steps (logical K256/K32768), two nine-round processes per mode/
length, seeds 1729/2718 with reversed length/mode order. Both pre-read and
completed-eviction preparations, original two-second warm-up unchanged.
No output readback, numerical tests, fixtures or correctness gates.

Comparison is with `results/shallow-k/`, measured before the user stopped the
server. The process state was not alternated experimentally; do not uniquely
attribute small timing differences to server residency/removal.

## Long-loop result

4096 steps, pooled medians, **core cycles/MFMA**, in-dispatch pre-read:

| Case | Server resident | Server stopped |
|---|---:|---:|
| FP8 D8 waited | 96.106 | 96.106 |
| FP8 D8 no wait | 96.113 | 96.109 |
| FP16 D8 waited | 128.998 | 129.226 |
| FP16 D8 no wait | 128.563 | 128.875 |
| Resident anchor, no-wait executable | 64.040 | 64.124 |

The eviction-prepared D8 result is equally unchanged in substance: FP8 no-wait
96.071 -> 96.093, FP16 no-wait 128.714 -> 128.851. Across **all** long-loop
cases/depths/cache preparations, the largest absolute median change is
0.862% (unwaited FP8 D1, eviction preparation, slightly slower).

## Short-loop result and limits

At 32 steps, pre-read FP8 D8 no-wait is 110.594 -> 110.203 cycles/MFMA.
FP16 D8 no-wait is 248.312 -> 274.969; short anchor/other-depth measurements
also vary, without consistent improvement after stopping the server.
The inherited selective warm-up and finite-length limitations from
[shallow-k.md](shallow-k.md) remain. These short-case changes do not establish
a causal server effect.

**Conclusion:** stopping the idle server does not remove the current
~96-cycle FP8 / ~129-cycle FP16 steady-state floor. There is no evidence in
this recheck that it caused that problem. This is narrower than saying idle
VRAM residency can never affect any benchmark.

Eight timing CSV/log pairs, before/after `amd-smi` snapshots, original flags/
hashes and full comparisons are in `results/server-off/`. Recreate comparisons
with `python3 research/fp8-w8a16/analyze_server_off.py`. No kernels changed.
