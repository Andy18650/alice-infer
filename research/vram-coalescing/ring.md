# Rotating register-buffer experiment at 104 × 256

User proposal: fill a buffer, consume its oldest slot, immediately refill
that slot, consume/refill the next, and finally drain. Aim: leave other
reads outstanding while doing accumulation with one CTA per CU.

## Result: modest repeatable gain for B, not A

The unrestricted C++ ring is mostly scheduled back into drain-and-refill
batches and does not improve the winner. A separately labeled ordered ring
constrains each consume/refill boundary with an AMD compiler scheduling
barrier (not a GPU synchronization or memory-completion wait).

The ordered depth-four loop implements the intended non-draining schedule:

- A: repeated `vmcnt(3)` before consuming/refilling each vector slot.
- B: repeated `vmcnt(15), (14), (13), (12)` for each four-scalar slot,
  then that slot's four new loads before the next slot is consumed.
- No `vmcnt(0)` inside these steady-state loops. Fill/drain/tails have their
  own waits, as expected. Cache hits can complete requests early: positive
  thresholds **permit** pending loads, not a measured guarantee of exactly
  that many in flight at every instant.

Two independent process runs, nine jointly randomized rounds × twelve
reads, native O3 HIP/C++20, gfx90a, HIP device 0 / PCI 0000:31:00.0,
CPU+memory NUMA node 0, no model/context, no clock overrides. 1/2/4 GiB
sizes, A/B batch controls and the preserved old kernel are interleaved in
one allocation per process. Rates are useful input bytes, not DRAM counters.

| Kernel | 1 GiB run 1 / run 2 | 4 GiB run 1 / run 2 |
|---|---:|---:|
| A batch, four chunks, 104×256 | 1444.0 / 1446.3 | 1399.2 / 1399.3 |
| A ordered ring, depth 4, 104×256 | 1405.2 / 1404.2 | 1391.0 / 1392.2 |
| B batch, four chunks, 104×256 | 1426.6 / 1434.9 | 1401.8 / 1398.9 |
| **B ordered ring, depth 4, 104×256** | **1470.3 / 1465.6** | **1423.7 / 1424.0** |
| Exact historical uint2/U16, 832×256 | 1400.4 / 1407.6 | 1387.1 / 1387.4 |

The comparison is paired by round, not merely best-of-sweep. B ring4 beats
its batch control in 7/9 and 6/9 rounds at 1 GiB; median paired differences
are +12.7/+10.8 GB/s. At 4 GiB it wins **9/9 rounds in both processes**, with
median paired differences +20.9/+18.4. The difference of medians at 1 GiB
is larger than the median paired difference because the sample distributions
are irregular. Do not inflate the gain by ignoring that distinction.

This is a small real benefit, not recovery of a sustained historical
1530 GB/s. It also beats the best contemporary A batch medians. Depths
1/2/8/16 do not beat ordered B depth 4 consistently. The repeatable B gain
means the requested failure/revert condition was **not triggered**; retain
this experiment rather than discarding the ring idea. A alone does not win.
The old configuration and source remain available unchanged for inspection.

## Code and compiler behavior

`ring.hip` retains ordinary and ordered rings as separate specializations.
The slots are statically unrolled, so the circular register schedule needs
no modulo operation or dynamic register-array indexing. Each slot holds a
HIP uint4; sum state and input are raw uint32_t. The source prime/consume/
refill/drain logic and all tails pass checksums.

The free ring's actual A/B depth-four loops reach `vmcnt(0)` **before their
first refill**, then refill as a batch. A source-level ring was not enough.
The ordered variant adds a compiler constraint after the initial fill and
after each refill. This keeps the intended alternating consume/load order;
no artificial holds, atomics, CTA/grid barriers or cooperative launch.

Depth-four ordered A and B both report 50 registers, zero scratch and
8-CTA/CU capacity; batch A/B report 54/55 and the same capacity. Register
allocation changes along with scheduling, so the bandwidth gain is not a
hardware-counter isolation of one unique mechanism. Some gains depend on
instruction format: A's wide-load ordered loop works but is slower, while
B's coalesced scalar version improves.

The historical control is **the exact code object extracted from the
preserved final-1 executable**, loaded through HIP module APIs. Its original
kernel is `read_stream<2,16,false>` at 832×256, and its source remains
`research/vram-ceiling/ceiling.hip`. Same uint32_t affine input and buffer
as the rings; the legacy argument counts uint2 vectors, not dwords. Its
current ~1400 result confirms the unresolved cross-campaign shift even
without recompiling that kernel. Reverting configuration would not by itself
guarantee recovery of the historical 1530 measurement.

## Reproduce / artifacts

```sh
research/vram-coalescing/build.sh
research/vram-coalescing/ring-run.sh temp/vram-coalescing/new-ring \
  temp/vram-ceiling/final-1/ceiling 9
```

The second argument is the historical executable; the runner extracts its
embedded gfx90a object. It does not substitute a new approximate kernel.
The reader-helper scheduling switch is off; only `_ring_ordered` cases use
their own explicit compiler constraints. Candidates with scratch are excluded
and recorded, rather than silently timed as raw memory reads.

`results/ring-{b,c}.csv`, paired per-round tables and `ring-resources.csv`
retain the final results. Full source/binary snapshots, hashes, raw rounds,
resource queries, disassembly and telemetry are under
`temp/vram-coalescing/ring-{b,c}/`. `ring-a/` is the initial unrestricted
experiment. Extracted ordered depth-four loop ISA is under `ring-b/`.
The ring remains wave indexed after the later [baseline indexing
simplification](indexing.md). Its batch controls use `read_wave_tiles` from
`wave-control.hpp` and its reader calls explicitly pass component stride 64,
so these comparisons do not silently switch B to block-wide stripes.
No production engine, package or system changes.
