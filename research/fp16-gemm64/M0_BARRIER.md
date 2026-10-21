# Move M0 setup before the barrier; remove the bulk NOP

The later [paired global-addressing candidate](PAIRED_GLOBAL.md) retains
this ordering and uses immediate displacements to halve pointer advances.
The measurements below remain the preserved baseline.

User-requested narrow scheduling probe. In each unrolled bulk unit, move
`s_add_u32 m0` immediately before the existing `s_barrier`, and remove the
explicit `s_nop 0` that previously separated M0 setup from its first use.
Startup and bounded tail retain their original setup/NOP handling, since
they do not all have the same immediate barrier adjacency.

```asm
; Before, checkpoint61904f8:
s_waitcnt vmcnt(24)
s_barrier
s_add_u32 m0, s0, 0
s_nop 0
global_load_dword v[8:9], off lds
global_load_dword v[8:9], off offset:256 lds

; After, measured16-stage candidate:
s_waitcnt vmcnt(24)
s_add_u32 m0, s0, 0
s_barrier
global_load_dword v[8:9], off lds
global_load_dword v[8:9], off offset:256 lds
```

The barrier now supplies the intervening instruction. No LDS storage is
written merely by changing M0; actual copies still issue after current-slot
retirement/publication. M0 is wave-local, restored after each pair as before.
Global/DS byte counts, two raw buffers, MFMA/read ordering, VM24/56, unroll,
source-pointer updates and timing brackets are unchanged. Bulk drops from
13 to12 instructions per wave/step, without removing any data operation.

## Matched results

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, one CTA /
256 threads,64x64x32768 GEMM; no model/context/clock overrides. Two randomized
process-order repetitions, nine checked samples per process after200 ms
selected-kernel warmup. Medians, run1/run2:

| LDS slots | Original barrier/add/NOP cycles/step | New add/barrier cycles/step | New whole body us |
|---:|---:|---:|---:|
| 8 | 155.773 / 153.102 | 153.681 / 154.852 | 373.920 / 376.640 |
| 16 | 99.338 / 99.248 | **96.311 / 96.273** | **234.400 / 234.240** |
| 32 | 100.706 / 100.577 | 97.574 / 97.834 | 237.440 / 238.240 |

Matched StageK128/two-buffer control:143.668/142.907 cycles/step. Pure-LDS
D2/U1 hot reference:64.195/64.187. The8-slot result changes direction between
repeats; no reliable gain is claimed there. Both16/32 repetitions save about
three cycles per four-wave MFMA step.

This shows a visible cost associated with the original per-step gap/setup
sequence at the larger rings, rather than it being fully hidden. It is
consistent with instruction-issue/write-use timing contributing to the
schedule. It does NOT uniquely identify scalar issue bandwidth, M0 latency,
barrier latency or a sole hardware bottleneck: setup placement and removal
of its explicit NOP are changed together. No further controls/tuning are
performed in this round. Body times cover all4096 steps/wave, excluding
packing/upload, verification and final harvest/output epilogue.

## Verification and assembly

**882 checked launches pass**:720 verification cases over2/4/8/16/32 slots,
plus162 timing samples including preserved baseline/control/diagnostics.
All4096 C values/32 guards and all allocated staging words/guards pass.
Coverage includes empty/short/odd/tails, unrolled bounds, split32-slot priming,
wraps, long K through65536 and independent input changes.

Native ISA checks enforce all of the following in the actual bulk body:
- No `s_nop` anywhere in the bulk.
- Each first global-to-LDS load is immediately preceded by `s_barrier`;
  that barrier is immediately preceded by `s_add_u32 m0`.
- Existing wait order, same-buffer DS reads immediately after MFMA, current-
  slot offsets, fixed parity and one backedge per group remain correct.
- No scratch/lane/accumulator spills, dtype conversion or payload DS writes.

Measured source/native hashes, raw clocks/CSVs, commands, snapshots and
inventories: `results/m0-pre-barrier/`. Baseline is the explicitly preserved
`temp/fp16-gemm64/post-mfma-read/probe` with frozen61904f8 source, not a second
runtime branch. Old data/binaries are untouched. Assembly:

- 16 slots: `results/m0-pre-barrier/builds/m0-pre-barrier/core-6.txt`, bulk
  around line288; corresponding8/32-slot files are core-5/core-7.
- Full disassembly: `temp/fp16-gemm64/m0-pre-barrier/probe.isa`.

No production/CMake/model or system/clock-policy changes.
