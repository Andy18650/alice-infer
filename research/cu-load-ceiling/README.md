# CU load-delivery ceiling: pure-load verification of the fixed-probe 32 B/cycle clue

User-directed standalone research, scoped down on request: **one CTA**, hot/LDS
paths only (no L2-rotation, stream, or all-CU grid modes), no 12-byte loads.
The question comes from the fp8-w8a16 [fixed-K diagnostic](../fp8-w8a16/fixed.md):
with every load cache-hot and addresses frozen, that probe settled at
~96 cycles/MFMA (FP8 B) / ~129 (FP16 B) — about **31.96 / 31.68 logical load
bytes per CTA core cycle** — and hypothesized "each CU can receive at most
~32 bytes per cycle". That probe entangles loads with MFMA issue, FP8
conversion VALUs, A copies and paired waits; this probe removes all of that.

**Result: the 32 B/core-cycle figure is NOT a universal per-CU receive
ceiling.** A single CU sustains **~64 B/core-cycle of hot dwordx4 loads**
(4 or 8 waves), **~63 B/core-cycle of hot dword loads** (8 waves), and
**~125 B/core-cycle from LDS** — but only **~32 B/core-cycle of dwordx2
loads**, because dwordx2 and dwordx4 take the *same* ~64 cycles per
instruction at 4 waves. The hot ceiling is **per-instruction-shaped, not
per-byte-shaped**: collapsing a dwordx4's unique data from 1024 B to 16 B
(64 lanes sharing one address) leaves its cost unchanged (64.2 → 64.4
cycles). The fixed probe's "32 B/cycle" is consistent with an instruction-
shaped service constraint (~1 wide load per ~14–16 CU cycles, ~1 dword load
per ~4–5 CU cycles) acting on its 8-loads-per-step pattern, rather than a
universal byte-rate wall. This does not identify a unique return-path bottleneck.
No hardware counters were used; the limiting unit is not identified.

## Probe design

`load_ceiling.hip` times pure load rings on one CTA (grid=1x1, block=64*waves):
each iteration issues exactly one load per wave into a Batch-deep register
ring, then `s_waitcnt vmcnt(Batch-1)` (or `lgkmcnt` for LDS) retires the
oldest entry and a single `v_xor_b32` folds element 0 into a checksum. The
wait and first read stay in one asm block (the fp8-w8a16 scheduling lesson).
Every element of each ring slot is passed as an asm input to keep the whole
slot register-live (see hazards below). One `v_xor_b32` per load is the only
consumption work; there is no MFMA, no conversion, no A copy.

- **Widths**: dword (4 B/lane), dwordx2 (8), dwordx4 (16) per lane,
  64 lanes/wave, coalesced per-wave fragments of 64*W bytes.
- **Modes**: hot (fragment 0 repeated, addresses frozen like fixed-K),
  hot2 (two alternating fragments: data varies, still L1-hot — stale-read
  control), LDS (`ds_read_b32/b64/b128`, contrast path).
- **Sweeps**: waves/CU {1,2,4,8} (block 64..512), ring depth B {1,2,4,8,16},
  address duplication dup {1,2,4,8,16,32,64} (lanes share addresses).
- **Protocols**: cache=0 (in-dispatch pre-read, hot) vs cache=1 (completed
  256 MiB eviction reader, ring priming re-warms the 1–2 KiB active set).
  They converge (max median delta 6.35 cycles/load, one case; all others
  < 0.8), matching the fixed probe's convergence.

Every timed sample is checked on the host against an exact per-dword XOR
oracle (`mix32`), including output guards; 35 full-width step-64 verify
launches cover every width/mode/dup/depth before timing. The ISA audit
(`audit_load_ceiling.py`) verifies on the actual executable: exactly 32 ring
loads and 32 `vmcnt(B-1)` waits per unrolled body, 1 XOR per load (W/4 for
verify variants), invariant hot/LDS address registers and no body rewrites,
pessimistic FIFO safety with 325 injected-premature-read negative controls,
a `vmcnt(0)` drain before the end timestamp, and **zero scratch / zero LDS**
in all global rings.

## Protocol

Native HIP O3/C++20, gfx90a (MI250X), ROCm 7.1 / Clang 21.1.8. HIP 0,
one timed CTA, CPU+memory NUMA node 0, no model/context, no clock/power
settings changed. Effective core rate ~1684 MHz via clock64/wall_clock64
cross-check. 4096 loads/wave per sample (1024 and 8192 controls), nine
rotating-order rounds x two independent processes (seeds 1729/2718),
randomized case/cache order, 2 s warm-up. Zero invalid samples in all runs.

## Main result (medians; run1/run2 agree; 8192-step controls stable)

Hot, B=8, dup=1 — **cycles per load per wave** and gross B/core-cycle/CU:

| W (B/lane) | waves=1 | waves=2 | waves=4 | waves=8 |
|---|---|---|---|---|
| 4 (dword) | 22.0 → 11.7 | 21.3 → 24.0 | 21.3 → 48.0 | 32.6 → **62.9** |
| 8 (dwordx2) | 21.3 → 24.0 | 32.8 → 31.3 | 64.1 → **31.9** | 105.9 → 38.7 |
| 16 (dwordx4) | 21.9 → 46.8 | 32.1 → 63.8 | 64.2 → **63.8** | 111.7 → **73.3** |

Ring depth (hot, W4, waves4): B1 = 124.9, B2 = 63.0, B4 = 32.7, B8 = 21.3,
B16 = 22.0 cycles/load — B1/B2/B4 scale as latency/depth (L1-hot global-load
latency ≈ 125–130 cycles), B8 saturates, B16 gains nothing.

Dup sweep (hot, B=8, waves4) — cycles per load:

| W | dup=1 | dup=2 | dup=4 | dup=8 | dup=16 | dup=32 | dup=64 |
|---|---|---|---|---|---|---|---|
| 4 | 21.3 | 21.6 | 21.5 | 21.6 | 21.6 | 21.6 | 21.5 |
| 16 | 64.2 | 64.4 | 64.5 | 64.5 | 64.3 | 64.5 | 64.4 |

Unique bytes per instruction fall 64x; cost does not move. **The hot
ceiling is not byte delivery.**

LDS contrast (B=8, dup=1): dword 21.5 cyc (47.6 B/cyc), dwordx2 21.5 cyc
(95.0 B/cyc), **dwordx4 (b128) 32.7 cyc → 125.3 B/core-cycle** at 4 waves —
essentially the 128 B/clk LDS width. The same CU "receives" 4x the claimed
32 B/cycle through the LDS path, so any real ceiling is specific to the
global-load return path.

Hot2 (data-varying, L1-hot) matches hot (W4: 21.7 vs 21.3; W16: 64.4 vs
64.2): frozen addresses did not distort the fixed-K style measurement.

## Interpretation

- The repeating cycles-per-load ladder (≈21.5 / 32.7 / 64.2 / 111.7 for
  1/2/4/8 waves) is width-independent for dwordx2/dwordx4: at 4 waves both
  take ~64 cycles per instruction. Per CU that is ~1 wide global-load
  instruction per ~16 cycles (waves=2: ~1/16; waves=8: ~1/14), and ~1
  dword load per ~4–5 cycles (W4, waves 4–8). The byte rate follows from
  the width, not the other way around.
- Reconstructing the fixed probe from these primitives: its FP16 B config
  issues 8 dwordx2 loads per step (4 waves x A+B) → 8 x ~16 ≈ 128 cycles,
  measured ~129. Its FP8 B config issues 4 dwordx2 + 4 dword per step →
  4 x 16 + 4 x 4.5 ≈ 82, plus ~10–11 conversion/copy VALUs per wave
  (~15–20 cycles of VALU-serialized work) ≈ 96–100, measured ~96. The
  "31.96/31.68 B per core-cycle" pair is then a coincidence of the two
  instruction mixes, not a common byte wall.
- This is a timestamp-only study: which hardware unit imposes the
  per-instruction rate (L1 tag/data return, VMEM issue, scoreboard
  granularity) is not identified. The dwordx2-vs-dwordx4 equality at equal
  cycles and the >64 B/cyc point at 8 waves (73.3) are recorded as open
  observations.

## Compiler hazards found while building this probe

Both are the async-VGPR-write hazard class the fp8-w8a16 rounds documented,
in new forms; both are now excluded by construction and by the audit.

1. **3-dword vector types are 16 bytes on this toolchain**
   (`sizeof(uint32_t __attribute__((ext_vector_type(3)))) == 16`), so
   `U3*` pointer arithmetic strides 16 bytes: every dwordx3 lane silently
   read the wrong address. All addressing here is done in dword units from
   the raw buffer. (The scoped-down probe dropped width 12 entirely.)
2. **Partially-dead ring slots corrupt the allocation.** With only element 0
   consumed, the register allocator (a) spilled *pending* ring registers to
   scratch — storing VGPRs before their loads returned, which "worked" from
   L1/L2 by latency luck and failed nondeterministically on DRAM-latency
   patterns — and (b) reused the dead slot-window registers as load
   *address* registers, which pending data returns then clobbered
   (data corruption and `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`).
   Runtime `ring[k%B]` indexing (k depends on runtime `steps`) was the
   trigger; the fix makes every slot index a compile-time constant (steps
   is a multiple of 32, Batch divides 32) and passes all W/4 elements as
   asm inputs so slots stay fully register-live.

## Reproduce

```sh
build=temp/cu-load-ceiling/<fresh-build>
results=temp/cu-load-ceiling/<fresh-results>
bash research/cu-load-ceiling/build-load-ceiling.sh "$build"
python3 research/cu-load-ceiling/audit_load_ceiling.py "$build"
mkdir "$results"
numactl --cpunodebind=0 --membind=0 "$build/load-ceiling-probe" 4096 9 1729 >"$results/run1.csv" 2>"$results/run1.log"
numactl --cpunodebind=0 --membind=0 "$build/load-ceiling-probe" 4096 9 2718 >"$results/run2.csv" 2>"$results/run2.log"
numactl --cpunodebind=0 --membind=0 "$build/load-ceiling-probe" 1024 3 31415 >"$results/small.csv" 2>"$results/small.log"
numactl --cpunodebind=0 --membind=0 "$build/load-ceiling-probe" 8192 3 16180 >"$results/longer.csv" 2>"$results/longer.log"
python3 research/cu-load-ceiling/analyze_load_ceiling.py "$results"
```

Raw runs, logs, hashes, audit and summary are in `results/load-ceiling/`;
ISA, loops, metadata and binaries are in `temp/cu-load-ceiling/`. No engine
or CMake changes; this does not touch the fp8-w8a16 checkpoints. The active
checker writes only into the selected build/check directory, not back into
the curated result set. Use fresh paths to preserve measured binaries/data.
