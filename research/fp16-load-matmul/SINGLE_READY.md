# Single-ready-bank diagnostic: hot inputs mask a correctness failure

Historical experiment at checkpoint `3986b13`. Current mandatory-wait build
paths append `-waited`; this copied-R1 negative diagnostic is otherwise
retained. See [README.md](README.md) for current commands.

Follow-up: [direct raw consumption](DIRECT_RAW.md) removes these copies and
passes verification. Adding only a post-timestamp harvest guard to copied R1
still fails its varying-data controls; it does not repair this issue.

User's next goal is eliminating raw->ready copies. The requested first step
is **one ready A+B bank** to test immediate producer->MFMA consumption.
**Copies are deliberately retained in this round** to isolate that change.
No copy-free implementation, NOP repair or delay tuning is included.

## Result

**The issue returns as a correctness failure, not a clear throughput collapse.**
With waits enabled, one ready bank passes hot operands but fails every tested
record-varying mode (global L2/stream and LDS sweep). Two-bank controls pass.
All guards remain intact. Fast invalid runs must not be called correct matmul
throughput.

| Verification-only cases | Cases | Invalid |
|---|---:|---:|
| One ready bank, hot/global or hot/LDS | 60 | 0 |
| One ready bank, record-varying | 90 | **90** |
| Two ready banks, all modes | 135 | **0** |

This includes original legacy values, hashed varying inputs and changed A/B
seeds, D1/D2/D8 at U32, all three load shapes; one-bank D3/U8 short runs use
an odd seven-record cyclic window and exit mid-phase. Stream allocations/
pitches are explicitly matched between R1 and R2. All 4096 outputs are checked
exactly per pass, plus 32 guard words. The reference is unchanged from the
previous [full verification](VERIFICATION.md).

The separate warmed timing comparison adds 198 launches per variant: all
198 R2 launches pass; **72 R1 sweep launches fail** and 126 R1 hot launches
pass. Combined: **681 checked launches, 162 invalid**, all invalid launches
belong to R1. The study exits nonzero intentionally and preserves failures.

## What changes in the pipeline

R2, existing schedule:

```text
ready0 holds operand 0; ready1 holds operand 1
prime raw operands 2..D
step n: MFMA ready[n%2], load n+D+1, wait, copy raw n+2 to ready[n%2]
```

R1 diagnostic:

```text
ready0 holds operand 0
prime raw operands 1..D-1
step n: MFMA ready0, load n+D, wait, copy raw n+1 to ready0
next step: MFMA immediately consumes that copied ready0
```

Both keep D raw slots and the same `loads*(D-1)` completion threshold (with
the existing LDS cap14). The memory issue->copy distance is unchanged; the
full intervening MFMA between copy and consumption is removed.

Raw registers still start at v8; the address and scratch registers keep the
same mapping. **v4..v7 stay explicitly reserved even for R1** so this diagnostic
does not quietly gain a different raw-register layout or occupancy budget.
R1 reads/writes v0..v3 only for its ready operands. Four `v_mov_b32` per MFMA
remain; no NOP is inserted. Depth and unroll phase logic use lcm(D,R).

Representative R1 hot LDS D2 body:

```asm
v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[0:15]
ds_read_b128 v[12:15], v16
s_waitcnt lgkmcnt(1)
v_mov_b32 v0, v8
v_mov_b32 v1, v9
v_mov_b32 v2, v10
v_mov_b32 v3, v11
v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[0:15]
```

## Inspection and attribution limits

`inspect_loops.py` confirms the ready sources/copy destinations, raw sources,
wait/load/copy counts, no NOPs, no scratch/lane spill for all 210 entries in
seven builds. The new R2 timed instruction lists are identical to the earlier
verified R2 baseline after address/branch-displacement normalization. Whole
code objects differ; no byte-identical-module claim is made.

`inspect_operand_sequence.py` symbolically executes the generated instructions
for cyclic/streaming modes and all shapes. With loads assumed complete, it
supplies exactly records 0..steps-1 and words A0,A1,B0,B1 for both variants.
This checks the revised seed/lookahead/ring progression and rules out a simple
source-level indexing discrepancy under that model. It does not simulate
hardware load/writeback timing or establish the precise hazard mechanism.

The observed pattern is consistent with a VGPR producer/MFMA operand hazard.
Memory waits guarantee load completion before the copies, not arbitrary
VALU-write->MFMA spacing. This round does **not** prove the required delay,
which read/write boundary causes the error, or that two banks are universally
necessary. No attempt is made to repair R1 or remove the copies yet.

**Hot input is a weak test here:** each lane may have distinct values, but
its values repeat every K step, so stale/mixed generations can look correct.
Full-period cyclic runs also fail, so this is not explained merely by rotating
whole, correct A+B fragments by a constant offset. Guard failures are zero.

## Timings (do not certify failed cases)

Native HIP O3/C++20, gfx90a, HIP0 / BDF31:00.0, CPU+memory NUMA0, one CTA/
four waves, K32768, U32, no model/context or clock changes. Each selected
configuration independently warmed 200 ms, two randomized process-order runs,
nine checked samples/run; waits enabled. Absolute-wave CTA makespan timing.

| D / mode / shape | R1 cycles/step, run1 / run2 | R2 cycles/step, run1 / run2 |
|---|---:|---:|
| D2 hot LDS, b128 | 65.420 / 65.318 | 65.273 / 65.229 |
| D2 hot LDS, 2xb64 | 65.700 / 65.769 | 66.211 / 66.066 |
| D2 sweep LDS, b128 | **INVALID** 67.490 / 67.728 | 67.179 / 67.137 |
| D2 sweep LDS, 2xb64 | **INVALID** 67.736 / 67.901 | 68.101 / 67.921 |
| D8 hot global, dwordx4 | 65.018 / 65.023 | 65.013 / 65.018 |
| D8 hot global, 2xdwordx2 | 129.442 / 129.368 | 129.264 / 129.330 |

The warmed D1 wide LDS cases reproduce the same distinction: ~90 hot passes,
~100 sweep fails for R1, while both modes pass R2. There is no relevant speed
benefit established by these data. Default ready-bank count remains **2**.

## Reproduce

```sh
# Arguments: D U output LDS-stride ready-bank-count.
bash research/fp16-load-matmul/build.sh 2 32 \
  temp/fp16-load-matmul/d2-u32-lds1-verified-r1 1 1

# Reproduces an exact-check failure; expected exit code 2.
numactl --cpunodebind=0 --membind=0 \
  temp/fp16-load-matmul/d2-u32-lds1-verified-r1/probe \
  --mode lds-sweep --loads 1 --wait 1 --pattern hash --rounds 1

# Entire checked diagnostic; expected nonzero status for preserved R1 failures.
python3 research/fp16-load-matmul/single_ready.py
```

Use ready count2 for controls; no extra output argument defaults to the
`...-verified-rR` directory. `results/single-ready/` contains every CSV/log,
command, hash/inventory, symbolic progression record and marked summary.
All older binaries are preserved, and pre-change source is saved under
`temp/fp16-load-matmul/pre-single-ready/source/`. No production changes.
