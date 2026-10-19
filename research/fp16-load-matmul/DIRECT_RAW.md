# Copy-free raw-ring consumption: current completion wait is sufficient here

User asks to eliminate ready storage/copies and feed MFMA directly from the
raw entry whose completion is guaranteed by the current wait. A previous-
iteration arrival/extra stage is a fallback only if needed.

**Result: no fallback was needed.** All **1224 checked positive launches pass
exactly**, including depth1 and varying operands. There are zero numerical
or guard failures. The copied one-ready negative controls still fail **3/3**
with the same post-timestamp harvest guard, distinguishing operand readiness
from accumulator harvesting.

This supports counter-completed-load -> MFMA consumption on the tested
gfx90a paths. It is not a measurement of the counter's internal decrement
point or an exhaustive architectural proof for arbitrary operations/data.

## Direct schedule

Build ready-bank count **R=0**. There are D raw A+B slots and **no separate
ready banks**. Raw storage starts at v0 rather than v8, releasing the eight
registers previously reserved for ready operands. All four copy instructions
are removed. Layouts, MFMA instruction and the accumulator bank remain.

```text
prime raw operands 0..D-2
for logical step n:
    issue operand n+D-1 into raw[(n+D-1)%D]
    wait for the whole raw[n%D] load group to be complete
    MFMA directly consumes raw[n%D]
```

The deterministic head is the entry **guaranteed complete by that watermark**,
not an opportunistic discovery of whichever request actually returned latest.
Global waits use vmcnt(N*(D-1)), LDS lgkmcnt(min(N*(D-1),14)), N=loads/pair.
D1 uses wait0 and consumes the just-issued group. D2 supports overlap.
Increasing D is not used as an additional post-completion aging stage.
Deep LDS caps remain explicit; the D1 evidence is uncapped.

Actual hot LDS D2 loop begins:

```asm
ds_read_b128 v[4:7], v8
s_waitcnt lgkmcnt(1)
v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[0:15]
ds_read_b128 v[0:3], v8
s_waitcnt lgkmcnt(1)
v_mfma_f32_32x32x8f16 a[0:15], v[4:5], v[6:7], a[0:15]
```

There is **no VALU producer between completion wait and MFMA**, no ready copy,
no intervening MFMA to age the entry and no in-loop NOP padding. With waits
disabled the existing deliberate unsafe semantics remain; this round's claims
and measurements all have waits enabled.

## Harvesting guard, not an operand delay

The direct loop ends with MFMA rather than a copy. To keep a final accumulator
readback hazard from contaminating the experiment, the generated asm places
four `s_nop 15` instructions **after both ending timestamps and their LGKM
completion waits**, before returning the accumulator to compiler-generated
output extraction. These are conservative scalar wait states, not a newly
claimed exact core-cycle calibration. They are common to freshly built R2 and
copied-R1 controls. They never separate a load/wait from its MFMA consumer.

The earlier copied-R1 varying-data failure persists under this postamble.
No repair or delay was inserted into that failing loop. No pipeline startup
or load->MFMA aging stage is introduced into R0.

## Verification coverage

The same [exact integer/dyadic reference](VERIFICATION.md) checks every one
of 4096 outputs and 32 guard words, outside timing:

- **495 direct-raw verification cases**: legacy input plus record/lane/half-
  varying hash data, baseline, A-only/B-only/both seed changes; depths1/2/3/4/8,
  all load shapes and modes as selected in the manifest; stride4 LDS bank
  controls; D3/U8 and D16/U1 including short partial-phase exits/odd windows.
- **45 positive R2 verification controls**, all five modes/three shapes at
  D1/D2/D8, using varying hash data.
- **684 warmed checked timing launches**, all correct.
- **3 copied-R1 negative controls**, all correctly detected as mismatches;
  guards stay intact. They are excluded from the 1224 positive total.

Stream pitches/allocation are explicitly matched across controls. No output
checks are disabled to obtain timings. The old verified and failing binaries
are preserved; new R2/R1 controls use `-direct-control` output directories.

Twelve builds / 360 actual code entries are inspected: direct R0 has zero
body copies, exactly N loads and one MFMA/step, its MFMA reads the proper raw
slot immediately after the wait, no body NOPs, no scratch/lane spill and the
correct counter thresholds. Symbolic completed-read execution independently
checks operand record/word progression for all shapes and cyclic/stream modes.
These inspections are not a replacement for the GPU numerical tests above.

## Timings

Native HIP O3/C++20, gfx90a, HIP0 / PCI31:00.0, CPU+memory NUMA0, one CTA /
256 threads/four waves, K32768/U32, no model/context or clock/power changes.
Each selected configuration warmed200 ms; two randomized process-order runs
of nine checked samples each. Absolute-wave CTA makespan, not a wave median.
Hash data, waits enabled; initialization/checking outside timing. LDS raw
priming remains after start-clock waits and inside timing.

**Copy-free R0, cycles/step, run1 / run2:**

| D / mode | b128 (or global dwordx4) | 2xb64 (or 2xdwordx2) | 4xb32 (or 4xdword) |
|---|---:|---:|---:|
| D1 hot LDS | 69.555 / 69.542 | 70.013 / 69.933 | 82.683 / 82.721 |
| D1 sweep LDS | 82.765 / 82.806 | 83.607 / 83.625 | 96.159 / 95.840 |
| D2 hot LDS | **64.560 / 64.598** | **64.982 / 65.084** | 65.686 / 65.692 |
| D2 sweep LDS | **66.036 / 66.077** | **66.485 / 66.604** | 67.757 / 67.736 |
| D4 hot LDS | 64.698 / 64.686 | 65.064 / 64.968 | 65.881 / 65.847 |
| D4 sweep LDS | 66.105 / 66.178 | 66.695 / 66.524 | 67.738 / 67.901 |
| D8 hot global | **64.309 / 64.310** | **128.532 / 128.416** | **85.390 / 85.354** |

Fresh copied-R2 controls:

| Case | R2 cycles/step, run1 / run2 |
|---|---:|
| D1 hot LDS b128 | 90.418 / 90.376 |
| D1 sweep LDS b128 | 99.431 / 98.778 |
| D2 hot LDS b128 | 65.124 / 64.933 |
| D2 sweep LDS b128 | 66.784 / 66.720 |
| D8 hot global dwordx4 | 65.016 / 65.296 |
| D8 hot global 2xdwordx2 | 129.027 / 129.381 |
| D8 hot global 4xdword | 98.935 / 98.927 |

Stride4 bank-concentrated R0 D4 hot remains ~128.3–128.7 for b128/b64. Copy
removal does not remove bank throughput limits. Shallow R0 queues are correct
but retain visible load latency; D2 is near the MFMA floor with only **two
total operand slots**, instead of raw2+ready2 in R2. The global four-scalar
case improves substantially, while the two-wide-load bottleneck remains.

These are whole-schedule comparisons: instruction order, lookahead and
register footprint change along with copy removal. Do not interpret every
cycle saved as an isolated MOV latency, identify a physical limiting port or
claim optimality for all streaming configurations. Default build R stays2
for compatibility; R0 is explicitly selected.

## Reproduce

```sh
# D U output LDS-stride ready-bank-count. R=0 selects the copy-free path.
bash research/fp16-load-matmul/build.sh 2 32 \
  temp/fp16-load-matmul/d2-u32-lds1-verified-r0 1 0
python3 research/fp16-load-matmul/inspect_loops.py \
  temp/fp16-load-matmul/d2-u32-lds1-verified-r0
numactl --cpunodebind=0 --membind=0 \
  temp/fp16-load-matmul/d2-u32-lds1-verified-r0/probe \
  --mode lds-sweep --loads 1 --wait 1 --pattern hash --rounds 9

python3 research/fp16-load-matmul/direct_raw.py
```

The runner expects the twelve builds named in its manifest; use `--resume`
only for an unchanged experiment. `results/direct-raw/` holds all cases,
commands, verification errors, hashes/inventories/symbolic sequences and
process snapshots. The R1 failures are expected negative controls; any positive
failure produces a nonzero suite result. Source before this extension is in
`temp/fp16-load-matmul/pre-direct-raw/`. No production changes.
