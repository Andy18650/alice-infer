# Pair global-copy addressing across two K8 stages

User-requested optimization: retain the source pointer across two records,
use0/256 then2048/2304 instruction displacements, and advance the64-bit
pointer by4096 only after the odd record. Signed13-bit displacement is
-4096..4095; both positive odd-stage offsets fit.

## Address correctness

A global_load_dword ... lds displacement affects BOTH the global source and
LDS destination. Therefore, for physical slot s:

```text
displacement = (s % 2) * 2048
m0 = wave_LDS_base + floor(s/2)*4096
first load:  source pair_ptr + displacement
             destination m0 + displacement + lane*4
second load: same, plus256
```

Effective LDS destination remains wave_base+s*2048+lane*4. M0 is still set
before the barrier and restored after every record; no separate M0-state
optimization is mixed in. On even records, no pointer update occurs. On odd
records, v_add_co_u32/v_addc_co_u32 advance4096. The same scheme covers bulk,
startup and bounded tails. All ring sizes are even, so refill parity agrees
with global record parity and bulk/tail transitions preserve pair alignment.

In32-slot startup, record30 leaves pair_ptr at30; the delayed record31 uses
+2048/+2304, then advances to32. An odd final record count can leave one
unconsumed half-pair without accessing it. No data, copy-count, wait/barrier,
raw-buffer, MFMA order, LDS allocation or payload datatype changes occur.

## Matched result:16 improves, other depths are mixed

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, one CTA /
256 threads, actual64x64x32768 GEMM; no model/context/clock changes. Two
randomized process-order repetitions, nine checked samples/process after
200 ms selected-kernel warmup. Medians, run1/run2:

| LDS slots | Advance every stage: cycles/step | Paired advance: cycles/step | Paired whole body us |
|---:|---:|---:|---:|
| 8 | 145.805 / 153.218 | 153.557 / 154.193 | 373.600 / 375.360 |
| 16 | 96.800 / 96.611 | **93.510 / 92.646** | **227.680 / 225.760** |
| 32 | 97.273 / 97.943 | 97.312 / 95.125 | 236.960 / 231.520 |

StageK128/two-buffer control:143.278/143.309 cycles/step. Pure-LDS hot:
64.178/64.191.16 improves in both repetitions;32 is effectively flat in the
first and improves in the second.8 is slower in both, with notable baseline
variation. No universal improvement or unique physical bottleneck is claimed.
Results and assembly are reported, with no further tuning in this round.

One step is one MFMA in each of four waves; core CTA span divides by4096,
not aggregate16384 MFMAs. Body excludes CPU packing/H2D upload, verification
and final accumulator-harvest/output epilogue. Full event timing is retained.

## Verification and native ISA

**882 checked launches pass:**720 verification cases over2/4/8/16/32 plus
162 timing/control samples. All4096 C outputs and32 guards pass. Cases
cover empty/odd/short fills,30/31/32-slot startup, pair endings, ring/bulk/tail
boundaries, multiple wraps, independent A-only/B-only changes and K<=65536.
Separate all-slot word checks remain; that unchanged copy-check kernel tests
the basic transfer, while the independent full-C oracle exercises the new
paired pipeline. Twenty additional targeted odd-fill/tail smoke launches pass.

Actual-code-object inspection asserts:
- Two global DWORD loads/record, alternating offsets0/256 and2048/2304.
- Compensated M0 bases0,0,4096,4096,...; effective destination equals each
  intended slot, not its successor.
- Only N/2 source adds and N/2 carry adds per N-step bulk, all advancing4096.
- Pre-barrier M0 setup, zero bulk NOPs, same post-MFMA LDS reads, VM24/56,
  two raw buffers, fixed offsets/parity and one group backedge.
- No scratch/lane/accumulator spills, dtype conversion or register producer.

For N>=4, average bulk instructions drop12->11 per wave/step. Startup/tails also use
paired addressing, retaining their existing M0-gap NOPs. No changes to the
larger-chunk producer or byte-check helper are made.

Measured assembly/source/native hashes/raw CSVs/clocks/commands:
`results/paired-global/`. Baseline is the preserved deb88e1 program at
`temp/fp16-gemm64/m0-pre-barrier/probe` with its frozen source. Candidate
full ISA: `temp/fp16-gemm64/paired-global/probe.isa`; committed timed extracts
are `results/paired-global/builds/paired-global/core-{5,6,7}.txt` for8/16/32.
All older binaries/results are retained. No production/CMake/model or system
configuration changes.

```sh
bash research/fp16-gemm64/build.sh 2 1 8 temp/fp16-gemm64/pair-replay
bash research/fp16-gemm64/build.sh 2 4 128 temp/fp16-gemm64/pair-control-replay
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/pair-replay
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/pair-control-replay
python3 research/fp16-gemm64/run_stage8.py --stages 16 32 --verify-stages 2 4 8 16 32 \
  --build temp/fp16-gemm64/pair-replay \
  --control temp/fp16-gemm64/pair-control-replay \
  --baseline-build temp/fp16-gemm64/m0-pre-barrier \
  --baseline-source research/fp16-gemm64/results/m0-pre-barrier/source \
  --baseline-protocol advance-each-stage --protocol paired-global \
  --output research/fp16-gemm64/results/pair-replay
```
