# StageK16: preserve64x64 output tile, group two K8 fragments

User-directed experiment. Primary case preserves the **32 KiB LDS budget**:
StageK16 x8 slots versus the prior StageK8 x16 slots. Also test the explicitly
authorized16-slot/64 KiB case. No M/N growth, b128 reads, data-layout change,
extra raw buffers or unrelated scheduling optimization is introduced.
The preserved StageK8 program remains at `temp/fp16-gemm64/active/` with its
frozen source in `results/active-cull/source/` (checkpoint0528d65).

## Work and resource accounting

| Configuration | K8 records/slot | LDS slots | LDS total | Resident K elements |
|---|---:|---:|---:|---:|
| Baseline StageK8 | 1 | 16 | 32 KiB | 128 |
| Primary StageK16 | 2 | 8 | 32 KiB | 128 |
| Additional StageK16 | 2 | 16 | 64 KiB | 256 |

Each K16 stage contains two adjacent original `[A0,A1,B0,B1]` K8 records.
Global copy: one M0 setup, four DWORD instructions at offsets0/256/2048/2304,
one M0 restore and one4096-byte source-pointer add/carry pair. Both source
and destination offsets remain valid and no operand conversion is added.
The independent all-slot byte checker uses the same two-record copy width.

Each stage computes two MFMAs/wave with two raw buffers:

```text
lgkmcnt(2); MFMA current fragment0; read next-stage fragment0 -> raw0
lgkmcnt(2); MFMA current fragment1; read next-stage fragment1 -> raw1
VM wait; set M0; CTA barrier; copy future K16 stage; restore M0; advance4096
```

Both A/B reads for each fragment remain ds_read_b64. There is no barrier
between these two MFMAs. LDS bytes read and global bytes copied per K are
unchanged; rendezvous, VM-wait instruction and M0 setup/restore frequency
halve. Source-pointer frequency was already once per two fragments in the
paired K8 baseline and does not decrease further.

Startup publishes current/next whole stages, primes raw0/raw1 from the two
fragments of stage0 and leaves their completion to partial LGKM waits.
Steady stage n publishes stage n+2 for the following iteration; four VMEM
instructions/stage give VM20/52 for8/16 slots. The16-slot fill primes15 stages
(60 instructions), retires the first at VM56, then issues the last stage;
no64-pending-instruction window is required. Tail reads next-stage fragments
only if a next stage exists; the last stage's second MFMA waits LGKM0.
K0 skips the loop; odd K zero-pads to16, reference uses actual K.

The outer loop unrolls8/16 stages, i.e.16/32 MFMAs. It keeps the conservative
remaining>2*N bound. The32 KiB case therefore has the same16-MFMA bulk
unroll and32-MFMA bounded tail at K32768 as the prior K8 winner.

## Matched results: no clear equal-budget gain

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, one CTA /
256 threads, real64x64x32768 GEMM, no model/context/clock overrides. Two
counterbalanced process-order repetitions, nine checked samples/process
after200 ms selected-kernel warmup. Medians, run1/run2:

| Experiment | Candidate cycles/K8-equivalent step | StageK8 baseline cycles/step | Candidate entire body us |
|---|---:|---:|---:|
| K16 x8 slots,32 KiB | 92.821 / 94.006 | 93.452 / 92.976 | 226.080 / 228.960 |
| K16 x16 slots,64 KiB | 92.241 / 91.959 | 93.271 / 93.276 | 224.640 / 223.840 |

The equal-budget result changes direction between repeats: **no clear win**.
The64 KiB variant improves slightly in both, at twice the LDS. This is not
a prefill/grid-level recommendation; increased shared memory can restrict
CTA residency, and no multi-CTA/real-model result is measured here. We do
not ascribe the neutral result to a particular hardware instruction or
implicitly assume halving barrier count halves elapsed barrier cost.

Cycle divisor remains4096 executed MFMAs/wave for K32768, not2048 K16 stages
or aggregate16384 hardware instructions. Body includes startup/compute/
staging/waits/barriers/tail; excludes CPU packing/H2D, verification and final
harvest/output. Timing and actual work are unchanged; no NOP ablation is used.

## Validation / native code

Both matched experiments pass **120 checked samples total**:24 focused
verification cases and36 timed/baseline samples per candidate. Eighteen
additional targeted short-fill/tail smoke launches pass. All4096 C values,
32 guards and every allocated LDS word/guard pass, including the64 KiB end.
Checks cover empty/odd/half-stage K, fills/wraps, bulk/tail boundaries, long
K<=65536, independent A/B changes and last-fragment readiness.

ISA inspection confirms:
- Two MFMAs/four b64 reads per stage, one CTA barrier and one VM watermark.
- One M0 setup/restore per stage; setup/barrier/global-load adjacency, no bulk NOP.
- Four global DWORD loads/stage and one4096-byte add/carry pair.
- Fixed stage/fragment offsets, alternating raw0/raw1 and one group backedge.
- VM20/52,15-stage split fill for the larger ring, no scratch/lane/accumulator
  spills or datatype conversions. Optional pure-LDS helpers retain their old math.

Per-build metadata now records `outer_unroll` in MFMA steps:16/32, not merely
physical slot count. New build choice is limited to8 or16 slots at StageK16;
no obsolete broad matrix is restored. The comparison uses the frozen K8
binary/source with explicitly verified hashes, not a hidden active fallback.

```sh
bash research/fp16-gemm64/build.sh temp/fp16-gemm64/stage16-replay8 8
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/stage16-replay8
python3 research/fp16-gemm64/run.py --build temp/fp16-gemm64/stage16-replay8 \
  --output research/fp16-gemm64/results/stage16-replay8 \
  --baseline-build temp/fp16-gemm64/active \
  --baseline-source research/fp16-gemm64/results/active-cull/source
# Repeat with a fresh output/build and slot argument16 for64 KiB.
```

Data/native extracts/source snapshots: `results/stage16-s8/` and
`results/stage16-s16/`. Distinct binaries: `temp/fp16-gemm64/stage16-s{8,16}/`.
All prior measured artifacts are preserved. No b128/data.hpp, production,
CMake, model-format, system or clock-policy changes; no further tuning.
