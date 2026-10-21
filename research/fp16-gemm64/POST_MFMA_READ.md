# Candidate: issue the recycled-buffer LDS read after MFMA

The later [M0/barrier scheduling probe](M0_BARRIER.md) moves setup across
the barrier/removes the bulk NOP, retaining this algorithm and measured data.

User-directed test of the original earlier-publication idea only. No extra
just-in-time publication barrier/alternative route was implemented or tried.
The mixed result includes a clear8-slot regression; performance work stops
here for the user's decision. The candidate is not a universally better
replacement across ring sizes. Prior baseline source/binaries remain intact.

## Schedule, N>=4

Startup publishes tiles0/1/2 (or all real tiles for a short fill) and primes
raw0/raw1 before entering the bulk. The same existing startup barriers are
used, with a tighter VM watermark. One bulk step now is:

```text
lgkmcnt(2)                        // current ready, next may remain pending
MFMA tile n from raw[n%2]
DS-read tile n+2 -> raw[n%2]       // immediately refill just-consumed A/B regs
vmcnt(2*(N-4))                    // publish n+3 for the following unit
CTA barrier                      // retire current n and publish n+3
copy global tile n+N -> LDS slot n%N
```

The read is of n+2, not n+1 into the opposite buffer. Its global writes were
published at the preceding boundary (or startup), so it is safe to read LDS
immediately after MFMA. Two raw buffers, literal slot offsets, one barrier
per step and the same unrolled bulk count/backedge remain. One more global
lookahead stage is sacrificed versus the previous current-retirement code.
16/32 steady VM counts26/58 become24/56. Source/data layout, M0 handling,
copy instructions, conservative remaining>2*N bulk guard and timing brackets
are unchanged. No payload VALU copy or padding after MFMA is added.

The final two tail records do not issue out-of-range future reads. The last
record still requires lgkmcnt(0). N2 retains its already-verified pre-MFMA
read/two-barrier special case: n+2 aliases current's LDS slot, so this hoist
is not applied there. No alternative route is added for larger rings.

## Result: mixed, including the requested regression report

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, **CPU+memory NUMA0**, one CTA /
256 threads,64x64x32768 GEMM, no model/context/clock changes. Two randomized
process-order repeats, nine checked samples per process after200 ms selected-
kernel warmup. Whole-body core cycles/four-wave MFMA step, run1/run2:

| LDS slots | Preserved pre-MFMA-read baseline | Candidate post-MFMA read | Candidate whole body us |
|---:|---:|---:|---:|
| 8 | 137.669 / 137.879 | **153.702 / 152.277 (regression)** | 373.920 / 370.560 |
| 16 | 109.625 / 109.694 | **99.271 / 99.221** | 241.600 / 241.440 |
| 32 | 110.772 / 110.878 | 100.663 / 100.463 | 244.960 / 244.640 |

Matched StageK128/two-buffer control:143.302/143.235 cycles,347.520/347.200 us.
Thus16/32 improve, but8 worsens in both repetitions. The result is consistent
with a trade of global-copy lookahead for earlier LDS reads, but individual
wait costs/hardware causes are not isolated. No fallback, per-size selection,
extra-barrier route, M0 optimization, padding or follow-up tuning was tried.

Times cover all4096 K8 steps/wave; the divisor is not aggregate16384 MFMAs.
Input construction/packing/H2D, verification and accumulator-harvest/output
epilogue are outside body timing. Full event measurements remain in CSV.

## Correctness and actual assembly

**882 exact checked launches pass**:720 verification cases across2/4/8/16/32
and162 timed samples including matched baseline/control/diagnostic samples.
All4096 C values and32 guards pass, along with all allocated LDS words/guards
in the separate staging check. Coverage includes K0..65536, independent
A-only/B-only changes, short fills, split32-slot priming, bulk/tail boundaries
and multiple ring wraps. No command timeout/recovery was needed in this run.

ISA inspection verifies LGKM2 immediately before each bulk MFMA, two same-
buffer ds_read_b64 immediately AFTER it, fixed n+2 slot offsets, VM24/56,
current-slot refills, one loop backedge, and no extra wait/NOP between MFMA
and those reads. No scratch/lane/accumulator spills, dtype conversions or
register-staging producer. This validates the tested MFMA->DS-read->later
writeback sequence, not arbitrary immediate VALU overwrites of MFMA sources.

Assembly in the committed measured-code-object extracts:
- 16 slots: `results/post-mfma-read/builds/post-mfma-read/core-6.txt`, bulk
  starts at line288 / GPU address0x7054.
- 8 slots (regression): corresponding `core-5.txt`, bulk starts around line152.
- 32 slots: corresponding `core-7.txt`.
Full native disassembly: `temp/fp16-gemm64/post-mfma-read/probe.isa`.
Baseline program: `temp/fp16-gemm64/retire-current/probe`, checkpoint63d14a0,
verified against its preserved source. Old/new binaries and all raw data are
kept separately. Current-study source/hash/command snapshots: `results/post-mfma-read/`.

Reproduction uses explicit binaries/frozen source and a fresh output:

```sh
bash research/fp16-gemm64/build.sh 2 1 8 temp/fp16-gemm64/post-replay
bash research/fp16-gemm64/build.sh 2 4 128 temp/fp16-gemm64/post-control-replay
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/post-replay
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/post-control-replay
python3 research/fp16-gemm64/run_stage8.py --stages 16 32 --verify-stages 2 4 8 16 32 \
  --build temp/fp16-gemm64/post-replay \
  --control temp/fp16-gemm64/post-control-replay \
  --baseline-build temp/fp16-gemm64/retire-current \
  --baseline-source research/fp16-gemm64/results/retire-current/source \
  --baseline-protocol pre-mfma-read --protocol post-mfma-read \
  --output research/fp16-gemm64/results/post-replay
```

No production/CMake/model or system/clock-policy changes.
