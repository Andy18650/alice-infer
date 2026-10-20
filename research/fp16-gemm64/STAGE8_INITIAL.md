# Archived initial StageK=8, two/four/eight LDS stages

**Superseded by [the persistent two-register-buffer loop](STAGE8.md).**
Commands below describe the archived D1 implementation, not current source.
The measured source is frozen in `results/stage8/source/`, and its binary is
retained at `temp/fp16-gemm64/d1-u1-t8-direct-only/`.

User-requested deeper LDS pipeline, direct-to-LDS only, unchanged 2x2 wave
sharing and row-major mathematical contract. Each chunk is one K8 record:
**2048 unique packed FP16 bytes** (A0,A1,B0,B1). Two global_load_dword ... lds
instructions per producer wave; two waves load A and two B, all four compute.
No FP8, dtype conversions, register-staging producer or ready-bank copies.

## Pipeline

Stages denotes storage slots, not raw-register depth. Prime min(stages,tiles)
slots. Each producer wave issues exactly two VMEM instructions per slot.

```text
prime slots0..N-1
vmcnt(2*(N-1)); CTA barrier       // oldest tile ready; future copies may remain
repeat:
    read current A/B from LDS; lgkmcnt(0); MFMA
    vmcnt(2*(N-2)); CTA barrier   // next ready AND current reads retired
    refill retired current slot with tile current+N
```

Tail waits decrease to the exact remaining-stage count; no out-of-range
lookahead is issued. When fewer than N tiles exist, only real tiles are
primed. K0 skips the producer/compute loop. This uses same-class ordered
VMEM completion counts plus a whole-CTA publication/retirement barrier,
not mbarrier/cp.async. The copy validator additionally checks every word in
all allocated slots with poisoned output guards.

Steady waits are **vmcnt(0), vmcnt(4), vmcnt(12)** for N2/4/8; initial waits
are2/6/14. ISA hot branches select these immediates without a subsequent
full drain before the barrier. Zero waits in the deeper entries are tail
paths, not steady-state waits. Only pointers—not data payload—use producer
VGPRs. The generated T8 producer uses fixed address registers beyond raw
operand slots; compiler address temporaries may reuse dead registers.

**D1/U1**: one raw operand slot and one MFMA per chunk. The previous helper
resets its raw ring each chunk; D2 here would read the same record twice,
not prefetch a different K record. This experiment does not implement a
persistent inter-chunk raw-register pipeline. All staged current LDS reads
are completed before consumption/reuse. Accumulators stay in AGPRs.

## Measurement

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, **CPU+memory NUMA0**, one
CTA/256 threads, real64x64x32768 GEMM; no model/context/clock overrides.
Two randomized process-order repeats, 200 ms selected-kernel warmup and
nine exact-checked samples/process. Medians, run1/run2:

| StageK | LDS slots | LDS total | Core cycles/four-wave MFMA step | Whole body us |
|---:|---:|---:|---:|---:|
| 8 | 2 | 4 KiB | 708.943 / 712.322 | 1723.040 / 1731.840 |
| 8 | 4 | 8 KiB | 435.474 / 435.458 | 1058.080 / 1058.880 |
| 8 | 8 | 16 KiB | 433.316 / 433.303 | 1052.160 / 1052.640 |
| 128 control | 2 | 64 KiB | 143.454 / 143.200 | 347.840 / 347.200 |

The control is the current larger-chunk **D2/U4** implementation, not an
otherwise instruction-identical stage-count comparison. Among the three
K8 cases only ring depth changes. D1/U1 pure-LDS diagnostic loops measure
108.167/108.178 hot and116.161/116.168 sweeping cycles/step, rather than the
older D2/unrolled ~64-cycle reference.

The times cover the entire K32768 body, including initial staging, all4096
K8 steps, waits and stage barriers—not one chunk. The cycle divisor is4096
steps/wave, not the aggregate16384 hardware MFMAs. Final accumulator guard,
output stores, CPU packing/upload and verification are outside body timing.
The full event intervals are in the raw CSV/summary as secondary measurements.

Four slots substantially improve over two; eight add little. **The small-
chunk version is not a win over the large-chunk control.** It keeps per-K8
producer address/m0 setup, helper reset, counter dispatch and a CTA barrier.
The data do not isolate their individual cost, establish a hardware
bandwidth/latency ceiling, or rule out a redesigned small-chunk pipeline.
No further tuning/counters/production integration are part of this round.

## Verification and provenance

**405 exact checked launches**:297 verification-only cases plus108 timing
launches (including matched control/LDS diagnostics). All4096 C values and32
guards pass; GEMM invocations also run full-word/all-slot staging checks.
K0..65536 includes odd/tails, fills shorter than each ring, fill boundaries,
multiple ring wraps, independent A-only/B-only seed changes and long inputs.
Six actual T8 entries have no scratch/lane spills or inner accumulator moves;
there are three GEMM entries, two LDS diagnostics and one copy validator.
Raw wave clocks independently reproduce each CTA span and cycle/step value.

During preparation, an old regression runner path error was discovered:
`run.py` still named pre-cull directories, so the earlier340-launch claim
was not numerical validation of the pruned binaries. Historical data are
retained and flagged in `results/direct-only/PROVENANCE.md`. Paths now use
`-direct-only` and source/binary hashes are checked before launching. Six
current builds were rebuilt and the **corrected340-launch regression passes**
(`results/direct-only-corrected/`). This is separate from the405 launches
above. Current sweep commands, exact binaries' hashes, generated asm,
resource inventories, source snapshots, clocks and logs are in `results/stage8/`.

```sh
bash research/fp16-gemm64/build.sh 1 1 8
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/d1-u1-t8-direct-only
numactl --cpunodebind=0 --membind=0 temp/fp16-gemm64/d1-u1-t8-direct-only/probe \
  --mode gemm --buffers 8 --k 32768 --rounds 9
# Build matched current control, then run all checks and both timing repeats:
bash research/fp16-gemm64/build.sh 2 4 128
python3 research/fp16-gemm64/inspect_code.py temp/fp16-gemm64/d2-u4-t128-direct-only
python3 research/fp16-gemm64/run_stage8.py
```

Prior measured binaries are preserved; the pre-Stage8 pruned builds were
copied to `temp/fp16-gemm64/pre-stage8/` before refreshing the current paths.
