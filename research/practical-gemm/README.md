# Practical-shaped GEMM: row-major A, packed FP8 B

First stage of the user-directed practical-interface study. **FP16 A is ordinary
row-major `[M,K]`, stride exactly K, with no M/K padding.** B retains the existing
finite E4M3FN packing and logical FP32 K128/N128 scale table. No B simplification,
smaller compute tile, split-K or schedule sweep yet. Standalone HIP O3/C++20
on gfx90a; no BLAS/CK, including references or activation packing.

**Result:** the row-major interface is correct, but this first loader is not
performance-ready. Its GEMM is slower on every measured shape. Removing a
plain GPU activation-pack pass sometimes pays for that loss, but not at small
M or on the large regression case. Keep the packed kernel as the performance
reference; these results do not establish an intrinsic row-major disadvantage.

## Isolated implementation

Retain the [native design](../native-gemm/DESIGN.md): 128x128 output, K32,
four wave64s, 16 KiB single LDS buffer, five raw operand slots, three scaled/raw
banks, balanced baseline MFMA order, late scale loads and packed FP32 FMAs.
The new packed baseline's **complete native_gemm instruction bytes are identical**
to the historical balanced kernel, not merely source-equivalent; proof is in
`results/rowmajor-a/checks/packed-baseline-identity.json`.

Only A production changes. For producer wave `w`, lane `l`, packet `p=0/1`:

```text
row = 128*tile_m + 32*w + floor(l/4) + 16*p
k   = 32*stage + 8*(l%4) + h                 // h=0..7
address = 2*(row*K + k)
```

Four adjacent lanes load four contiguous 16-byte K8 vectors: 64 bytes of one
row per stage. Each wave produces 32 rows, rather than one K8 slice across128
rows. It scatters each pair of packets through **four LDS b64 stores**, replacing
two b128 stores, to reconstruct the original A consumer layout exactly. B
production, matrix/scale operands and order, consumer LDS reads and fences are
unchanged. Local access coalescing does **not** certify whole-GCD channel balance,
cache behavior or efficient LDS bank access.

Reuse v43 for transient A addresses and v52 before B permutation copies; restore
B's producer address before its stores. No extra persistent VGPRs. M-tail loads
are EXEC-masked, inactive payloads zeroed, and EXEC restored before barriers.
A wave-uniform full-32-row path skips masking/zeroing. **Initial support is
K multiple-of32 or zero**, explicitly rejected otherwise; there is no hidden
padded row stride. N/M tails remain supported.

Both kernels: 256 VGPR/0 AGPR, private0, two CTAs/CU. SGPRs rise51->59.
The added address/scatter work also lengthens the first q0 MFMA gap: the old
13/15/15 balance is not preserved. The compute instruction order alone being
unchanged does not preserve its issue timing. The row-major LDS address pattern
and long row strides change locality too; no unique bottleneck has been proved.

## Inputs, protocols and shape coverage

- 8B-sized projections: QKV `(N,K)=(6144,4096)`, fused gate/up `(28672,4096)`,
  down `(4096,14336)` at M=1/16/128/512/2048; extra QKV M33/129 tails.
- Retain104/208/312 CTAs at K8192/32768 and1664-CTA large-grid regression case.
- Synthetic original logical operands/scales, **not real model weights** or
  model/context runs. M is token count, not attention context length. M1 is
  diagnostic; the unchanged128-row tile is not a proposed decode replacement.
- **Kernel:** GPU-pack A once outside timing, then time GEMM alone. Row-major
  uses the original device A directly.
- **Activation:** start from device-resident row-major FP16 A. Packed variant
  times a plain GPU pack plus GEMM, row-major times one GEMM. Capture component
  times too; total includes the midpoint event used to separate pack/GEMM.
- The packer is deliberately plain, fully checked, **not an optimal packer**.
  Neither protocol includes FP32->FP16 conversion, H2D, CPU checks, weight
  packing or scale upload. A fused conversion/packing path could change the
  comparison. The harness retains both A allocations for validation; reported
  byte sizes are interface sizes, not a measured production memory footprint.

HIP0/BDF0000:31:00.0, CPU+memory NUMA0, automatic clocks, no overrides. Each
variant gets200 ms selected-GEMM warmup and nine checked eager launches/protocol.
Two paired process repeats, same inputs between variants within a repeat,
changed seeds and reversed case/variant order in repeat2. Protocol order alternates
by round within one process; their cache histories are not identical because
packing can perturb subsequent cache/clock state. Useful FLOPs only; padding and
scale work are not useful GEMM FLOPs. No whole-input-hot or cold-HBM certification,
pure instruction-cost attribution, or statistical claim for tiny differences.

## First measurements

Mean of two process medians; microseconds. Positive delta = row-major takes
longer. Full24-shape results: [comparison.md](results/rowmajor-a/comparison.md).

|Case|Packed kernel|Row-major kernel|Kernel delta|Packed activation|Row-major activation|Activation delta|
|---|---:|---:|---:|---:|---:|---:|
|QKV M1|113.44|131.52|+15.94%|125.52|131.60|+4.84%|
|Gate/up M128|307.20|384.80|+25.26%|323.68|383.76|+18.56%|
|Down M2048|1716.96|1883.04|+9.67%|2019.52|1881.04|-6.86%|
|208 CTAs, K32768|1612.40|1892.64|+17.38%|3312.23|1906.80|-42.43%|
|Large grid, K32768|12247.82|18752.05|+53.11%|14364.22|18802.29|+30.90%|

- At M1/16, row-major loses even after charging packing on all three projections.
- Narrow-N, large-M controls favor avoiding this plain packer, but do not prove
  optimal packing loses. Practical-shape wins near1% are not firm conclusions.
- The large case falls from about145.9 to95.3 useful TFLOP/s. Row-major clocks
  are higher, not lower; clock droop does not explain that loss. The312-CTA
  K32768 row-major medians vary3654.6/4014.9 us across repeats; preserve that
  variability rather than presenting its mean as a stable precise result.
- A coalesced interface alone is insufficient. Next candidates are cheaper/
  better-mapped A LDS staging and rebalanced issue windows, then small-M compute
  geometry. Do not change all three together or simplify B in the same comparison.

## Validation and artifacts

**1964 checked GEMM launches /214 executable-hashed processes:**236 boundary/
seed/scale/finite-code launches and1728 timing launches across24 shapes.
K32 stage counts0..29, long K65536, M/N tails, all four scale patterns and
finite FP8 cases. Every process fully checks GPU-packed A words against the
independent CPU logical packer; every GEMM checks logical references and all C
finite after NaN poison plus guards. Small references full, large sampled;
this is not arbitrary-IEEE or model-level numerical certification.

`checks/summary.json`:1,387,008 coordinate/masking cases, complete A LDS
coverage, unchanged generated/native non-A projections and resource/opcode
gates. Row-major static ISA contains both full/tail branches:40 global x4
loads in its eight-stage loop, but only24 execute on the full-row path (same
payload count as packed). Do not mistake static alternatives for extra traffic.
1964 raw clock captures independently rechecked using the recorded HIP wall
clock rate and event-span containment;1964 duplicated-low-word negatives rejected.
Unsupported K and nonempty build paths are rejected without overwriting binaries.
These are bounded checks, not a formal ISA/memory-return emulator. Derived
summaries and the coordinate/ISA checks reproduce exactly. HIP builds are
warning-free; clangd main-file check passes. Header check has no compiler/AST
errors but reports three interactive DefineOutline failures (single-TU kernel
header); these are recorded separately, not hidden as a clean tool exit.

```sh
bash research/practical-gemm/build.sh packed <fresh-packed>
bash research/practical-gemm/build.sh rowmajor <fresh-rowmajor>
python3 research/practical-gemm/run.py --packed <fresh-packed> --rowmajor <fresh-rowmajor> --output <fresh-results>
python3 research/practical-gemm/verify.py <results>
# Read-only result check above; analysis rewrites derived reports, so copy first:
python3 research/practical-gemm/run.py --analyze-only --output <copied-results>
```

The recorded run uses `temp/practical-gemm/rowmajor-a-v2/{packed,rowmajor}`.
V1 pilot binaries and compilation sources remain intact; v2 adds host wall-rate
logging for independent capture checks. Compilation inputs and as-run drivers
are mirrored under each result build's `source/research/` tree, preserving relative
includes/imports. Rebuild historical code with its snapshot build.sh, never the
current source merely to satisfy an old fingerprint. ISA, metadata, commands,
raw clocks/CSVs, provenance and SHA256 manifest live under `results/rowmajor-a/`;
measured binaries/code objects remain in durable gitignored `temp/`.
No production/CMake/model/session/system/package/clock-policy changes.
