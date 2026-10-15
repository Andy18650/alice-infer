# Matched affine / B-panel / AB-panel comparison

## What the result supports

For the working CK V3 256x256x64 kernel, selecting the better of the two
measured CTA traversal orders separately for each storage variant gives
**AB-panel > affine > B-panel at all nine confirmed problems**. This is an
empirical ranking of these compiled implementations, not a monotonic rule
about tiling. The earlier README named the controls but did not display
a comparison table; the supporting rows are in
[`results/working-confirmation.csv`](results/working-confirmation.csv).

Native O3 HIP/C++20, one gfx90a GCD (HIP 0), CPU+memory NUMA node 0,
no model/attention context, BF16 A/B with FP32 accumulation/output. Two
independent nine-round runs, 18 pooled samples per row, warm graph batches
of 16 launches. Packing/initialization is excluded; each operation is one
full GEMM kernel, no split-K. Other eager/isolated protocols remain separate
in the CSV. Shapes/geometry are held fixed within each comparison.

- `affine`: affine A and affine B, physical `[row][K]` for both operands.
- `b-panel`: affine A; B is `[K_panel][N_tile][N_inside][K_inside=64]`.
- `ab-panel`: the same B packing; A is
  `[K_panel][M_tile][M_inside][K_inside=64]`.
- `n-fast`: successive CTA indices advance N first; `m-fast` advances M first.
- These compute-sized panels are **not** the fixed shared 32x128 B tiles
  in the earlier GEMV/prefill study.

## Better measured order per storage (TFLOP/s)

Numbers are pooled medians from the CSV; selection here is between two
observed order medians, not an additional independent confirmation of that
selection. No order is silently discarded in the source data.

| M x N x K | Affine A/B | Affine A, panel B | Panel A/B |
|---|---:|---:|---:|
| 2560 x 2560 x 8192 | 133.09 | 125.46 | 134.83 |
| 2048 x 3328 x 8192 | 137.50 | 129.75 | 139.32 |
| 2048 x 3328 x 32768 | 142.66 | 133.89 | 143.86 |
| 4096 x 3328 x 8192 | 138.41 | 130.44 | 140.32 |
| 4096 x 3328 x 32768 | 142.83 | 134.14 | 144.10 |
| 4096 x 6656 x 4096 | 133.59 | 126.97 | 136.09 |
| 4096 x 6656 x 8192 | 138.54 | 131.12 | 140.78 |
| 4096 x 6656 x 16384 | 141.36 | 133.25 | 143.06 |
| 4096 x 6656 x 32768 | 142.90 | 134.56 | 144.32 |

## Fixed-order controls and exception

At M=4096,N=6656,K=8192, holding **n-fast** fixed gives:
**137.88 / 131.12 / 140.78 TFLOP/s** for affine / B-panel / AB-panel.
Thus the ordering is not merely a consequence of choosing different orders.

However, it is **not universal at fixed order**. At M=4096,N=6656,K=32768,
n-fast gives **130.64 / 134.56 / 144.32**: B-panel beats affine.
Changing only affine to m-fast restores **142.90**. The affine n-fast loss
repeats separately in both processes (~130.67 and ~130.60), rather than
being a lone noisy sample. Retain this order sensitivity; do not summarize
the data as every affine row beating every B-panel row.

## Important code-generation confounder

| Storage | Reported registers | Private bytes/thread | LDS | CTAs/CU |
|---|---:|---:|---:|---:|
| affine | 512 | 0 | 64 KiB | 1 |
| b-panel | 512 | **16** | 64 KiB | 1 |
| ab-panel | 508 | 0 | 64 KiB | 1 |

The same resource values appear throughout the confirmation CSV. ELF notes
in the measured wide-V3 code object independently confirm private segments
0/16/0. In its B-panel entry (base 0x5400), scratch-descriptor buffer stores
appear in the prologue; a private `buffer_load_dword` at **0x65c4** lies
inside the **0x65a4..0x6df4** K-loop backedge. The absence of a `scratch_`
mnemonic would not reveal this spill. These are reported register fields,
not counts to sum blindly with the separate AGPR metadata.

So changing only B does not simply improve contiguous reads while leaving
everything else equal: it also changes descriptor/address code generation
and crosses a private-storage threshold in this implementation. Packing
both operands produces a different, spill-free kernel. Joint address/locality
and CTA-order effects can also matter. No matched spill-free B-only control
or per-layout clock audit isolates how much of the loss comes from each
factor. In particular, do not claim that the spill alone explains the loss
or that asymmetric tiling is inherently slow.

The measured ordering is real, but a general layout principle would require
additional controls. No production activation repack or layout is adopted.
