# Archived register-staging comparison

The complete pre-cull study is preserved at checkpoint **3217640**, including
source and 620 exact checked launches. Original raw data remain under
`results/final/`; old binaries remain at `temp/fp16-gemm64/dD-uU-tT/`.
The register producer is no longer part of current source/build/CLI.

It was a bring-up and comparative control, not a demonstrated winning option.
It lost to direct-to-LDS in every tested StageK/buffer configuration. The
register control batched vector loads, waited, and wrote LDS before compute;
its double-buffer mode did not overlap return latency with computation.
Thus the results do not establish that every hypothetical asynchronous
register-prefetch design would lose, nor identify a physical bottleneck.
They are sufficient to retire this implementation from the active matrix.
The first post-cull runner's340 launches used archived default-direct
executables by mistake; see `results/direct-only/PROVENANCE.md`. That is
not current-binary validation. The corrected and persistent-era current
regressions each pass340 launches with explicit paths and hash checks.

Native O3 gfx90a, HIP0, CPU+memory NUMA0, one CTA/256 threads, K32768,
D2/U4, no model/context or clock changes. Packed FP16 inputs, packing excluded.
**Whole-body core cycles per four-wave MFMA step**, run1/run2:

| StageK | Reg, one LDS buffer | Reg, two LDS buffers | Direct, one | Direct, two |
|---:|---:|---:|---:|---:|
| 32 | 333.441/334.229 | 342.267/340.271 | 312.198/312.561 | 247.816/247.437 |
| 64 | 245.847/240.451 | 246.861/244.389 | 214.424/214.378 | 147.353/146.988 |
| 128 | 213.188/214.060 | 215.710/216.545 | 187.285/187.642 | 143.351/142.813 |

Kernel body includes staging, computation, waits and stage barriers. It is
not isolated MFMA execution latency. The current direct-only regression checks340 launches, and all24 surviving
larger-chunk benchmark instruction/register lists match the pre-Stage8
pruned builds after addresses/branch displacements are normalized.
