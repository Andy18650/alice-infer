# Superseded trial: invalid packed timestamp words

Do NOT use this trial's raw clocks, body spans, cycle-derived metrics or order
proof as validated evidence. It is retained unchanged for provenance and is
superseded by `../scaled-packed-fixed/`.

The packed timestamp moves omitted `op_sel:[0,1]`. Unlike V_PK_FMA_F32,
V_PK_MOV_B32's default selection copied the low SGPR word to BOTH destination
words. GEMM outputs/reference checks passed, and HIP event timing is a separate
measurement, but stored64-bit timestamps were wrong.

The first expansion auditor also assumed an incorrect default high-word select
for V_PK_MOV, so it falsely accepted these moves. The original dump checker
recomputed ratios from the same corrupted clocks and did not independently
require the instrumented span to fit inside the HIP event interval. Both gaps
are corrected. A dedicated hardware raw-bit gate demonstrates the old failure
and validates explicit selection. A negative regression rejects this trial's
packed-move/packed-FMA clock records with the new event-span containment check.

No source, executable, CSV, raw clock or original proof file is rewritten or
relabelled. The follow-up uses fresh binaries and reruns all568 checks.
