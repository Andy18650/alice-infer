# First culling step: remove wait-free execution

Progress before cleanup is committed at **3986b13**. The checkpoint includes
the FP8 diagnostics, configurable FP16 framework, raw results, exact
verification, copied-R1 failures and working copy-free raw rings. Unrelated
memory-latency/GEMM/smaller-agent files were not included.

## Active FP16 changes

- Delete the wait boolean from kernel/pipeline templates and dispatch.
- Generate **15 mandatory-wait entries**, not 30 waited/unwaited entries.
  Every loop retains its appropriate vmcnt/lgkmcnt threshold and final drain.
- Keep `--wait 1` only as a legacy safe-command spelling. `--wait 0` or any
  other value errors before explicit HIP initialization, even with --verify 0.
- Restrict original-module launches to the selected mode/load's old waited
  symbol (`...ELb1EE...`); old wait-free symbols are rejected before loading.
  Safe original-code-object verification remains supported.
- Matrix runner enumerates only waited cases and no longer tolerates an
  unsafe mathematical failure. Remove unsafe cases from `verify_cases.py`.
- Update instruction/FIFO/symbolic inspectors for the two-parameter kernel
  template and 15-entry set, with no conditional safety-check bypass.

The `wait` CSV field remains fixed at1 for format compatibility. Historical
CSV analysis can still read old wait0 records; that is not an execution path.
The `--verify 0` opt-out is independent and does not remove any GPU waits.
R1 copied-ready remains an explicit negative diagnostic; it was not part of
this first cull. R0/R2, modes, layouts, depths and unrolls are unchanged.

## Checks after culling

Seven builds cover R0 D1/D2/D8, D3/U8, D16/U1, bank-stride4, and R2 D2.
All 105 surviving timed instruction/register lists match the checkpoint's
waited counterparts after stripping addresses/branch displacements. There
are no new wait-free kernel symbols in those builds.

- **223 exact full-output cases pass**, including original waited-module
  compatibility, plus **6 current matrix-runner cases**.
- **8 rejection checks pass**: seven --wait0/--verify0 attempts and one old
  wait-free module-symbol attempt. All return argument error without a CSV.
- Every current generated entry has the expected partial waits, no scratch/
  lane spills, correct raw input mapping and checked FIFO register lifetimes.

These are correctness/structural regressions, not a new variance-controlled
performance campaign. No production code, model fixtures or CMake changed.

```sh
bash research/fp16-load-matmul/build.sh 2 32 \
  temp/fp16-load-matmul/d2-u32-lds1-verified-r0-waited 1 0
python3 research/fp16-load-matmul/inspect_loops.py \
  temp/fp16-load-matmul/d2-u32-lds1-verified-r0-waited
python3 research/fp16-load-matmul/check_waited.py
```

The checker expects the seven builds listed in its source. Results, raw logs,
commands, rejection messages and instruction comparisons are under
`results/waited-only/`. Current build names append **-waited**, preserving
prior binaries. Historical FP8 diagnostic sources and existing raw datasets
are deliberately not deleted in this first active-FP16 cleanup; the checkpoint
is the recovery point for removed FP16 wait-free generation.
