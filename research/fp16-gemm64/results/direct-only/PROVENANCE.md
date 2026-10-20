# Correction: the first post-cull runner used archived executables

The original `run.py` output here reports340 valid checked launches, but its
executable paths still named `temp/fp16-gemm64/dD-u4-tT/`, not the pruned
`dD-u4-tT-direct-only/` directories. See `commands.txt`. Those are the
checkpoint3217640 pre-cull executables, running their default direct producer.
Their numerical passes/timings are real, but are NOT launches of the pruned
binaries. This corrects the original post-cull validation attribution.
The source/actual-ISA comparison of24 surviving timed bodies was a separate
check against the pruned builds and remains valid.

No raw CSV, command, hash, clock or log is relabeled or removed. Paths were
fixed, explicit source/binary SHA checks were added before launching, all six
current builds were rebuilt, and340 current-binary checks passed anew in
`../direct-only-corrected/`. After the persistent StageK8 update, another
340-check rebuild/regression passes in `../direct-only-persistent-regression/`.
The StageK8 experiments also assert their compiled depth/StageK/unroll fields
and use new explicit executable paths, hashes and source snapshots.
