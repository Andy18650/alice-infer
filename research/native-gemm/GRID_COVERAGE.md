# Selected kernel with104,208 and312 CTAs

User asked whether the128x128x32 kernel was tested with fewer CTAs/CU, and
specifically how a208-CTA grid performs. Previous final suites covered104,
416 and1664 CTAs, not208. This targeted follow-up uses the preserved readable
executable with **no kernel, packing or scheduling changes**.

## Protocol

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, tile128/K32,
four waves/CTA, linear LDS, no model/context/clock override. Packed operands
and CPU reference outside timing. Two counterbalanced process repeats, nine
checked eager launches/process after200 ms selected-kernel warmup. Same
integer/16 test domain, exact sampled row-major references, all-C finite
scan after NaN poison and output guards as the main suite.

Keep N=1024 and increase M for the primary104/208/312 sequence. Also test a
transposed grid aspect at208 CTAs; the requested matrix product is still
A[M,K] times B[K,N], not a matrix-transpose flag. All dimensions tile-aligned,
so no useful/padded arithmetic discrepancy.

The measurement driver invoked existing `run.Case`, `run.run_case` and
`run.write_summary` helpers from Python, with the explicit problem list in
`results/grid-coverage/invocation.json`. No separate timing implementation.
Commands in `commands.txt` are the exact executed CLI calls; the source/build
snapshots include the shared runner and executable fingerprints.

## Full-kernel useful TFLOP/s

Process medians run1/run2:

| M x N | CTA grid | CTAs | K8192 | K32768 |
|---|---:|---:|---:|---:|
|1664x1024 |13x8 |104 |113.01 /113.15 |119.73 /119.80 |
|3328x1024 |26x8 |208 |127.69 /127.73 |132.52 /132.59 |
|1664x2048 |13x16 |208 |127.78 /128.01 |132.61 /132.73 |
|4992x1024 |39x8 |312 |134.91 /134.88 |138.45 /138.45 |

At208 CTAs the two grid orientations agree closely. Whole-GCD clock-adjusted
efficiency for K32768 is~86.5-86.6%, versus~76.6-76.7% at104 and~91.3% at312.
The largest earlier1664-CTA problem reached~144 TFLOP/s/~95.4%; that is a
different full problem, not a matched occupancy-only control.

This shows that the kernel remains useful at208 CTAs but does not reproduce
the large-grid result. More coverage improves throughput here without changing
the kernel. However, M/N, cross-CTA source reuse and admission/phase behavior
also change, so do not assign the whole difference uniquely to occupancy.

## Residency qualification

The executable's resource limit remains **three CTAs/CU** for every case.
208 grid CTAs are not an enforced two-CTAs-per-CU resource limit. No CU IDs
are recorded, so exact spatial placement is not established.

Representative repeat1/round4 raw steady-clock intervals show maximum104,
208 and312 overlapping CTA bodies for the corresponding grids, at both K
lengths. This confirms simultaneous body coverage, not allocation timestamps
or uniform one/two/three blocks on each CU. The overlap JSON retains each
selected file and its SHA256; selection is fixed, not a maximum-sample search.

A strict residency-only experiment would keep the full problem fixed and
change a resource cap/placement measurement. It was not performed in this
round; no LDS padding or alternate launch bounds were introduced.

## Checks / replay

**144 grid timing launches** pass all sampled references, every-C finite/
poison coverage and guards. Per-wave binary clocks are independently checked
by the shared runner; core rates are per CTA, grid makespans use steady
clocks, never cross-CU core timestamps.

Data, commands, original shared-driver/source/native fingerprints, per-round
CSV and clock dumps: `results/grid-coverage/`. The kernel is still
`temp/native-gemm/readable/s32-t128/probe`.

```sh
numactl --cpunodebind=0 --membind=0 temp/native-gemm/readable/s32-t128/probe \
  --m 3328 --n 1024 --k 8192 --rounds 9 --warmup-ms 200 --swizzle 0
# K32768, or M1664/N2048 for the second208-CTA orientation.
```

No kernel/algorithm/reference policy, production/CMake/model/system/package/
clock-policy changes. Research status appended to research/research-log.md.
