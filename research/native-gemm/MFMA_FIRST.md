# One ready MFMA immediately after the head VM0

User proposed that after the head LGKM0/barrier/VM0 waits, the previous MFMA
may already have finished, leaving staging work to run with the matrix unit
idle. This test moves **one existing current-q0/C00 MFMA** immediately after
VM0, before the next-panel LDS writes. No other scheduling change is included.

## Exact selected128/K32 change

```text
BEFORE                            AFTER
LGKM0; retirement barrier         LGKM0; retirement barrier
VM0                               VM0
                                  MFMA current q0/C00
four LDS b128 writes              four LDS b128 writes
four future global x4 loads       four future global x4 loads
source pointer advance           source pointer advance
MFMA current q0/C00
MFMA current q0/C01               MFMA current q0/C01
MFMA current q0/C10               MFMA current q0/C10
MFMA current q0/C11               MFMA current q0/C11
LGKM0; publication barrier        LGKM0; publication barrier
...                              ...
```

The moved instruction consumes current operands already in raw VGPRs. It
neither reads the producer register set nor needs next-panel LDS contents.
It is moved, not duplicated. All per-accumulator K order and all memory/
wait/barrier/pointer order stay the same; retirement/publication correctness
and the q3 spare slot are unchanged. There is no interleaving of the four
future global packets among the remaining MFMAs in this test.

The remaining three q0 MFMAs give the LDS writes elapsed execution opportunity
before publication. Net performance improves, but individual wait stall
cycles were not instrumented: this does NOT prove every LDS write always
finishes before the publication wait or establish a fixed192-cycle window.
MFMA issue remains in-order/pipeline-constrained, not an assumed OOO backlog.

Native selected hot loop stays81 instructions,32 MFMAs,8 global x4,8 LDS
writes,16 LDS reads,six waits,four barriers and three loop-control instructions.
Both VM0 instructions in the two-stage backedge are immediately followed by
one MFMA, then LDS stores. Metadata136 VGPR/64 AGPR/46 SGPR,16 KiB LDS,zero
private segment remains identical to the branch-clean control.

## Matched measurements

Native HIP O3/C++20 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0, block256,
no model/context/clock override. Same packed independent operands,reference,
output poison/guards and event protocol. Two counterbalanced process repeats,
nine checked eager launches/process after200 ms selected-kernel warmup.
Packing/H2D/CPU reference/checking outside timing.104/208/312 CTAs remain
in the grid, plus the large-grid regression check.

Useful full-kernel TFLOP/s, run1/run2:

|CTAs|M x N x K|Before(branch-clean)|After(one MFMA moved)|After clock-adjusted efficiency|
|---:|---|---:|---:|---:|
|104|1664x1024x8192|115.629/115.552|120.003/119.920|77.31/77.26%|
|104|1664x1024x32768|122.401/122.444|127.453/127.430|82.47/82.41%|
|208|3328x1024x8192|130.357/130.455|132.637/132.536|86.83/86.71%|
|208|3328x1024x32768|135.141/135.245|137.904/137.986|90.65/90.63%|
|312|4992x1024x8192|136.280/136.244|137.425/137.388|90.39/90.44%|
|312|4992x1024x32768|139.941/139.782|141.588/141.588|93.61/93.62%|
|1664|4096x6656x32768|145.773/145.897|146.550/146.436|97.16/97.08%|

The208-CTA long-K case now exceeds90% clock-adjusted efficiency, without
requiring the312-CTA grid. The104-CTA case benefits more in relative useful
rate (~4%); larger grids gain less because their existing scheduling coverage
already hides more gaps. This is consistent with improved issue overlap,
not an isolated measurement of matrix idle cycles or publication-wait cost.
No fixed per-CU spatial placement is inferred from grid count.

## Checks and audit

**312 checked launches** =60 functional/boundary/input-seed checks plus252
matched timing launches. All original-row-major full/sampled references as
appropriate,all-C finite/poison checks,guards and raw-clock calculations pass.
Large matrices remain sampled-reference, not full CPU dot products.

`audit_mfma_first.py` takes the before executed trace, skips startup VM0, and
moves the first following MFMA immediately after every stage-head VM0. For29
stage counts and each of four tile/StageK configurations, this exact expected
trace matches after; the count is stages-1, including zero/small boundary
cases. Every other instruction retains its order. Resident-control headers
are byte-identical. Actual128/256 native hot loops match the same exact two-
move transformation and unchanged instruction count. Native audits also
check no spills, unchanged traffic/math/control and immediate post-VM0 MFMA.
This generated control-flow model is not a hardware emulator; GPU tests are
a separate gate.

Before compilation sources/binaries are the preserved branch-clean snapshot;
after uses new `mfma-first` paths. Exact executable hashes are checked before
every process. `run_mfma_first.py` is a small entry point reusing the same
Case/run_case/shared A/B driver; it does not duplicate a timing harness.
Source compilation snapshots are separate from actual driver code, so old
before paths are never checked against today's changed generator.

## Replay / records

```sh
# Historical comparison uses preserved branch-clean/mfma-first builds.
python3 research/native-gemm/audit_mfma_first.py --output <fresh-order-audit>
python3 research/native-gemm/run_mfma_first.py --output <fresh-A-B-results>
```

The recorded builds are under `temp/native-gemm/mfma-first/`; old branch-clean,
readable and original measured executables remain untouched. Raw clocks,
reference/check results,resource snapshots,source/native fingerprints and
executed commands: `results/mfma-first/`. Order proof:
`results/mfma-first/checks/`. The historical branch-flow audit now pins its
completed before/after source snapshots rather than assuming that later
intentional MFMA scheduling changes preserve its old projection.

No extra tile/ring/wait/padding/skew tuning,production/CMake/model/system/
package/clock-policy changes. Research status appended to research-log.md. Later FP8 host scaffolding moves
normal build defaults to `fp16-bridge` (GPU-byte-identical); the historical
comparison pins its original source snapshots rather than today's host file.
