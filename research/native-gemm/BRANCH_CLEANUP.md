# Branch-only steady-loop cleanup

User requested this change before trying MFMA/prefetch interleaving, with312
CTAs retained in the test grid. This experiment changes **control flow only**.
The producer/consumer memory and math schedule, raw slots, double-q3 parity,
wait placement, barriers and register/LDS footprint stay unchanged.

## Synchronization clarification

In steady state, the publication LGKM0 follows next-panel LDS writes with
current math in between. It does not drain the future VMEM prefetch. The last
next-fragment read has current q3 math before the following retirement LGKM0.
The spare q3 slot permits next q3 to start early without overwriting current
q3. Startup necessarily reads then waits before first use. None of this assumes
OOO MFMA issue or a backlog of summed MFMA latencies after an instruction batch.

## Removed hot-path control

The old two-stage backedge has:
- two final-stage comparisons and two forward branches;
- two future-fetch-bound comparisons and two forward branches;
- two single-stage counter decrements;
- two unconditional phase jumps, including one fall-through `s_branch 0`.

Those12 instructions become **three**:

```asm
# Two complete stages, with their unchanged memory/math instruction order.
s_sub_u32 remaining, remaining, 2
s_cmp_ge_u32 remaining, 4
s_cbranch_scc1 steady
```

The pair is entered only with at least four stages remaining. Thus both
stage+2 prefetches are in bounds, with no per-phase checks. A pair consumes
two stages and restores phase0 q3 parity. After the loop, one/two/three stages
are statically handled; startup separately handles zero. The bounded tails
no longer update the local remaining counter, which is unused after the asm.
No unconditional jump to the immediately following instruction remains in
the generated timed body.

|Selected128/K32 loop|Before|After|
|---|---:|---:|
|Total instructions|90|81|
|Control instructions|12|3|
|MFMA /global x4 /DS write128 /DS read128|32/8/8/16|32/8/8/16|
|LGKM waits /VM waits /barriers|4/2/4|4/2/4|
|VGPR /AGPR /SGPR metadata|136/64/46|136/64/46|
|Dynamic LDS /private bytes|16KiB /0|16KiB /0|

The optional256 tile hot loop similarly drops222->213 instructions. Static
whole-body math/read/write counts increase because several mutually exclusive
bounded tails are now materialized. Executed work for any K is unchanged.
Whole selected GPU .text grows7872->8704 bytes; this is not a code-size-free
transformation. No individual branch latency is inferred from the speedup.

## Equivalence and bounded-tail checks

`audit_branch_flow.py` interprets only generated labels, remaining-counter
updates and SCC branches. For29 stage counts (including0/1/2/3, pair boundaries,
odd/even counts and2048), each of four tile/StageK configurations has an exact
match in its **ordered non-control instruction trace**: loads, stores, waits,
barriers, MFMAs, pointer updates and clocks. Four resident-control headers stay
byte-identical. This is not a hardware emulator or a substitute for GPU tests.

Actual native hot-loop non-control instructions also match in order/text for
128 and256. `inspect_code.py` checks the new3-instruction control backedge,
unchanged math/memory counts, absence of zero-displacement unconditional
branches in timing and zero spills. `results/branch-flow-audit/` stores trace
hashes/native-loop-order proof and10 additional StageK64 GPU boundary checks.

Functional tests cover the original dimensions/seeds plus K95/96/97,
127/128/129,191/192/193,255/256/257 on both output tiles. These stress full and
partial final stages at the peeled-tail boundaries. No source access beyond
packed K is introduced.

## Matched A/B results

Native HIP O3/C++20 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0, four waves/CTA,
no model/context/clock override. Same packed input matrices/reference contract.
Two counterbalanced repetitions, nine checked eager launches/process after
200ms selected-kernel warmup. Full HIP-event useful TFLOP/s; packing/H2D/CPU
checking excluded.104/208/312 CTAs are all retained, plus one large-grid check.

|Grid CTAs|M x N x K|Before TFLOP/s|After TFLOP/s|After clock-adjusted efficiency|
|---:|---|---:|---:|---:|
|104|1664x1024x8192|113.153/113.374|115.781/115.781|74.11/74.07%|
|104|1664x1024x32768|119.816/119.857|122.487/122.465|78.69/78.67%|
|208|3328x1024x8192|127.592/127.639|130.454/130.405|85.13/85.01%|
|208|3328x1024x32768|132.611/132.560|135.258/135.323|88.53/88.53%|
|312|4992x1024x8192|134.944/134.944|136.385/136.279|89.70/89.61%|
|312|4992x1024x32768|138.478/138.451|139.838/140.025|92.30/92.44%|
|1664|4096x6656x32768|143.985/144.013|145.919/145.879|96.71/96.72%|

This is a modest repeatable gain, especially useful at104/208 CTAs.208 long-K
improves~2% in useful rate and~2 percentage points in clock-adjusted efficiency,
but remains below90%. We have not yet moved the future global prefetch among
MFMAs or otherwise addressed the coarse vector-issue schedule.

## Provenance / replay

The before executable is the original preserved readability-cleanup build;
its compilation sources are rebased to `results/readable/source/`. The after
build uses a fresh namespace. The driver verifies exact executable hashes
before each process, asserts equal selected-kernel resource fields, and
records compilation snapshots separately from the actual shared driver.

**312 checked launches** =60 bounded-tail/seed functional checks plus252 grid
timing launches. All sampled/full references as applicable, every-C finite
poison/coverage checks and output guards pass. All raw clocks are independently
validated by the same Case/run_case helpers; no reference/count/clock policy
change. Large matrices are still sampled-reference, not full CPU products.

`results/branch-clean/{before,after,after256}/` holds per-variant sources,
headers, metadata, manifests, CSV/logs and clocks. Root `driver/` records the
actual shared harness and hash verifier used. Native executables/full ISA:
`temp/native-gemm/branch-clean/s32-t*`; old `readable/` builds are unchanged.

```sh
# Replay against preserved branch-clean builds/source snapshots, not today's generator.
python3 research/native-gemm/audit_branch_flow.py --output <fresh-flow-audit>
python3 research/native-gemm/run_branch_compare.py --output <fresh-A-B-results>
```

The later MFMA-first experiment has a separate generator/build namespace;
current build.sh no longer builds this historical branch-only algorithm.
The branch audit and branch-comparison driver pin its completed snapshots.

Current default full-problem runner also retains208/312 coverage; its expanded
suite is distinct from the original180-launch cleanup suite. Historical
cleanup proof now pins the immutable readable source/build snapshot rather
than expecting later intentional algorithm changes to be byte-identical.
No production/CMake/model/system/package/clock-policy change. Research status
is appended in `research/research-log.md`.
