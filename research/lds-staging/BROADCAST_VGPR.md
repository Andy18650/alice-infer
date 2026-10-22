#64-CTA8x8 A-row/B-column source sharing

User-confirmed matched experiment:64 CTAs, four waves/CTA, unchanged32-slot
DWORDx4 VGPR ring, no periodic waits. Compare the disjoint source layout with
an8x8 outer-product-like row/column sharing pattern. Pure loading, no GEMM,
LDS staging, inter-CTA messaging or additional producer optimization.

## Source mapping

CTA c has i=c/8,j=c%8. Eight A panels and eight B panels,512 KiB each:

```text
A[0..7]: bytes0..4MiB
B[0..7]: bytes4MiB..8MiB

CTA(i,j), waves0/1: A_i +wave*256KiB
CTA(i,j), waves2/3: B_j +(wave%2)*256KiB
```

Each wave still traverses a private256 KiB range; each CTA still requests
its1 MiB working set. Source mask/window, ring layout and timed load loop
are unchanged. A_i is reused by eight CTAs along its grid row; B_j by eight
along its column. This is cache-mediated reuse, not an explicit hardware
broadcast instruction or a request to communicate between CTAs.

Disjoint control: CTA c reads bytes[c*1MiB,(c+1)*1MiB), with original wave
quarter offsets. Distinct active source64 MiB versus8 MiB outer. **Both cases
allocate/initialize64 MiB**, keeping allocation size controlled; outer's
remaining56 MiB is not read by the measured kernel. Footprint means reachable
addresses, not allocation size. Nominal L2 is8 MiB, but no counter-based
residency/miss-rate or physical-traffic claim.

## Address proof / matched kernel

Same executable/kernel for both layouts. The layout argument changes only
source-address setup before timing. The generated volatile-asm header is
identical to the preceding VGPR study:32 wide loads+16 address+5 control
instructions, no loop waits/consumers/copies/NOPs, terminal VM0 outside issue
interval.32 x4-VGPR payload slots, zero LDS/scratch/spills.

After timing, lane0 records the actual source pointer used to construct each
wave's input address. CPU checks all256 GPU-reported bases against the map
and bounds. Raw per-CTA CSV retains their offsets from the input allocation.
The driver independently verifies:
- Disjoint:256 unique256 KiB wave regions, each referenced once.
- Outer:32 unique256 KiB wave regions, each referenced exactly eight times.

This supplements the periodic-data oracle, which by itself cannot detect
source aliases or incorrect panel strides. Address capture adds live state:
native metadata138 VGPRs/30 SGPRs versus preceding136/28. Both compared
layouts use the same allocation and kernel resources; the fresh disjoint
control is replayed, not substituted with the old binary's numbers.

## Measurement

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, grid64/block256,
no model/context/clock-policy overrides. Two counterbalanced process repeats
(disjoint/outer, then outer/disjoint), nine checked launches/process,200 ms
selected-kernel warmup and unmeasured full source traversal. Original post-
warmup H2D output initialization protocol retained; no uniformly-hot claim.

Default2048 outer groups/65536 loads per wave,256 MiB requested/CTA,
**16 GiB/grid/launch**,256 source traversals per CTA. Source sharing reduces
distinct addresses, NOT instructions or requested byte accounting.

Core period is each CTA's absolute wave makespan divided by loads/wave;
median CTA per launch, then median of nine launches. One step includes four
native load instructions and returns **4 KiB/CTA**. Global requested GB/s
uses global steady-clock grid makespan, not cross-CU core timestamps. Full
HIP event includes output/setup; issue-end excludes drain and ring dumps,
completion-end follows terminal VM0. Epilogues may overlap other CTAs' loops.

## Results

Process medians run1/run2, timing overhead included:

| Source layout | Distinct active source | CTA cycles/four-wave x4 step | Grid requested GB/s | Full-event GB/s |
|---|---:|---:|---:|---:|
| Disjoint | 64 MiB | 309.1617 /309.2876 | 1376.75 /1375.78 | 1375.29 /1374.29 |
|8x8 outer | 8 MiB | **139.3126 /138.0228** | **3013.17 /3036.34** | 3005.58 /3026.50 |

Requested grid throughput improves **2.19x/2.21x**. The cycle-period reduction
is2.22x/2.24x; measured core/steady ratios differ slightly (~1633 MHz disjoint,
~1604 MHz outer), so cycle and wall-time ratios need not match exactly.
Shared case achieves more requested GB/s despite its lower measured ratio.
Completion bandwidth closely tracks issue bandwidth. All CTA body intervals
overlap in every sample; reported residency3 CTAs/CU for this kernel.

Thus the disjoint-streaming curve is substantially pessimistic for this
specific row/column sharing pattern. This is **requested useful traffic**:
~3 TB/s does not mean3 TB/s from HBM. The same source bytes feed eight CTAs
through caches/miss coalescing/return paths. We do not infer physical traffic,
perfect L2 residency, exact broadcast/coalescing machinery,8x speedup or a
universal cache bandwidth ceiling. Allocation/address placement and an8 MiB
working set at cache capacity are limitations. No actual GEMM or real-model
prefill speedup is measured.

## Validation / records

**36 grid timing launches /2304 CTA records** plus four64-CTA fill/wrap smoke
launches pass. Every final slot/lane/DWORD and32 global output guards checked;
all256 starting GPU source addresses verified per launch. Source-mask/wrap
address sequence audited in unchanged native loop. Periodic payload remains
same-valued under overwrite/reordered returns and is not a last-iteration
oracle; no intermediate consumers.

Exact commands, all clocks/source offsets, summaries, native/source hashes,
metadata and extracted loop: `results/broadcast-vgpr/`. Actual distinct binary/
full ISA/CDB: `temp/lds-staging/broadcast-vgpr/`. Earlier measured binaries
and sources preserved. Research status is in `research/research-log.md`.

```sh
bash research/lds-staging/build-broadcast-vgpr.sh
numactl --cpunodebind=0 --membind=0 temp/lds-staging/broadcast-vgpr/probe \
  --layout outer --rounds 9 --warmup-ms 200
# --layout disjoint for the matched control; fixed64 CTAs/four waves.
python3 research/lds-staging/run_broadcast_vgpr.py --output <fresh-results-directory>
```

No ring-depth/wait/width/address-skew tuning, production/CMake/model/system/
package/clock-policy changes or further experiment in this round.
