# A / B / C: coalescing per instruction versus eventual byte coverage

User-directed comparison. One wave owns a 256-float (1 KiB) tile; every
variant eventually reads exactly that tile, once. No floating-point math:
finite float values are loaded, their bits are accumulated as a checked
modulo-2^32 checksum. This keeps numerical reduction differences out of a
memory-access experiment.

## Result

**A and B perform similarly once there is enough independent work. Explicitly
scalar C is about half as fast and is not rescued by more concurrency.**
The normal compiler vectorizes C into A's wide-load pattern, removing the
penalty. This is a statement about generated instructions, not a blanket
warning against contiguous values per lane.

All numbers: native O3 HIP/C++20, gfx90a, HIP logical device 0,
**PCI 0000:31:00.0, 104 CUs**, CPU+memory NUMA node 0 via numactl,
**no model/context**. Input-byte decimal GB/s, not physical HBM traffic.
HIP/hipcc 7.1, Ubuntu clang 21.1.8. No clock/power changes; sampled active
clocks 1700/1600 MHz core/memory. Sampled memory temperatures in the
controlled repeats were about 56–68 C.

### Accesses within one tile

For lane l and component k:

| Reader | Float index | Intended hot memory instructions per tile |
|---|---|---|
| A, vector contiguous | `4*l + [0,1,2,3]` | one 16-byte vector load per lane |
| B, scalar striped | `l + 64*k` | four scalar loads, adjacent lanes adjacent floats |
| C, scalar contiguous | `4*l + k` | four scalar loads, adjacent lanes 16 bytes apart |

A/C have the same per-lane ownership; B transposes ownership within the
wave. They preserve the global byte set, not necessarily any real algorithm's
per-lane result. Every full-tile load is aligned as required; base/size
alignment is held fixed. Tail reads are masked separately.

### Controlled comparison: intended widths verified

Two independent process runs, nine randomized rounds × 12 reads/batch,
104×256, 1 GiB. `chunks` is the number of independent tiles staged per
iteration, not the element/vector width:

| Chunks | A median A/B | B median A/B | C median A/B |
|---|---:|---:|---:|
| 1 | 957.6 / 954.6 | 929.7 / 930.3 | 652.4 / 652.7 |
| 2 | 1336.4 / 1369.6 | 1352.5 / 1349.4 | 676.8 / 676.8 |
| **4** | **1432.3 / 1427.2** | **1430.3 / 1434.6** | **675.6 / 675.6** |
| 8 | 1353.3 / 1353.8 | 1427.5 / 1435.5 | 675.8 / 675.6 |

At 4 GiB/chunks=4, A is 1398.0/1398.6, B 1392.1/1392.6,
C 677.2/677.2 GB/s. At 832×256, C stays about 674 at 1 GiB and 676
at 4 GiB across the chunk counts. Those larger grids are launch sizes,
not claims that every variant fits entirely resident at once; capacities
are recorded in `resources.csv`.

This is not the earlier compiler-serialization cliff: at chunks=4,
**B and C both issue sixteen scalar loads before the first wait** and have
the same graduated wait sequence, 52 reported registers, zero scratch,
and 8-CTA/CU occupancy capacity. A reports 50 registers and the same
capacity. No `glc`, `slc`, or `dlc` flag differences are present.

### Normal compiler: C is no longer scalar

Controlled mode applies these flags equally to all readers:

```
-fno-vectorize -fno-slp-vectorize -mllvm -amdgpu-load-store-vectorizer=false
```

A uses an explicit vector type and retains `global_load_dwordx4`.
B/C retain scalar `global_load_dword`. No volatile loads or altered cache
policies are used. AMD scheduling constraints prevent the staged results
being consumed before the intended load batch; they are compiler controls,
not GPU synchronization. Earlier IR vectorization can still merge C's
source loads in the normal build, before those scheduling constraints act.

Two normal-O3 process repeats, chunks=4, 104×256, 1 GiB:

| Reader | Median run A | Median run B | Actual hot load width |
|---|---:|---:|---|
| A | 1426.0 | 1441.7 | 16 bytes |
| B | 1421.0 | 1442.7 | 4 bytes |
| C | **1432.1** | **1422.0** | **16 bytes** |

At 4 GiB all three are roughly 1390–1400. The scalar-width controls do not
materially change A/B's result; C recovers because its four adjacent loads
are combined into the wide instruction. The controlled build deliberately
measures a real scalar instruction pattern, not what this particular
normal compiler chooses for that source.

## What this establishes

- Eventually reading the same bytes does not imply equivalent throughput.
- Four coalesced scalar passes (B) can match one wide pass (A) with adequate
  batching; fewer instructions alone are not a universal throughput winner.
- Scalarizing per-lane contiguous records (C) produces inefficient accesses
  within each instruction: 256 useful bytes per wave instruction spread over
  a 1 KiB address span, versus B's contiguous 256-byte span.
- Any cache reuse between C's passes does not restore A/B's measured rate here.
- This supports a transaction/request/coalescing penalty, **not a measured
  fourfold increase in HBM traffic or a uniquely identified LSU/cache unit**.
  Hardware counters remain unavailable due to the previously recorded
  AQL-profile initialization failure. Do not infer a physical channel map.
- Simply doubling batching from four to eight can hurt A while B remains
  fairly steady. More unroll or wider loads is not automatically better.

## CU/SIMD packing

The follow-up [packing study](packing.md) records hardware CU, SIMD and
wave-slot IDs plus per-wave entry/exit times. A/B/C spread identically:
104 CTAs use all 104 CUs with one CTA/CU and one wave on each SIMD; grids
208/416/832 give 2/4/8 co-resident CTAs/CU. A 1664-CTA control queues later
work rather than making sixteen CTAs resident per CU. See the linked report
for repeated traces, resource checks and instrumentation caveats.

## Code and protocol

Three named readers (`A`, `B`, `C`) in private `readers.hpp` make their
accesses explicit, with one shared traversal/checksum used by the bandwidth
and packing probes. No W=1/2/4 element-width branch ladder. Native
Clang vector types keep the four-component accumulation concise.

All readers use identical full-tile traversal: start at the global wave
index, stage `chunks` tiles separated by the number of launched waves, then
advance by `chunks * waves`. Every wave tile is disjoint. One unique checksum
output per wave avoids races; the host checks the sum of all wave outputs.
Float initialization generates nonzero finite values in [0.5,1), with a
permutation of 23-bit mantissas and an analytic whole-buffer checksum.
It is synthetic data, not a model or a claim of realistic FMA cost.

One ordinary hipMalloc 4 GiB input, initialized outside timing. Small/tail
checks: 64 B, 4160 B, 1 MiB+64 B. Full checksum after every timed batch.
No allocations or host transfers inside the event interval. Reduction and
sparse checksum writes are timed. No explicit cache flush; 1/2/4 GiB working
sets exceed L2. No host/io/model/attention activity inside the benchmark.

The quick phase measures A/B/C, chunks=1, 104×256, 1 GiB. The sweep uses
chunks=1/2/4/8, grids=104/832, block=256, sizes=1/2/4 GiB jointly shuffled
in each round. 250 preheat reads, two case warmups, nine rounds of twelve
reads. Two controlled and two normal compiler process repeats are retained.
No scratch-using candidate is accepted.

`inspect.py` checks hot-loop widths, cache flags and scratch instructions
**before** future runs, failing loudly if a controlled reader changes.
It separates the first hot load batch from masked tail instructions, which
are scalar even for A. `load-widths.csv` records what the normal build
actually emits for C rather than assuming vectorization is guaranteed.
The recorded runs were also inspected retrospectively with this checker.

## Artifacts and reproduction

Summaries, load-width reports and resource tables are in `results/`.
Raw rounds, sources, executables, hashes, ISA, device information and
read-only telemetry are in `temp/vram-coalescing/{sweep-a,sweep-b,default-a,
default-b,quick-a}/` (gitignored, durable repository-local storage).

```sh
research/vram-coalescing/build.sh
research/vram-coalescing/run.sh temp/vram-coalescing/new-quick controlled quick 9
research/vram-coalescing/run.sh temp/vram-coalescing/new-sweep controlled sweep 9
research/vram-coalescing/run.sh temp/vram-coalescing/new-normal default sweep 9
```

Both binaries use O3 C++20/gfx90a. The runner rejects stale binaries,
verifies generated load shapes, records metadata and reuses the ceiling
study's read-only telemetry helper. Python is standard-library only.
No production engine, package or system changes.
