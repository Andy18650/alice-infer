# Local VRAM address-pattern and contention probes

Question: does MI250X/gfx90a behave like a simple 256-byte, 32-way HBM
interleave, and can address swizzling be detected experimentally?

## Findings

**There are strong, repeatable address-dependent contention signatures.**
But the measurements do **not** identify physical HBM channels, prove a
256-byte channel interleave, or recover an XOR swizzle. The simple
`channel = floor(address / 256) mod 32` model is inadequate as an
explanation of the observed effective bandwidth.

Conditions for all numbers below: standalone native **O3 HIP/C++20**,
MI250X/gfx90a, HIP logical device 0, **PCI 0000:31:00.0, 104 CUs**, CPU and
memory NUMA node 0 via numactl, **no model/context**. Useful input bytes
only, decimal GB/s. No clock/power changes; final-run 0.5 s samples read
1700/1600 MHz core/memory, sampled memory temperatures roughly 50–58 C.
HIP runtime 7.1.52801, Ubuntu clang 21.1.8. HIP device 0 is rocm-smi GPU 2.

### 1. 8192-byte stride is not the predicted catastrophic alias

A wave reads a contiguous 256-byte tile. Successive tiles are separated
by stride S. Fixed useful bytes and identical kernel/launch geometry:
832 CTAs × 256 threads, four source-level independent tile accumulators.
Two independent process runs per traffic size, seven randomized rounds,
12 reads per batch. Medians:

| S (bytes) | 32 MiB run A | 32 MiB run B | 64 MiB run A | 64 MiB run B |
|---|---:|---:|---:|---:|
| 8192 | 1100.4 | 1102.8 | 1146.5 | 1147.0 |
| 8448 | 1111.6 | 1098.9 | 1148.9 | 1144.9 |
| 16384 | 887.4 | 888.9 | 952.7 | 948.9 |
| 16640 | 1100.9 | 1104.7 | 1148.6 | 1142.3 |
| 32768 | 556.0 | 556.5 | 581.1 | 580.6 |
| 33024 | 1089.4 | 1084.7 | 1138.5 | 1129.8 |
| 65536 | 303.2 | 303.5 | 309.9 | 309.9 |
| 65792 | 1063.2 | 1070.0 | 1121.2 | 1115.5 |

A 256-byte padding increment consistently rescues 16/32/64 KiB strides;
the run-B 32 MiB gains are approximately 24%, 95%, and 253%. The 8 KiB
stride already performs well. The useful payload, wave count, tile count,
instructions and arithmetic are unchanged within each stride comparison.
Padding increases the sparse address span by only the relative stride
increment, not by the large measured speedup.

Under the literal no-swizzle 32×256 model, all 8 KiB-spaced tile starts
would select one channel; that is not the measured behavior. This does
not license a new physical channel count or stripe size. Channel, bank,
cache partition, translation and address-scheduling effects are not yet
separated, and HIP addresses are virtual.

### 2. 256-byte alignment matters, but is not a channel-map proof

For S=8192, sweep the tile's base offset in 32-byte increments. Final
32 MiB screen, same high-concurrency geometry, five randomized rounds:

| Base offset | GB/s |
|---|---:|
| 0 | 1094.2 |
| 32 | 900.1 |
| 64 | 904.6 |
| 96 | 756.2 |
| 128 | 809.2 |
| 160 | 772.9 |
| 192 | 903.3 |
| 224 | 896.9 |
| 256 | 1092.3 |

The shape repeats every 256 bytes through offset 1024, also present in
an earlier independent screen. It demonstrates an alignment/transaction
sensitivity for this access pattern. Misalignment changes how the tile
straddles cache lines/transactions as well as any channel phase, so this
is **not evidence sufficient to call the HBM interleave 256 bytes**.

### 3. Two disjoint streams reveal a conflict period

Each wave reads A and B using separate accumulators in the same kernel.
The streams have identical strides; B begins at A + 8 GiB + delta.
Both sparse address spans are disjoint (host-enforced), so there are no
shared input cache lines. Combined useful payload remains 32 MiB.

The full `conflict-final` sweep samples delta from zero through 2S in
steps S/64. It shows a repeating penalty near delta=0,S,2S and relief
near S/2,3S/2 for S=16/32/64 KiB. Independent process confirmations in
`planes-extended-{a,b}` retain the same behavior. Run B, unroll 4, base 0:

| Stride | 8 GiB gap, delta 0 | 8 GiB gap, delta S/2 |
|---|---:|---:|
| 8192 | 1132.6 | 1129.5 |
| 16384 | 873.2 | 1128.5 |
| 32768 | 542.6 | 881.8 |
| 65536 | 295.8 | 547.8 |

The 16 KiB case improves about 29% by changing **only B's displacement**.
This is a contention signature, not simply a one-stream stride observation.
The dip/relief pattern is not perfectly square; intermediate delta values
have additional structure. Four-stream cases are also retained in the
full sweep, but do not establish a channel count: adding streams also
changes request scheduling and register work.

### 4. Higher address information affects the signature

Within one allocation/process and randomized paired rounds, change only
B's plane separation among 8, 16 and 24 GiB, with delta=0. Same two-stream
kernel, byte count and geometry. Two final runs, base 0, unroll 4:

| Stride | 8 GiB gap A/B | 16 GiB gap A/B | 24 GiB gap A/B |
|---|---:|---:|---:|
| 8192 | 1135.6 / 1132.6 | 1115.5 / 1112.6 | 1137.7 / 1135.6 |
| 16384 | 877.5 / 873.2 | 1056.5 / 1061.0 | 872.0 / 869.6 |
| 32768 | 542.4 / 542.6 | 805.6 / 806.1 | 546.8 / 546.6 |
| 65536 | 295.6 / 295.8 | 527.8 / 528.0 | 296.2 / 296.1 |

All three gaps preserve the same low 16 displacement bits, yet the
16 GiB gap behaves differently from 8/24 GiB. This is evidence that
**low displacement bits alone do not explain contention**. It is consistent
with higher-bit routing/address mixing, but could also reflect physical
page placement, bank organization, cache routing or translation effects.
It does not prove XOR swizzling or identify which physical bit matters.
Changing a relative displacement is not necessarily toggling a particular
absolute virtual or physical bit.

The earlier `bits`/`pair-bits` screens perturb offsets by powers of two
from bit 12 through 30, with a second low offset of 256. Single-stream
aligned results barely move compared with stride effects. These streams
span many pages and higher-bit values already, so absence of an effect
there does not imply absence of a swizzle. The more informative plane-gap
comparison above was added after discovering the paired-stream signature.

## Method and validation

`probe.hip` is self-contained; no engine objects, headers or vendor BLAS.
One 36 GiB hipMalloc input allocation, initialized on-device outside timing,
and a separate sink with one unique checksum per native wave. The input
initializer includes both low and high word-index bits; a 16 GiB shift
**does not duplicate stream contents**. Analytic host checksums account for
all selected words, including transitions across the high-bit thresholds.

- Tiles 64/128/256/512 B: 16/32 active lanes with uint32 loads, all 64 lanes
  with uint32, or all 64 with uint2. Comparisons across tile sizes change
  both transaction coverage and active lanes; do not attribute them to a
  channel width alone.
- Low concurrency: 104 CTAs × 64 threads. High: 832 CTAs × 256 threads.
  These are launch geometries, not claims of exact residency/CU placement.
- Stride screen: 600 cases, 64 B through 64 KiB plus nearby offsets,
  base offsets 0/256, all tile sizes and both geometries.
- Offset screen: 396 cases, bases 0..1024 step 32, six strides, both geometries.
- Bits / pair-bits screens: 320 / 312 cases.
- Pair screen: 1056 cases, two/four streams, four strides, bases 0/256,
  delta 0..8192 step 256, both geometries.
- Conflict sweep: 1548 cases, two/four streams, bases 0/256, three strides,
  delta 0..2S step S/64, high geometry.
- Plane-gap confirmation: 288 cases, eight strides, bases 0/256, delta
  0/S/2, unroll 1/4/16, gaps 8/16/24 GiB, high geometry. Use 32 MiB.
- Confirmation: 384 predeclared cases, eight strides, bases
  0/256/1 MiB/256 MiB, unroll 1/4/16, both geometries, plus paired-stream
  displacement controls. Run at 32/64 MiB with a fixed 16 GiB plane gap.

Before timing, all cases pass 17-tile-per-stream and full-payload checks.
Each timed batch is followed by full wave-output checksum validation.
Preheat is 500 dense reads; case warmup is two reads. Screening uses five
independently shuffled rounds, six reads per batch. Confirmation uses
seven shuffled rounds, 12 reads per batch (`planes` uses six reads).
Initialization, allocation, and host checksum readback are outside events;
wave reduction and sink stores are inside. No cache invalidation between
reads: selected input is at least 32 MiB, above the GCD's L2 capacity.
Useful bytes differ from cache-line or HBM traffic; cache misses were not
measured, so do not label these rates as physical DRAM bandwidth.

All streams' bounding spans must be disjoint. A first 64 MiB confirmation
with an 8 GiB gap rejected the padded 64 KiB case because its span exceeded
the gap; no timing from that failed run is used. Both final traffic sizes
use a 16 GiB gap. The guard was retained, not bypassed.

Actual executable disassembly under `planes-final-b/disassembly.txt`
confirms uint32/uint2 load widths, retained unrolled reads, and no scratch
load/store instructions in checked 256 B single/pair and 512 B kernels,
including paired unroll 16. The later extended-plane case list does not
change the kernel body.

## Counter collection: blocked, not silently worked around

An existing rocprofv3 installation was found outside PATH under
`~/.cache/uv/archive-v0/2EEHNuXZame80Ad1/_rocm_sdk_core/bin/`.
`--list-avail` reports TCC_HIT, TCC_MISS, TCC_EA_RDREQ and
TCC_EA_RDREQ_32B. An attempted filtered collection aborted at initialization:

```
aqlprofile API table load failed: HSA_STATUS_ERROR
```

No usable counter data was produced. The timed runs are unprofiled and
unaffected by that attempt. Availability/help/logs are retained under
`temp/vram-addressing/`; `profile/run.log` records the error. No packages,
vendored copies, library symlinks or system changes were made. Repairing the
profiler setup is a separate prerequisite for a counter-backed study.

The next useful discriminator is per-instance TCC request/miss/EA traffic
and, if available, per-memory-controller activity. TCC instances are cache
interfaces, **not automatically HBM channels**. Counters could distinguish
cache/transaction effects from an imbalanced downstream path, but physical
channel naming still needs architecture/address-translation evidence.

## Reproduce and artifacts

```sh
research/vram-addressing/build.sh
research/vram-addressing/run.sh temp/vram-addressing/new-stride stride 0 32 5
research/vram-addressing/run.sh temp/vram-addressing/new-offset offset 0 32 5
research/vram-addressing/run.sh temp/vram-addressing/new-conflict conflict 0 32 5 8
research/vram-addressing/run.sh temp/vram-addressing/new-planes planes 0 32 7 8
research/vram-addressing/run.sh temp/vram-addressing/new-confirm confirm 0 64 7 16
```

Arguments: output, phase, HIP device, useful MiB, odd round count, default
plane GiB. Bounds errors fail loudly; not every phase fits every size/gap.
`profile` is a small one-launch-per-case mode for future counter collection,
not a performance result. All Python helpers use the standard library;
runner reuses the ceiling study's read-only telemetry helper.

Final summaries are retained in `results/*.csv`; full raw rounds, hashes,
preserved sources/binaries, device addresses and telemetry are in matching
`temp/vram-addressing/` directories. Exploratory older-source runs remain
there too; final tables use the high-word-mixed initializer. No production
engine changes. This research supplies reproducible conflict patterns,
not a physical channel map or a new engine layout policy.
