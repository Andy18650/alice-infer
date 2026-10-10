# Block/thread load indexing versus explicit wave/lane indexing

**Block/thread indexing is now the baseline.** Matched comparisons show
similar read throughput, without wave/lane arithmetic in the load loop.
Wave indexing survives only in the checksum reduction and the separate
historical control (`wave-control.hpp`).

## Matched experiment

A/B, four chunks, 256 threads, normal O3 HIP/C++20, scheduling barriers off,
uint32_t input / HIP uint4. HIP device 0, PCI 0000:31:00.0, 104 CUs,
CPU+memory NUMA node 0, no model/context, no clock overrides. Two independent
processes, nine jointly shuffled rounds × twelve reads for reader, indexing,
grid 104/208/416 and sizes 1/2/4 GiB. All short/tail and full checksums pass.

104×256 medians, decimal useful input-byte GB/s:

| Kernel | 1 GiB run 1 / run 2 | 4 GiB run 1 / run 2 |
|---|---:|---:|
| A, wave indexed | 1434.3 / 1437.9 | 1401.7 / 1400.7 |
| A, block/thread indexed | **1422.6 / 1429.3** | **1400.2 / 1397.6** |
| B, wave indexed | 1437.9 / 1445.0 | 1401.7 / 1399.5 |
| B, block/thread indexed | **1436.0 / 1436.4** | **1387.3 / 1384.4** |

The change is not uniformly faster: B has a repeatable ~1% 4 GiB loss at
104 CTAs, while some larger-grid cases improve slightly. This is sufficiently
similar to simplify the baseline, not proof that addressing never matters.
A post-promotion rerun gives 1421/1446 GB/s for block A/B at 1 GiB,
compared with wave controls 1430/1461. Do not compare best batches across
runs; the matched difference remains small.

Both layouts retain intended batches: A issues four 16-byte loads before
its first consumption wait; B issues sixteen scalar loads. There is no
scratch traffic, and both have the same 8-CTA/CU occupancy capacity.
Reported registers are 52/54 for block A/B, versus 54/55 for wave controls.

## New load loop

For a block of T threads, one block tile has `4*T` dwords. With T=256,
that is 1024 dwords / 4 KiB. Input remains a flat contiguous uint32_t array,
with no repacking. Thread t reads:

- A/C: `base + 4*t + component`;
- B: `base + t + component*T`.

Each B scalar pass covers one contiguous T-dword stripe across the block.
Within a 256-thread block, its four native waves read adjacent 64-dword
subranges. B transposes per-thread ownership compared with the wave control,
whose component stride was 64; both cover the same 4 KiB block region.

The shared traversal is:

```text
block_words = 4 * blockDim.x
grid_words  = gridDim.x * block_words
base        = blockIdx.x * block_words

load chunks at base + chunk * grid_words
consume their four-word values independently
base += Chunks * grid_words
```

Full-block iterations use a uniform whole-block fit check. Masked tails
cover short inputs, partial block tiles and leftover chunks. A/C's word
addresses are algebraically identical to the old mapping for complete
blocks; loop control and generated address arithmetic can still differ.

The load loop does not calculate a global wave number or lane number.
The final `__shfl_down` checksum still works in wave64 units and writes
one output per wave, so its lane-zero/output index bookkeeping remains.
This is independent of how reads are indexed.

## Controls and compatibility

`readers.hpp` contains the current block/thread `read_body` and ordinary
`read_tiles` entry. The reader's `component_stride` is explicit, normally
blockDim.x; A/C ignore that parameter, B uses it. `wave-control.hpp` holds
the old wave-indexed traversal and `read_wave_tiles` entry, explicitly passing
64. The `index` phase selects both entries in one matched process.

The ring-buffer study intentionally remains wave indexed, including its
batch controls. It includes `wave-control.hpp` and passes 64 explicitly,
so changing the baseline does not silently alter historical ring ownership
or comparisons. It is not evidence about the new block-striped ring shape.

A three-sample new-baseline placement check (`block-packing/`) still sees
1/2/4/8 CTAs per CU at grids 104/208/416/832 and one wave per SIMD per CTA.
All entry/exit identities and slot-overlap checks pass. The all-layout
baseline smoke sweep (three rounds, 1/2/4 GiB) also passes all checksums.

For production kernels, block/thread coordinates are enough to express
coalesced reads. But B's new layout changes which inputs each thread owns:
real per-thread arithmetic/output assignments must be adapted correctly,
not merely replaced because a global checksum matches.

## Reproduce / artifacts

```sh
research/vram-coalescing/build.sh
research/vram-coalescing/run.sh temp/vram-coalescing/new-index default index 9 off
```

Comparison summaries: `results/index-{a,b,promoted}.csv`,
`index-resources.csv`, `block-packing.csv`. Raw rounds, source/header/binary
snapshots, ISA and telemetry: matching `temp/vram-coalescing/` directories.
`index-a/b` are pre-promotion comparison binaries; `index-promoted` checks
the promoted baseline against the unchanged wave control. Historical
artifacts are not overwritten. No production engine or system changes.
