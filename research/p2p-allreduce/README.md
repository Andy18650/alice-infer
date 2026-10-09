# Out-of-place gather: traversal experiments

Self-contained O3 HIP/C++20 study. No engine headers, objects, libraries, or
build dependency. The normal study links only HIP; an optional separate
RCCL baseline uses the documented AMD-overlay workaround. Published
contribution slots are immutable through batch completion; output is separate. Source and timing conventions are unchanged
from the out-of-place study in commit `3283411`.

## Pruned search

Cooperative launch, its control, CTA election, and barrier-only measurements
are removed from the active code. Earlier results showed cooperative launch
losing badly even without `grid.sync()`, and election had no clear consistent
advantage. Historical results/code remain in git, not new default sweeps.

The first traversal pass tested 62 configurations in 119.9 s. Its few-lane
points and peer-major results were poor, so the default is now **30 cases**,
which completed in **49.5 s** on this machine. Peer-major remains available
explicitly for the requested experiment / future comparisons, not in the
routine sweep. Bounded payload lookahead is now a separate 32-case `prefetch`
suite; it does not expand the default traversal search.

## Readable paths

One launch function is selected at startup. Each measured operation begins
with the same GPU producer, followed by the selected function. There is no
size-dependent backend dispatch in that loop.

All traversal variants use the **same single-CTA ready barrier**, followed by
one gather kernel. Compare geometry explicitly, not different synchronization
schemes:

| Operation | Gather traversal |
|---|---|
| `pipe` | Original grid-stride data-major, canonical peer order |
| `chunk` | Contiguous CTA chunks, rank-rotated work assignment; canonical peers |
| `hoisted` | Same chunks, peer bases hoisted to shared memory; canonical peers |
| `phase` | Same chunks/hoist, cyclic dephased peer order |
| `peer-major` | Same chunks/hoist/dephasing, **peer outer / data inner** |
| `prefetch2` / `prefetch4` | Dephased data-major, 2/4 adjacent peer loads in flight |
| `hoist-prefetch2` | Canonical-order data-major, 2-peer lookahead control |
| `peer-prefetch2` / `peer-prefetch4` | Peer-major, 2/4 successive assigned vectors from one peer |
| `peer-reg1/4/8/32` | Peer-major with static register accumulators, one final store |
| `fused` | Historical 1×1024 decode reference; no separate barrier launch |
| `fill` | Producer-only timing control |

`chunk` separates work-assignment/layout effects from pointer hoisting and
peer dephasing. `hoisted` separates pointer-hoisting gains from dephasing.

### Dephasing and pointer hoisting

Contiguous chunks are rounded to whole-wave boundaries (64 float4 values).
CTA `b` on rank `r` owns logical chunk `(b+r) % blocks`. Its cyclic peer
traversal starts at `(r+b) % world_size`. Every wave keeps one peer address,
so float4 payload accesses remain coalesced; the rank-dependent order changes
which sources are requested together.

Only **addresses** are hoisted: the first `world_size` threads copy peer bases
plus the slab offset into a shared pointer table, then `__syncthreads()`.
At eight ranks this is 64 B of LDS. The data loop no longer globally loads
and waits for a peer pointer for every payload read. This is deliberately not
a dynamically indexed private pointer array that might spill to scratch.

Disassembly confirms LDS `ds_read_b64` pointer lookups and vector
`flat_load_dwordx4` payload loads. These original controls still have no
multiple-payload lookahead; the separate variants below add it explicitly.

### Flipped loop order

Data-major holds one float4 accumulator in registers:

```text
for each assigned float4:
    for each peer:
        read peer and add
    write output once
```

Peer-major streams the whole assigned chunk from a peer before moving on:

```text
for each peer:
    for each assigned float4:
        read peer and add to local output partial
```

Every thread owns its output elements, so no inter-peer CTA barrier is
required. The first peer initializes the partial; later peers read/update it.
The storage trade-off is explicit: **extra local output loads/stores per peer**,
versus one final output store in data-major. The whole arbitrary-size chunk
is not silently assumed to fit in registers. Few-lane / long-stream cases
were tested explicitly and lost; their larger sequential reads did not offset
the latency/partial-output costs.

### Numerics caveat

Rank-dependent peer order can change FP32 addition order. The synthetic
inputs here are small integer multiples of 1/32, so all sums are exact and
stale-data/address errors are checked bit-exactly on every rank. This does
**not** prove identical replicas for general FP32 inputs. Before production,
we must retain a common accumulation order or validate the changed numerics
against real canonical fixtures. No engine dispatch or numerics changed here.

## Method and reproduction

```sh
research/p2p-allreduce/build.sh
python3 research/p2p-allreduce/sweep.py --out temp/p2p-allreduce/traversal-pruned

# Pointer-only control at the useful geometry:
python3 research/p2p-allreduce/sweep.py --suite custom --worlds 4,8 \
  --bytes 32768,1048576 --points 8x256 \
  --backends chunk,hoisted,phase,peer-major --samples 15 \
  --out temp/p2p-allreduce/traversal-controls

# Explicit few-lane test, NOT part of the default grid:
numactl --cpunodebind=0 --membind=0 \
  temp/p2p-allreduce/study peer-major 8 1048576 1 64 64 5 0 graph
```

Arguments: `study operation ws bytes [blocks=0] [threads=256] [slots=64]
[samples=5] [device_base=0] [graph|eager]`. Bytes must be divisible by 16;
2–8 ranks; threads 64/128/256/512/1024. `blocks=0` is capped at 220.
Use the fresh `temp/` executable, not the historical tracked `./study` binary.

- All peer enables are serial and pre-allocation.
- Each GPU producer writes dirty L2 and changing rank/epoch/element values.
  Median latency includes this identical producer, not a subtracted estimate.
- Every rank/element of four slots is checked over eight changing-input
  replays, plus input preservation, final epochs, and the final timed output.
- Timer events disable system fences; no timer publishes data for the kernel.
- Three warm batches; median/min/p90 over timed samples; max-rank elapsed
  divided by slots. CPU wall latency is also recorded.
- The current ready helper reproduces the gfx90a cache-maintenance idiom:
  system release, relaxed peer polling, own-flag system acquire to invalidate
  reader L2. It is not a portable peer release/acquire pairing; see atomics.md.
- Default: ws=4/8, 32 KiB at 2×1024 / 8×256, 1 MiB at 8×256 / 32×256;
  `pipe`, `chunk`, `phase`, plus once-only fill/fused controls: 30 cases.
- Custom runs require explicit payloads AND geometry points. No accidental
  Cartesian sweep over hundreds of combinations. Raw CSV/logs, runtime,
  compiler/placement, and source/binary hashes are saved under `temp/`.

## Results

Release/O3, ROCm 7.1, gfx90a (104 CUs/GCD), graph mode,
`numactl --cpunodebind=0 --membind=0`. **No attention context**; 32 KiB
corresponds to 70B S=1 and 1 MiB to S=32. Times are producer-inclusive
median µs/op, not full-engine performance.

### Controls: 8×256, 64 slots, 15 samples

| ranks | payload | chunk | pointers only | pointers + dephase | peer-major + dephase |
|---|---|---:|---:|---:|---:|
| 4 | 32 KiB | 15.74 | 16.13 | 15.07 | 15.43 |
| 8 | 32 KiB | 27.32 | 21.18 | 28.79 | 33.53 |
| 4 | 1 MiB | 144.79 | 120.58 | 85.12 | 105.59 |
| 8 | 1 MiB | 302.60 | 126.64 | 125.44 | 212.34 |

At 1 MiB/ws=8, most of the gain is pointer hoisting, not dephasing. At ws=4,
dephasing adds a substantial gain. At decode size the ordering benefit is
not stable: three independent ws=8 confirmations at 160 slots / 15 samples
measured pointer-only **24.52 / 24.04 / 25.28** versus combined
**24.89 / 24.88 / 25.73**. Do not present the first pass's low combined number
as a consistent decode win.

### Few-lane cases: 1×64, 64 slots, 5 samples

| ranks | payload | original pipe | combined data-major | flipped peer-major |
|---|---|---:|---:|---:|
| 4 | 32 KiB | 117.60 | 44.94 | 104.98 |
| 8 | 32 KiB | 236.97 | 91.91 | 219.71 |
| 4 | 1 MiB | 3356.27 | 1047.60 | 2750.65 |
| 8 | 1 MiB | 6970.39 | 2189.02 | 6344.59 |

These points are removed from future default searches. Peer-major generally
loses to data-major at the same geometry; the nearest case was 1 MiB/ws=8,
32×256: 155.53 versus 152.37 in the first pass, not a clear win.

All 62 first-pass cases, 16 pointer controls, 30 pruned cases, six independent
decode confirmations, and four small/tail-chunk tests passed exact checks.
Nothing is adopted into production. The following lookahead experiments
were added only after the user's prefetch instruction.

## Bounded payload lookahead

Data-major loads two/four adjacent peers' float4 values into registers before
adding them in the original traversal order. Peer-major instead loads
two/four successive vectors assigned to a thread from the same peer, then
updates the same local partial outputs. Neither changes the reduction order
relative to its control. Full batches have no per-load tail checks; scalar
tails handle short peer groups / final data vectors.

The generated instructions are checked, not inferred from C++ source:

- Automatic outer-loop unrolling initially made a two-load variant issue
  eight loads. It is disabled around batches so the reported depth is real.
- Clang could interleave LDS pointer waits with flat payload loads. An empty
  asm **compiler dependency barrier** resolves the two/four pointer operands
  before payload issue; it emits no fence/invalidation instruction.
- Final disassembly shows 2/4 `flat_load_dwordx4` instructions before their
  payload completion wait. Peer-major also retains local output traffic.
- Metadata: no SGPR/VGPR spills or private scratch. Data-major 2/4 lookahead
  uses 20/32 VGPRs; peer-major uses 24/36. This is batching, not an asynchronous
  copy or a rolling pipeline across batches.

```sh
# 32 cases; geometries actually have >=4 vectors/thread for peer-major.
python3 research/p2p-allreduce/sweep.py --suite prefetch --samples 9 \
  --out temp/p2p-allreduce/prefetch

# Decode alternative: peer-major lookahead would be all-tail here, so omit it.
python3 research/p2p-allreduce/sweep.py --suite custom --worlds 4,8 \
  --bytes 32768 --points 2x1024 \
  --backends hoisted,phase,hoist-prefetch2,prefetch2,prefetch4 --samples 15 \
  --out temp/p2p-allreduce/prefetch-decode-wide
```

The bounded suite uses 32 KiB at 2×256 and 1 MiB at 8×256, with 64 slots,
ws=4/8. It completed in **54.2 s** for nine samples. Additional confirmations
at decode 2×1024 and bulk 32×256 took 15.4 / 30.7 s. All 32+10+16 cases,
five partial-peer/tail cases (ws=3), and eight independent decode repeats
passed exact checks.

### Data-major bulk: phase control versus 2/4-peer lookahead

Same O3/graph/NUMA-node-0 setup, no attention context (1 MiB = 70B S=32).
Producer-inclusive median µs/op, 64 slots; 9 samples at 8×256, 15 at 32×256:

| ranks | geometry | phase | prefetch2 | prefetch4 |
|---|---|---:|---:|---:|
| 4 | 8×256 | 81.18 | 82.27 | 82.11 |
| 8 | 8×256 | 133.81 | 152.42 | 125.24 |
| 4 | 32×256 | 83.74 | 82.64 | 81.61 |
| 8 | 32×256 | 147.93 | 135.97 | 151.55 |

No universal winning depth: four is better at ws=8/8 CTAs, two at 32 CTAs.
At 8 CTAs the pointer-only control was already 127.16, near prefetch4's
125.24. More outstanding remote requests do not guarantee a speedup.

### Peer-major bulk: lookahead helps but does not rescue the layout

At 1 MiB, 8×256, nine samples:

| ranks | peer-major | peer-prefetch2 | peer-prefetch4 | best data-major control/candidate |
|---|---:|---:|---:|---:|
| 4 | 105.47 | 102.11 | 103.60 | 81.18 |
| 8 | 210.36 | 174.07 | 173.05 | 125.24 |

At 32×256/ws=8 it regressed: 160.49 → 164.92 / 180.48. Partial-output
traffic remains; read lookahead alone is not sufficient.

### Decode: no stable benefit at the established wide geometry

32 KiB (S=1), 2×1024, 64 slots, 15 samples:

| ranks | phase | prefetch2 | prefetch4 |
|---|---:|---:|---:|
| 4 | 15.50 | 16.41 | 17.29 |
| 8 | 24.95 | 25.11 | 25.57 |

Independent 160-slot repeats are noisy and reverse the small differences.
The isolated canonical-order prefetch regression was not stable either; do
not infer a proven fabric mechanism from it. Keep non-prefetch controls and
choose any production candidate only after matched end-to-end validation.

## Register-backed peer-major

The earlier peer-major result was a memory-backed implementation, not a
verdict on peer-major itself. New static tiles hold 1/4/8/32 float4 running
sums per thread through all peers and store only once at completion. Larger
assignments iterate register-sized tiles; there is no partial-output pass.
Peer order is unchanged relative to peer-major. Full tiles avoid repeated
per-vector bounds masks; only ragged tails are masked.

```sh
# 40 selected cases, up to 4 MiB. 16 MiB deliberately excluded.
python3 research/p2p-allreduce/sweep.py --suite register --samples 9 \
  --out temp/p2p-allreduce/register-full
```

At 32 KiB/2×1024 a thread owns one float4; at 32 KiB/2×256 it owns four.
At 1 MiB/8×256 it owns 32; at 1 MiB/32×256, eight; at 4 MiB/32×256, 32.
These are output accumulators, not one temporary per peer. A 16 MiB full
chunk would require 128 float4/512 FP32 accumulators and is not tested.

ISA metadata is checked rather than assuming C++ arrays stay in registers:

| tile float4s | final VGPRs/thread | private scratch / spills |
|---|---:|---|
| 1 | 14 | zero |
| 4 | 52 | zero |
| 8 | 62 | zero |
| 32 | 204 | zero |

The first unconstrained 32-vector kernel spilled. The large tile now uses a
`__launch_bounds__(256,1)` budget on gfx90a (one wave/EU minimum), and its
wrapper rejects >256 threads, rather than silently launching a different
kernel. Smaller tiles allow up to 1024 threads. Source indexing is statically
unrolled; every result stays register-resident. The 204-VGPR tile can have
low occupancy: fitting is not equivalent to being fast.

### Final full-tile comparison

O3, graph mode, CPU/memory NUMA node 0, 64 slots, nine samples,
producer-inclusive median µs/op. No attention context; 32 KiB / 1 MiB /
4 MiB correspond to 70B S=1 / 32 / 128.

| ranks | payload / geometry | phase data-major | memory peer-major | register peer-major |
|---|---|---:|---:|---:|
| 4 | 32 KiB / 2×1024 | 15.64 | 17.87 | 15.98 (tile 1) |
| 8 | 32 KiB / 2×1024 | 26.26 | 30.73 | 33.52 (tile 1) |
| 4 | 32 KiB / 2×256 | 17.15 | 24.56 | 17.36 (tile 4) |
| 8 | 32 KiB / 2×256 | 27.60 | 41.12 | 23.41 (tile 4) |
| 4 | 1 MiB / 8×256 | 81.15 | 106.68 | 103.86 (tile 32) |
| 8 | 1 MiB / 8×256 | 125.89 | 212.31 | 202.45 (tile 32) |
| 4 | 1 MiB / 32×256 | 82.94 | 95.60 | 98.44 (tile 8) |
| 8 | 1 MiB / 32×256 | 150.94 | 155.55 | 157.23 (tile 8) |
| 4 | 4 MiB / 32×256 | 297.44 | 357.03 | 342.14 (tile 32) |
| 8 | 4 MiB / 32×256 | 579.37 | 558.81 | 576.50 (tile 32) |

At 1 MiB/8 CTAs, the tiled-eight version also lost: 103.58 / 211.14 at
ws=4/8. Keeping all 32 values, or iterating smaller register tiles, therefore
does not rescue that case. At ws=8/32 CTAs the data-major prefetch2 control
was 138.56, ahead of all peer-major variants.

The narrow decode tile-four benefit was confirmed in three independent
160-slot/15-sample runs at ws=8, 2×256: phase **31.41 / 31.92 / 30.99**,
register **29.56 / 29.10 / 30.11**. It is a candidate, not proof that this
beats every other data-major geometry or the whole engine.
At 4 MiB/ws=8/32×256 a separate 160-slot confirmation measured phase 572.95,
memory-backed peer-major 557.38, register tile-32 575.87. The small advantage
belongs to memory-backed traversal in this case, not register storage.

**Conclusion:** retain the tile-four decode candidate for now; register
peer-major is not a general bulk winner. Do not discard peer-major purely
because the first implementation used output memory, but do not keep its
entire tile-size Cartesian grid either. Future tests should be targeted.
No production kernels or dispatch rules have changed.

The final 40-case matrix completed in 78.2 s. Both initial/final matrices,
four final tail tests (ws=3), six decode confirmations, and three bulk
confirmations passed exact checks. The tests use exact synthetic operands;
real-model numerical validation is still required before adoption.

## Out-of-place RCCL baseline

`rccl.cpp` is a separate optional executable, not a vendor branch inside the
P2P dispatcher. It uses the identical study GPU producer, immutable input
slots and separate output, the same exact input/output validation, unfenced
per-device timing events, and all-rank batch completion. All counts are fp32
AllReduce sum; no hand-written P2P barrier is inserted around RCCL.

**Do not use Canonical's system RCCL:** its >=2-rank launch path is known to
segfault. Use the already validated AMD 7.1 overlay described in
[../rocm-overlay-method.md](../rocm-overlay-method.md) and
[../rccl-single-process/README.md](../rccl-single-process/README.md).
The executable uses `dladdr` and canonical paths to verify that the loaded
`ncclAllReduce` comes from the exact overlay library selected at build time.
It refuses the system library before any collective. API version alone is
not sufficient: the working and broken builds both report 2.27.7.

```sh
research/p2p-allreduce/build.sh --rccl
export LD_LIBRARY_PATH="$HOME/scratch/rocm-debs/amd-7.1/opt/rocm-7.1.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
numactl --cpunodebind=0 --membind=0 \
  temp/p2p-allreduce/study-rccl 8 64 15 0 both

# Two world sizes, raw logs/CSV and source/library/binary hashes:
research/p2p-allreduce/run_rccl.sh
```

`ALICE_RCCL_LIB_DIR` may select another existing verified AMD overlay.
The build does not download/extract packages or change the system.
The CLI is `study-rccl ws [slots=64] [samples=9] [device_base=0]
[graph|eager|both]`. It measures 32 KiB, 1 MiB, 4 MiB in one process,
reusing one communicator set. Initialization took 15.1 s / 22.1 s at ws=4/8,
not that cost for every size/mode. 16 MiB is excluded as requested.

HIP setup and peer enable are serial/pre-allocation. `ncclCommInitAll` creates
the single-process clique. Each operation groups submissions from all ranks
with `ncclGroupStart/End`; all capture streams are active before grouped
capture. Groups are per collective, not per batch, so producers remain
interleaved with collectives on each stream. Stop events are recorded only
**after GroupEnd has issued communication**. Warmup is grouped across all
ranks; no rank waits alone. Graphs are destroyed before normal communicator
cleanup. On an exception, communicators are aborted before freeing buffers.

### Vendor baseline results

AMD RCCL 2.27.7.70100 against system HIP 7.1, native O3 build,
`numactl --cpunodebind=0 --membind=0`, 64 slots, 15 samples. No attention
context; 32 KiB / 1 MiB / 4 MiB = 70B S=1 / 32 / 128. Median **producer +
collective µs/op**. Algorithm/protocol selection is RCCL's default, not forced.
The CSV's 0/0 geometry means library-managed, not zero GPU work.

| ranks | payload | RCCL graph | RCCL eager | matched P2P graph reference |
|---|---|---:|---:|---:|
| 4 | 32 KiB | 23.67 | 39.06 | 15.42 (phase, 2×1024) |
| 8 | 32 KiB | 39.09 | 81.10 | 26.03 (phase, 2×1024) |
| 4 | 1 MiB | 63.89 | 72.94 | 80.86 (prefetch4, 8×256) |
| 8 | 1 MiB | 79.05 | 101.75 | 122.68 (prefetch4, 8×256) |
| 4 | 4 MiB | 116.12 | 130.57 | 296.21 (phase, 32×256) |
| 8 | 4 MiB | 123.38 | 139.90 | 558.52 (peer-major, 32×256) |

The matched P2P confirmations use the same 64 slots / 15 samples and placement.
RCCL loses at decode size but wins at bulk sizes, especially 4 MiB (~2.55×
ws=4, ~4.53× ws=8 versus these flat references). Graph mode materially reduces
small-payload host submission cost. The eager control uses one host coordinator
with grouped submissions, not the engine's per-rank threaded schedule; primary
comparisons are graph-to-graph.

All 12 vendor rows and 12 matched P2P rows passed exact checks: inputs remain
unchanged, outputs correct through changing-input replays and final hot runs.
The wrong-library guard was tested and rejects the system RCCL safely.
Raw data: `temp/p2p-allreduce/rccl/` and `rccl-p2p-{decode,register,prefill,bulk}/`.

These results reopen vendor collectives as a serious **prefill** candidate,
not an instruction to replace the engine's decode path. No production
collective/dispatch/dependency changed, and real producer/cache/numerical
validation remains necessary before adoption.

## Counter visibility / synchronization skeleton

A separate persistent counter probe implements the user-requested
increment → wait for peers → increment loop, with no payload or per-step
launch/host synchronization. See [counter.md](counter.md) for code/timing
semantics, measured serial/parallel all-peer and neighbor controls, and
limitations. Build with `build.sh --counter`; run `run_counter.sh`.
It does not change the collective kernels or the engine.

## Files

- `study.cpp`: P2P setup, capture/replay, validation/timing; explicit startup selection.
- `rccl.cpp`: optional grouped out-of-place vendor baseline, no P2P dispatch changes.
- `run_rccl.sh`: validated overlay + NUMA placement + two reusable-comm sweeps.
- `counter.hip` / `counter.md` / `run_counter.sh`: persistent atomic visibility probe.
- `kernels.hip` / `kernels.hpp`: ready barrier and readable traversal variants.
- `build.sh`: standalone O3 build. `sweep.py`: bounded serial NUMA-pinned runs.
- `atomics.md` and existing coherency/peer-enable probes: historical evidence.
- Prior cooperative/election results: commit `3283411`.
