# Out-of-place P2P allreduce research

Current study: immutable contribution slots, produced-only epochs, separate
local output, HIP graph replay. The old event/RCCL/in-place study is historical
(git history); its barrier counts and conclusions do not describe this design.
The coherency evidence and probes remain in [atomics.md](atomics.md).

## Code structure

This study is **self-contained**: no production headers, objects, or engine
build dependency. Changing the engine cannot break the experiment. The
release/acquire protocol and float4 reduction are reproduced here; these are
source-equivalent research baselines, not the engine's compiled binary.

`study.cpp` selects an explicit launch function once at startup. Each slot:

```text
GPU producer writes this rank's slot
selected launch function
```

There is no size-dependent dispatch or second backend switch in that loop.
The launch functions in `kernels.hip` are intentionally direct:

| Operation | What is launched after the common producer |
|---|---|
| `fill` | nothing (producer-only control) |
| `barrier` | ready-barrier kernel only |
| `fused` | `k_wide`, 1 block × 1024 threads |
| `pipe` | `k_barrier` → `k_gather`, configurable geometry |
| `wide` | `k_wide`, configurable geometry |
| `elected` | `k_elected`, configurable geometry |
| `coop` | cooperative `k_coop`: leader barrier → `grid.sync()` → gather |
| `coop-wide` | cooperative launch of **the same `k_wide`** used by `wide` |

There is one force-inlined device `gather()` body. It is shared by the kernels
whose synchronization differs. There is **one pipeline gather kernel**, not
separate default/tuned copies: changing the pipeline geometry launches the
same `k_gather` through the same `pipe()` function. `pipe-tuned` and `engine`
backends have been removed.

### Synchronization and progress

- Sources use ordinary `hipMalloc`; flags use fine-grained device memory.
- Previous stream work has completed before an allreduce kernel starts. Thus
  any CTA can publish that rank's ready epoch, even if other CTAs have not run.
- `wide`: each CTA's thread 0 publishes, polls peers, and performs a system
  acquire; block-local sync precedes gathering. Redundant cache fences are a
  measurable cost, not a correctness requirement for additional computation.
- `elected`: a local atomic exchange elects a **resident** CTA to publish.
  Other CTAs wait for publication, poll peers, and acquire. No block-0
  scheduling assumption; oversubscribed grids do not wait on an unscheduled
  fixed leader.
- `coop`: block 0 performs the cross-GCD flag barrier, then every thread reaches
  the **single-device** cooperative grid barrier. Residency is checked with
  `hipOccupancyMaxActiveBlocksPerMultiprocessor` on every participating GCD.
  This is not `hipLaunchCooperativeKernelMultiDevice` or cross-device grid sync.
- All contributions remain immutable until **all ranks finish the batch**.
  No consumed epoch, read→write barrier, or input/output aliasing.

## Build and run

```sh
research/p2p-allreduce/build.sh
# Native hipcc -O3 C++20 gfx90a; g++ -O3 C++20 host; HIP runtime only.
# Writes the executable to temp/, never overwrites the old tracked binaries.

numactl --cpunodebind=0 --membind=0 \
  temp/p2p-allreduce/study pipe 8 1048576 32 256 160 21 0 graph
```

Arguments:

```text
study operation ws bytes [blocks=0] [threads=256] [slots=160]
      [samples=7] [device_base=0] [graph|eager]
```

- Bytes must be divisible by 16 (float4).
- `blocks=0`: ceil(payload float4s / threads), capped at 220; cooperative
  automatic grids are additionally capped at their residency limit.
- `fused` is always 1 × 1024; `barrier` always 1 × 32.
- `pipe`, `wide`, `elected`, and cooperative variants accept explicit geometry.
- Explicit oversized cooperative grids return **SKIP / exit 3**, never fall
  back to a regular launch.
- `fill` is a timing control; its producer always uses up to 220 × 256.
- Do not run the historical `research/p2p-allreduce/study` binary. Use the
  freshly built `temp/p2p-allreduce/study`.

## Measurement contract

The host enables all peer pairs serially **before any rank allocation**.
Each rank then owns its stream, slab, flags, output, and graphs. One host
coordinator submits every rank before waiting; no host API/barrier occurs
inside a graph's collective.

**Each measured operation includes an identical GPU producer plus the
collective.** The producer writes changing, exactly representable
rank/epoch/element values into that operation's slot. This exercises dirty
writer L2 and publication, rather than reading a host-prefilled buffer that
is already visible. `fill` and `barrier` controls are reported separately;
we do not subtract their times as though overlapping costs were additive.
The producer does not model GEMV scratch, KV updates, or their cache footprint.

- Default batch: 160 independent slots (70B's collective count per forward).
- Timing output is reused locally, as in the engine; published inputs are not.
- Validation uses separate outputs for four slots, NaN-poisons them, and checks
  **every element, slot, and rank** over eight changing-input replays. It also
  verifies untouched inputs and final epochs. The last timed output/input is
  checked after warm/hot runs.
- Three warm batches; median/min/p90 over independent timed batch samples.
- Each GPU is timed on its own event clock; report max-rank elapsed / slots.
  Timing events disable system fences, so they cannot publish each producer's
  dirty data in place of the flag protocol. CPU wall latency is also reported.
- `output_GBs = ws * bytes / time`: output-rate convention, **not physical
  XGMI bandwidth**. Flat gather reads every peer; do not compare this figure
  directly with ring bandwidth reported under another convention.
- Graph mode amortizes serial host launch skew over the batch. Eager mode is
  useful as a submission-overhead control, not the engine's threaded schedule.

No attention runs here; **attention context length is not applicable**.
Payload corresponds to `4 * S * hidden` bytes: for 70B hidden=8192, 32 KiB
means S=1, 1 MiB means S=32, and 4 MiB means S=128.

### Reproducible sweeps

```sh
python3 research/p2p-allreduce/sweep.py --suite baseline \
  --out temp/p2p-allreduce/standalone-baseline

# Same pipeline kernel, only grid changes:
python3 research/p2p-allreduce/sweep.py --suite geometry \
  --worlds 4,8 --bytes 1048576,4194304 --backends pipe \
  --blocks 32,220 --threads 256 --samples 21 \
  --out temp/p2p-allreduce/standalone-pipe-confirm

python3 research/p2p-allreduce/sweep.py --suite geometry \
  --worlds 4,8 --bytes 32768 --backends fused,pipe,wide,elected,coop,coop-wide \
  --blocks 2 --threads 1024 --samples 21 \
  --out temp/p2p-allreduce/standalone-decode-confirm
```

The runner serializes experiments, pins CPU and memory to NUMA node 0, and
saves CSV, raw stdout/stderr, compiler/hardware/placement metadata under
`temp/`. It stops on failures/timeouts; skips are recorded explicitly. The
old production-linked sweep results are **not used in the tables below**.

## Results: cleaned standalone harness

ROCm 7.1, Clang 21.1.8, gfx90a, **104 CUs/GCD**, graph mode, Release/O3,
`numactl --cpunodebind=0 --membind=0`, 160 slots, 21 timed samples. All rows
below are **producer + collective, median µs/op**, not whole-engine latency.
Context: no attention; 32 KiB = 70B S=1, 1/4 MiB = S=32/128.
All 144 baseline configurations and the 20 confirmation configurations pass
exact correctness checks.

### Decode-sized payload: ordinary wide launch beats cooperative launch

| ranks | fused 1×1024 | pipe 2×1024 | wide 2×1024 | elected 2×1024 | coop 2×1024 | coop-wide 2×1024 |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 18.28 | 18.39 | 14.64 | 14.53 | 54.51 | 53.83 |
| 8 | 35.95 | 32.11 | 27.58 | 27.84 | 92.45 | 73.66 |

Cooperative launch is supported, including capture/replay. Reported limits
are 832 blocks at 256 threads and 208 at 1024 threads on this kernel/device.
But it is expensive here. **`wide` and `coop-wide` execute identical kernel
code with identical geometry; only launch mode changes.** The cooperative
penalty therefore cannot be attributed solely to `grid.sync()` (which
`coop-wide` does not call). Pinpointing its runtime/dispatch cause would need
a separate trace, not speculation about the grid barrier's instructions.

An ordinary wide launch is valid specifically because producer computation
is in a preceding kernel and the slots are immutable. A future fused
projection/communication kernel has different producer dependencies and
might genuinely need grid or tile-level coordination.

### Wide payload: geometry, isolated with the same pipeline kernel

| ranks | bytes | pipe 220×256 | pipe 32×256 |
|---|---:|---:|---:|
| 4 | 1 MiB | 124.70 | 87.07 |
| 4 | 4 MiB | 373.09 | 299.14 |
| 8 | 1 MiB | 357.32 | 163.81 |
| 8 | 4 MiB | 967.15 | 534.99 |

This is now a controlled geometry comparison: **same barrier, same gather
kernel, same argument layout and launch function**. Fewer blocks improve
these cases substantially. More simultaneous remote requests are not
necessarily better; the result establishes a geometry effect, not its
precise fabric/cache mechanism.

## Recommendations / limits

1. Do not adopt cooperative launch merely to get more threads. Normal wide
   kernels already do that for immutable, externally produced contributions.
2. Candidate decode experiment: 2 CTAs × 1024 threads, produced-only barrier
   per CTA; election offers no stable clear advantage across these runs.
3. Candidate prefill experiment: tune pipeline block count rather than keeping
   the inherited 220-block cap. 32×256 is promising at ws=4/8 for 1/4 MiB.
4. Re-measure the fused/pipe crossover; the old in-place 64 KiB threshold is
   not justified by this new layout. The baseline includes 16/32/64/128 KiB.
5. These are research candidates, **not new engine dispatch rules or predicted
   tok/s gains**. End-to-end validation with real producer/cache behavior is
   required before changing production. There are no engine-code changes here.

## Files and historical probes

- `study.cpp`: setup, capture/replay, validation, timing; no engine dependency.
- `kernels.hip` / `kernels.hpp`: explicit launch functions, one gather body.
- `build.sh`: standalone O3 build; `sweep.py`: serial measured sweeps.
- `atomics.md`: release/acquire codegen and coherency evidence.
- `atomicprobe.hip`, `atomtest.hip`, `fence_sass.hip`: atomic/fence probes.
- `l2probe.hip`, `l2alloc.hip`, `l2pool.hip`, `bwprobe.hip`, `kbprobe.hip`:
  peer cache behavior.
- `clockprobe.hip`: cross-GCD clock offsets; event deltas here are per-device.
- `enableprobe{,2,3,4}.cpp`: pre-allocation peer-enable ordering investigation.
- `flagprobe.hip`: historical in-place/RCCL comparison, not this harness.
