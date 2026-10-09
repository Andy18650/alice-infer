# Persistent atomic-counter visibility probe

Question: how quickly can a rank atomically increment a device counter, see
its peers do the same, and advance again? This isolates a synchronization
skeleton without payload, matrix work, collective libraries, launches/events
inside the measured loop, or host participation between increments.

## Kernel

One fine-grained device-memory uint64 counter per rank, initially zero.
Each rank launches one 64-thread block on its own stream. Per step:

```text
thread 0: atomic_fetch_add(my_counter, 1, SYSTEM scope)
wait until the required peer counters are >= this step's expected value
advance to the next step
```

Counters are never cleared inside the loop. A peer may already be ahead, so
poll `>=`, not equality. Resets between samples complete on ALL devices
before ANY new kernel is launched. Every final device counter and the
kernel-reported value must equal `warmup + target` exactly.

- `serial`: only thread 0 is active; polls every other rank in order.
- `parallel`: thread 0 increments, CTA-local sync, one lane per peer polls,
  CTA-local sync. Unused lanes still participate in the block barriers.
- `neighbor`: only thread 0 active; waits for the previous rank in the natural
  cycle. **Not an all-peer barrier.** This is a ring-like readiness control,
  not a ring AllReduce implementation.
- Increment order: `relaxed` or `release`, always SYSTEM scope.
- All polls are RELAXED/SYSTEM. There is no associated payload to acquire and
  no own-flag acquire/L2 invalidation masquerading as counter visibility.

Each rank increments its OWN counter. This does not measure remote atomic
RMW or a contended global counter. It measures local publication plus peer
observation and progress. The real engine currently uses release stores,
not fetch-add, so this is a distinct primitive baseline.

## Timing and codegen

64 warm steps establish progress after all GPUs have been launched. Thread 0
drains its own warmup increments, reads `wall_clock64`, runs `target` more
steps, waits until its own counter actually reaches `warmup + target`, then
reads the clock again. This measures completion, not just queued atomic issue.
Clock
rate comes from `hipDeviceAttributeWallClockRate` (kHz; 25,000 on this box).
Elapsed ticks are subtracted **on the same device**. Cross-card clock offsets
are not compared or assumed coherent; the old clockprobe comment is not a
valid cross-card timing assumption.

Report max-rank elapsed per sample, median/min/p90 over samples, and elapsed /
target. The percentile describes whole runs, not an individual-step latency
histogram. CPU wall time is reported separately and includes launch, warmup,
and completion waits. Warmup=0 explicitly exposes startup skew.

Metadata/disassembly verifies:

- native `flat_atomic_add_x2` (discarded return value, no extra validation
  round trip inside the timed loop);
- peer polls use `flat_load_dwordx2 ... glc`;
- release variants emit `buffer_wbl2`; relaxed ones avoid that release cost;
- all six specializations have zero private scratch and zero register spills.

Peer pointers are preloaded before timing. Serial pointers are statically
indexed/unrolled; parallel/neighbor keep a single selected pointer, not a
runtime-indexed private pointer array that could spill.

## Build / run

```sh
research/p2p-allreduce/build.sh --counter
numactl --cpunodebind=0 --membind=0 \
  temp/p2p-allreduce/counter 8 10000 9 0 parallel relaxed 64

# 28 bounded cases, 1/2/4/8 ranks and one cross-card pair:
research/p2p-allreduce/run_counter.sh
```

```text
counter ws [target=10000] [samples=7] [device_base=0]
        [serial|parallel|neighbor] [relaxed|release] [warmup=64]
```

`target` is the number of TIMED increments per rank, not the sum across ranks;
final count includes warmup. Single-rank runs are local primitive controls.
Pre-allocation peer enable is serial/owner-thread. This code is standalone
HIP C++20/O3/gfx90a and does not depend on production or RCCL. The runner
pins CPU and memory to NUMA node 0, saves raw CSV/logs and source/binary
hashes under `temp/p2p-allreduce/counter-results/`, and stops on timeout/failure.
There is no attention/model context or payload size in this experiment.

## Initial measurements

ROCm 7.1, gfx90a, native O3, CPU+memory NUMA node 0. No payload/attention
context. 10,000 timed increments + 64 warm steps, nine samples. Median
max-rank **microseconds per increment/wait cycle**:

| ranks, base=0 | serial relaxed | serial release | parallel relaxed | parallel release | neighbor relaxed | neighbor release |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.124 | 0.295 | 0.109 | 0.370 | 0.109 | 0.332 |
| 2 | 0.912 | 1.097 | 0.977 | 1.237 | 0.837 | 1.000 |
| 4 | 2.236 | 2.457 | 1.217 | 1.512 | 0.996 | 1.159 |
| 8 | 5.581 | 5.654 | 1.808 | 2.075 | 0.998 | 1.163 |

Two-rank base=0 uses GCDs 0/1 (intra-card); base=1 uses GCDs 1/2 (cross-card):
serial relaxed/release **1.005 / 1.189**, parallel **1.153 / 1.405** us/cycle.
This is that particular pair, not every physical link's measured latency.

At eight ranks, 10,000 parallel-relaxed steps take 18,080.80 us. Scaling
checks (three samples) give 1,000 steps 1,802.08 us and 100,000 steps
179,983.52 us: ~1.80 us per cycle, consistent with device-loop timing
rather than a launch-dominated measurement. All 28 primary cases, two scaling
checks, three no-warmup target=1 cases, and three smoke modes finish exactly.

## Interpretation / limits

Final data is under `temp/p2p-allreduce/counter-final/` (the earlier
counter-results files precede the explicit warmup/final RMW drain).

- Serial polling is a major part of the all-peer progress cost at ws=4/8.
  Parallel polling substantially lowers this particular readiness skeleton.
- Neighbor readiness is cheaper than a global barrier and is the more relevant
  control for ring-like stages, but ring data forwarding is not implemented.
- These are **not one-way propagation measurements**. They include local RMW,
  polling, slowest-rank progress, and (parallel mode) block synchronization.
  Do not simply subtract the single-rank time and label it fabric latency.
  The single-rank control is steady atomic throughput, not an isolated RMW
  round-trip latency; only the warmup and final outstanding increments drain.
- Release overhead is measured with essentially no dirty payload cache.
  Real producer data requires its own ordering/coherency and can make release
  writeback more expensive; this probe cannot replace a payload correctness
  test or justify dropping an acquire.
- No bytes are moved or reduced. A ring's data traffic, routing, reduction,
  buffer-reuse protocol, and number of dependent rounds remain additional
  costs. A synchronization-heavy design can use these cycles as a skeleton
  budget, not as a claimed AllReduce latency or a universal hard lower bound.

No production kernels or dispatch rules were changed.
