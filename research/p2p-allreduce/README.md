# P2P allreduce study: cross-GCD sync strategies for TP decode (2026-10)

Question (user): the 70B TP=4 decode is ~44% AllReduce (~25 ms/tok of a
57 ms token; see the engine profile). RCCL owns that cost. How much can be
gained by throwing away RCCL, and which cross-GCD *synchronization* strategy
is the right replacement? This harness isolates the collective — N ranks, one
thread per GCD, each fills its io row (simulated forward) then allreduces — so
backends / world_size / message size can be swept without touching the engine.

## Backends

- **hostsync** — `hipMemcpyPeer` gather to rank 0 + local sum kernel +
  `hipMemcpyPeer` scatter, with `hipStreamSynchronize(0)` + a host atomic
  barrier for the inputs-ready / outputs-ready ordering. **This is the
  engine's current hand-rolled impl** (`src/gpu/model.cpp::allreduce`).
  Correct; host-bound.
- **event** — same data movement, but `hipEventCreateWithFlags(
  hipEventReleaseToSystem | hipEventDisableTiming)` + `hipStreamWaitEvent`
  for the cross-device dependency. **No `hipStreamSynchronize`, no host
  blocking** — the GPU-side stream dep replaces it. Note: `hipEventReleaseToSystem`
  is equivalent to the DEFAULT event flag — both do a system-scope fence
  (writeback + invalidation) on record. See §Coherency probes below.
  A host atomic only orders record-before-wait (dodges the stale-event
  re-recording race); it does not block on the GPU.
- **rccl** — `ncclAllReduce` (baseline). Needs the AMD-overlay librccl on
  `LD_LIBRARY_PATH` (Canonical's build segfaults ≥2 ranks). **Caveat: this
  harness segfaults RCCL at ≥2 ranks** (concurrent per-thread context
  creation — `research/rccl-single-process/` fixed it with a fuller serial
  setup not replicated here). RCCL numbers below are cited from that study +
  the engine profile, not re-measured here.
- **kernelp2p** — parallel flat allreduce: every rank reads all peers' io
  via kernel remote loads (the p2p.cpp-measured 134 GB/s SM path) into a local
  scratch, then writes its own io. Two cross-GCD barriers (inputs-ready +
  read→write) via system-fence events + `hipStreamWaitEvent`; io is
  `hipDeviceMallocFinegrained` (though coarse `hipMalloc` also works — see
  §Coherency probes). **Implemented.**
- **kernelp2p_oo** — out-of-place flat: read peers' io → write a SEPARATE io2
  (dst). No in-place read→write race within the kernel, so read+write fuse into
  one kernel (one fewer launch/allreduce). The read→write barrier still exists
  (relocated to before the next forward overwrites io) — same two barriers.
  Tests the io-reuse hypothesis: does out-of-place src/dst help?
- **kernelp2p_oo_pipe** — pipelined out-of-place: double-buffered io (io/io2
  alternate as src/dst) so the next-but-one forward reuses a buffer, and the
  reads-complete barrier is DEFERRED two iterations (lag-2) — non-blocking
  ("free") in steady state. Only the inputs-ready barrier still blocks per
  iteration. Tests whether pipelining recovers the barrier cost.
- **coop** *(TODO)* — `hipLaunchCooperativeKernelMultiDevice` + inter-device
  `grid_group::sync()` barrier. One kernel across devices, no events.

## Build & run

```sh
./build.sh                       # hipcc kernels.hip + g++ study.cpp -lamdhip64 -lrccl
./study <backend> <world_size> <bytes> <iters> [device_base]
# backend: hostsync | event | rccl
# rccl needs: LD_LIBRARY_PATH=~/scratch/rocm-debs/amd-7.1/opt/rocm-7.1.0/lib
```

## Results (single box, 8× MI250X GCD, gfx90a, no profiler; 100 iters)

| ws | bytes | hostsync µs | event µs | kernelp2p µs | kp_oo µs | kp_oo_pipe µs | GB/s (pipe) |
|---|--|--|--|--|--|--|--|
| 2 | 32 KB (decode row) | 67.5 | 53.7 | 46.3 | 49.0 | 47.9 | 1.4 |
| 2 | 16 MB (prefill chunk) | 907.6 | 920.7 | 225.9 | 192.5 | 191.7 | 175.0 |
| 4 | 32 KB | 103.8 | 112.8 | 108.5 | 106.9 | 100.0 | 1.3 |
| 4 | 1 MB | 248.3 | 237.2 | 165.7 | 174.4 | 173.0 | 24.2 |
| 4 | 16 MB | 2769.1 | 2735.7 | 1274.1 | 1276.8 | 1252.3 | 53.6 |

All rows pass the sum correctness check.

**RCCL baseline (cited, not re-measured here — harness segfaults RCCL):**
- 32 KB, 4 ranks, sync: **45.7 µs/call** (`research/rccl-single-process/`).
- 16 MB, 8 ranks, pipelined: **815 GB/s** effective (same study).
- Engine 70B TP=4 decode: ~25 ms/tok AllReduce = ~312 µs/call *under rocprofv3*
  (inflated; isolated ≈ the 45–90 µs range).
## Findings

1. **`kernelp2p` (flat parallel reads) beats the centralized-DMA backends at
   every size ≥ 256 KB, dramatically at prefill sizes**: ws=2 16 MB → 223 µs
   (150 GB/s) vs ~910 µs (37 GB/s) for the DMA backends — **4× faster**; ws=4
   16 MB → 1274 µs (53 GB/s) vs ~2750 µs (24 GB/s) — **2× faster**. The SM
   remote-load path (parallel, every rank reads concurrently) scales where the
   rank-0 gather/scatter (serial, one GCD's DMA engines) cannot. The
   the system-fence events (default or `hipEventReleaseToSystem`) between
   allreduces invalidate each rank's L2, flushing stale peer lines — see
   §Coherency probes for the full mechanism.

2. **But `kernelp2p` still loses to RCCL at every size.** At decode (32 KB,
   ws=4): ~108 µs vs RCCL's cited 45.7 µs — **RCCL ~2× faster**. At prefill
   (16 MB, ws=4): 53 GB/s vs RCCL's 815 GB/s (8-rank pipelined) — **RCCL ~15×
   faster**. Two structural reasons:
   - *Decode (32 KB)*: the allreduce is **sync-bound, not bandwidth-bound**.
     `kernelp2p`'s two barriers (inputs-ready + read→write) cost ~100 µs of
     event/atomic overhead regardless of the 32 KB of data moved (~0.5 µs of
     reads). RCCL's single in-kernel collective avoids the per-rank host-event
     churn. The rccl study's "~6 µs" projection was the *data-movement* floor
     only — it ignored barrier latency, which dominates at this size. **The
     projection does not survive contact with the sync cost.**
   - *Prefill (16 MB)*: the flat allreduce is **O(N²) traffic** (every rank
     reads every peer's full buffer) vs RCCL's ring at **O((N−1)/N)** (each
     link carries 1/N). A flat read cannot pipeline-match a ring at scale;
     this is why RCCL hits 815 GB/s while the flat read plateaus at ~150 GB/s
     even at ws=2.

3. **`event` vs `hostsync`** (sync mechanism, independent of data movement):
   `event` beats `hostsync` at ws=2 / small sizes (32 KB: 54 vs 67 µs, −20%)
   — removing the host `hipStreamSynchronize` wins when the host-sync block
   exceeds the event-API overhead. The gain vanishes at ws=4 (tied): the
   event backend's per-call API overhead (`(ws+1)` records + `~2·ws` waits
   ≈ 36 µs at ws=4) offsets the two `hipStreamSynchronize` calls (~20 µs) it
   removes. It trades *host blocking* for *host API-call overhead*.

4. **Coherency is event-fence-driven, NOT uncached-read-driven.** (Corrected
   from an earlier claim that peer reads are uncached — they are NOT.) All
   backends pass correctness at ws=2/4. The kernel-read path needs
   `hipDeviceEnablePeerAccess` — **enabled BEFORE any allocation**
   (corrected 2026-10, `enableprobe*.cpp` + engine bisect: post-load enable
   deterministically aborts the process via the HSA
   `MappedHandleAllowedAgent` assert on state created during load; the old
   "enable serially, after all allocs" advice is WRONG — it must be serial,
   owner-thread, and PRE-alloc, with all ranks held until every enable is
   done so no hipMalloc overlaps an enable. Ruled out as triggers: plain
   hipMalloc, rocBLAS handle, THP-mmap pageable H2D, staging munmap,
   worker-thread fine-grained alloc. `hipHostRegister` on a THP mmap also
   deterministically breaks enable, but the engine never registers). The
   coherency
   mechanism is the **system-scope fence on event record** (default events
   already do this; `hipEventReleaseToSystem` is the explicit form): writeback
   + **invalidation** of the recording device's L2. Between allreduces, every
   rank records events → every rank's L2 is invalidated → stale peer-buffer
   lines are flushed → the next peer read goes to VRAM (fresh). This is why
   coarse `hipMalloc` io works (verified `KP_IO_COARSE=1`, 32 KB–16 MB, ws=2/4,
   2000 iters, zero staleness) — fine-grained allocation is unnecessary and
   ~5–10% slower (it bypasses L2 on local access too). See §Coherency probes
   for the verification that peer reads ARE L2-cached and peer writes do NOT
   invalidate the target's L2.

5. **Buffer layout & pipelining (the io-reuse question).** Out-of-place src/dst
   (`kernelp2p_oo`) and double-buffered pipelining (`kernelp2p_oo_pipe`) both
   help, but only marginally:
   - `kernelp2p_oo` fuses read+write into one kernel (saves one launch/allreduce):
     ws=4 32 KB 121→107 µs (−12%); ws=2 16 MB 226→192 (−15%, bigger at large
     sizes where the extra kernel pass matters). Confirms the io-reuse forces
     an *extra kernel*, not an extra barrier — the read→write barrier is
     *relocated* to before the next forward, not removed.
   - `kernelp2p_oo_pipe` additionally frees the 2nd barrier's GPU block
     (lag-2 reads-complete is non-blocking): ws=4 32 KB 107→100 (−7%); ~0 at
     large sizes (kernel-dominated).
   - **The combined win is ~−17% at decode size, and the floor is neither
     io-reuse nor the barrier count — it's per-iteration host API overhead.**
     At ws=4 32 KB the three are 100–121 µs but the GPU work (kernels + GPU
     barrier stalls) is ~20 µs; the other ~80–100 µs is `hipEventRecord` /
     `hipStreamWaitEvent` (~10 calls/iter) + host-atomic spins. `kernelp2p_oo_pipe`
     keeps the *same* host-API call count (it only removes the GPU stall), so
     it can't break that floor. Out-of-place is a prerequisite for pipelining,
     but neither addresses the dominant cost. Only **device-side sync** (one
     in-kernel collective, ~zero per-iteration host events — what RCCL does)
     removes the host-API churn. This is the real reason RCCL (45.7 µs) beats
     the flat (100–121 µs): not the ring's O(N) traffic at 32 KB (data is
     tiny), but the single in-kernel collective vs N per-iteration host-event
     barriers.

## Conclusion

**Throwing away RCCL is not worth it.** Every hand-rolled variant —
centralized DMA (`hostsync`, `event`) and flat parallel reads (`kernelp2p`) —
loses to RCCL's ring at every size. `kernelp2p` is the best of the three and
still 2× slower than RCCL at decode (sync-bound) and 15× slower at prefill
(ring pipelining). The rccl study's "~6 µs hand-rolled vs 45 µs RCCL"
projection was the data-movement floor only; the real allreduce is
barrier-bound at decode sizes, where RCCL's single in-kernel collective beats
N per-rank host-event barriers. **Recommendation: keep RCCL; revert the
engine's hand-rolled `allreduce` to `ncclAllReduce`.** The one real engine
win identified earlier stands separately and is independent of this study:
drop the per-call `hipStreamSynchronize` after `ncclAllReduce` (RCCL is
correct without it on the shared stream) — already applied.

The only hand-rolled path that could plausibly compete with RCCL would be a
**ring/tree** (reduce-scatter + all-gather) with kernel P2P and device-side
barriers — i.e. reimplementing RCCL. Not worth it; RCCL's ring is already
tuned and works (via the overlay).

## Coherency probes (gfx90a cross-GCD L2 behavior)

Three standalone probes (`l2probe.hip`, `bwprobe.hip`, `clockprobe.hip` in
this dir) verify the actual memory semantics that the allreduce backends
rely on. These corrected two earlier errors in this study.

### Event flag inventory (`hipEventCreateWithFlags`)

| flag | scope | on record |
|---|---|---|
| `hipEventDefault` (0x0) | **system** | system-scope fence: **writeback + invalidation** (the DEFAULT!) |
| `hipEventReleaseToDevice` (0x40000000) | device | device-scope release (same-GPU only, cheaper) |
| `hipEventReleaseToSystem` (0x80000000) | system | explicit system-scope release (same as default) |
| `hipEventDisableSystemFence` (0x20000000) | none | DISABLES the default system fence (timing-only) |

Key: the `hipEventDisableSystemFence` comment — *"Disable performing a system
scope sequentially consistent memory fence... avoiding the cost of cache
writeback and invalidation"* — reveals that **default events already do a
system-scope fence (writeback + invalidation)** on record. `hipEventReleaseToSystem`
is the explicit form; it is NOT required. There is **no "to peer" scope** —
only system (cross-device + host) or device (same-GPU).

### Probe 1: do peer writes invalidate the target's L2? (`l2probe.hip`)

Design (the user's register approach): GCD0 buf0=100, GCD1 buf1=200. Each
reads the peer's buffer (saves the value), writes to the peer's buffer, syncs,
then reads its OWN buffer locally and compares against the saved peer value.

```
GCD0: 8192 mismatches -> STALE (local L2 not invalidated)
GCD1: 8192 mismatches -> STALE (local L2 not invalidated)
buf0[0]=200 (VRAM fresh)   buf1[0]=100 (VRAM fresh)
```

**Peer writes land in VRAM (D2H sees the fresh value) but do NOT invalidate
the target's L2.** The target's local L2 read returns the stale initial value.
`__threadfence_system` on the writer flushes the writer's L2 (writeback) but
does nothing to the target's L2. This confirms: **remote writes to peer VRAM
are useless for the target's subsequent local reads** without an intervening
L2 invalidation on the target. The flat allreduce (remote reads + local
writes) avoids this hazard entirely; a reduce-broadcast (rank 0 remote-writes
to all) would be broken unless each target performs a system-scope ACQUIRE
before its local read. **Update:** `atomics.md` shows exactly how to get that
invalidation — `__hip_atomic_load(ACQUIRE, SYSTEM)` emits `buffer_invl2`
(full L2 invalidate), and with a release/acquire atomic flag protocol the
l2probe scenario goes fully FRESH.

### Probe 2: are peer reads uncached? (`bwprobe.hip`)

Design: read the same peer buffer 100000× with `volatile` loads (prevents
compiler CSE), measure bandwidth. If uncached, ~= fabric (~134 GB/s); if
L2-cached, ~= local L2 (much higher).

```
local read  (GCD0 -> buf0):      157.3 GB/s
peer read   (GCD0 -> buf1@GCD1): 159.0 GB/s
peer read   (GCD1 -> buf0@GCD0): 157.5 GB/s
```

**Peer reads ARE L2-cached** — same bandwidth as local reads (~157 GB/s,
L2-latency-limited, NOT fabric-limited). This **corrects the earlier claim**
in this study that "peer reads appear uncached on the reader." They are not.

So why does `kernelp2p` pass correctness (2000 iters, zero staleness) if peer
reads are L2-cached? Because the **system-scope event fences between
allreduces invalidate each rank's L2** (the default event's writeback +
invalidation), flushing stale peer-buffer lines before the next read. The
coherency is event-fence-driven, not uncached-read-driven.

### Probe 3: cross-GCD synchronized clock (`clockprobe.hip`)

`wall_clock64()` (device-side `__ockl_steadyctr_u64`) is a 25 MHz (40 ns/tick)
steady counter, queried via `hipDeviceAttributeWallClockRate` (kHz). It is
coherent **within a card** (GCD0↔GCD1: ~0 offset, ~4 µs flag-propagation
SOL) but has a **fixed ~17 ms offset between cards** (GCD0↔GCD2) with a slow
~µs/sec drift. For cross-card timing, calibrate the offset once and subtract.
Note: the ~4 µs SOL measured was via a **host-pinned flag** (PCIe to CPU RAM,
coherent by nature) — NOT device VRAM peer access. The VRAM peer-access
latency is a separate (unmeasured) quantity.

## Files

- `study.cpp` — host orchestrator (N threads, backends, timing, correctness).
- `kernels.hip` / `kernels.hpp` — `fill_value` (simulated forward) + `sum_reduce`
  + `flat_read` / `flat_write` / `flat_read_write` (kernelp2p variants).
- `build.sh` — hipcc kernels + g++ study (-lamdhip64 -lrccl).
- `l2probe.hip` — peer-write L2-invalidation test (§Coherency probes).
- `bwprobe.hip` — peer-read cached/uncached bandwidth test.
- `clockprobe.hip` — cross-GCD `wall_clock64()` coherence + latency probe.
- `kbprobe.hip` — kernel-boundary vs system-fence-event L2 invalidation.
- `atomtest.hip` / `atomicprobe.hip` — `__hip_atomic_*` codegen + runtime
  probes (see `atomics.md`).
- `flagprobe.hip` — device-side cross-GCD barrier + write-scatter AND
  read-gather (scalar/float4) allreduce variants with fine-grained
  device-VRAM flags (host-pinned as reference mode); see `atomics.md` for
  measured numbers. **Breaks the host-API floor: 1.8 µs barrier at ws=2;
  read-gather-f4 allreduce 8.1 µs ws=2 / 17.9 µs ws=4 at 32 KB (2.6-5.7x
  faster than RCCL's 45.7 µs).**
- `atomics.md` — `__hip_atomic_load`/`__hip_atomic_store` report: release/
  acquire semantics on gfx90a, the fine-grained flag protocol, and the
  device-side barrier results.
- `enableprobe{,2,3,4}.cpp` — the `hipDeviceEnablePeerAccess` bisect series:
  post-load enable deterministically aborts (HSA `MappedHandleAllowedAgent`
  assert); enable must be PRE-alloc, serial, owner-thread. Landed in the
  engine's `Model::load`.
- `fence_sass.hip` — SASS proof that `__threadfence_system()` emits
  `buffer_wbl2` (L2→HBM writeback; load-bearing because fabric reads are
  served by HBM, not owner L2).
- `l2alloc.hip` — peer reads allocate in the reader's L2 (cold-L1 blocks on
  other CUs hit at L2 speed after one block warms).
- `l2pool.hip` — random-cycle pointer chase: one CU effectively sees ~2 MiB
  of L2 (SE-local allocation), NOT the pooled 8 MiB; invalidation is
  broadcast (see `atomics.md`).
