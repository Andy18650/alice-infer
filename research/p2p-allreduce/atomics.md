# `__hip_atomic_load` / `__hip_atomic_store` — investigation report

(2026-10, ROCm 7.1 system headers + hipcc, gfx90a. Probes: `atomtest.hip`,
`atomicprobe.hip` in this dir. Sources: local headers, LLVM reviews, measured
SASS + runtime behavior.)

## What they are

**Clang compiler builtins** (note the double underscore — there is no public
`hip_atomic_load`/`hip_atomic_store` function in the ROCm 7.1 headers).
Introduced in 2021 by two LLVM reviews:

- D87858 — scoped HIP atomic ops (the scope parameter).
- D114553 — `__hip_atomic_load`, `__hip_atomic_store`,
  `__hip_atomic_compare_exchange_weak` builtins with a memory-order parameter.

They are the primitives HIP's own headers are built on:
`/usr/include/hip/amd_detail/amd_hip_atomic.h` implements `atomicCAS`,
`atomicAdd`, `atomicExch`, etc. as wrappers around
`__hip_atomic_load` / `__hip_atomic_compare_exchange_strong` /
`__hip_atomic_store`. RAJA and tilelang also call them directly.

Signatures (device code):

```cpp
T    __hip_atomic_load (T* addr, int mem_order, int mem_scope);
void __hip_atomic_store(T* addr, T val, int mem_order, int mem_scope);
```

- `mem_order`: `__ATOMIC_RELAXED`, `__ATOMIC_ACQUIRE`, `__ATOMIC_RELEASE`,
  `__ATOMIC_ACQ_REL`, `__ATOMIC_SEQ_CST`.
- `mem_scope`: `__HIP_MEMORY_SCOPE_SINGLETHREAD`(1), `WAVEFRONT`(2),
  `WORKGROUP`(3), `AGENT`(4 = this GCD), `SYSTEM`(5 = cross-device + host).

## What they do (measured, gfx90a)

Codegen (`atomtest.hip`, `-O2`):

| call | SASS |
|---|---|
| load RELAXED SYSTEM | `global_load_dword ... glc` |
| load ACQUIRE SYSTEM | `global_load_dword ... glc` + `s_waitcnt` + **`buffer_invl2`** + `buffer_wbinvl1_vol` |
| store RELEASE SYSTEM | `global_store_dword` + `s_waitcnt` (writeback ordering) |

- `glc` = globally-coherent load: bypasses L1, reads the coherence point.
- **ACQUIRE at SYSTEM scope emits `buffer_invl2` — a full L2 invalidate.**
  This is the acquire side of the release/acquire pair on gfx90a: release =
  writeback (what `__threadfence_system()` does), acquire = L1+L2 invalidate
  (what `__threadfence_system()` does NOT do).

Runtime proof (`atomicprobe.hip`): the exact scenario that defeated
`l2probe` (GCD0 primes its L2 with buf0=100; GCD1 peer-writes buf0=200; GCD0
reads buf0 locally). With the flag protocol implemented as
`__hip_atomic_store(RELEASE, SYSTEM)` / `__hip_atomic_load(ACQUIRE, SYSTEM)`:

```
A: normal local read (control)    GCD0 mism=0  GCD1 mism=0 -> FRESH
B: atomic load RELAXED SYSTEM     mism=0/0 -> FRESH
C: atomic load ACQUIRE SYSTEM     mism=0/0 -> FRESH
```

All 8192 elements fresh in every mode — including the plain normal load
(control), because the flag's ACQUIRE already invalidated the reader's whole
L2 (`buffer_invl2`), so the subsequent normal load misses to VRAM. Compare
`l2probe` (volatile flags + `__threadfence_system`): 8192/8192 STALE.

**This resolves the l2probe finding:** a peer write IS usable by the target's
local reads if the target performs a system-scope ACQUIRE afterwards. The
release/acquire pair is the correct cross-GCD producer/consumer primitive on
gfx90a; `__threadfence_system` + volatile flags is only the release half.

## Can we use them?

Technically yes — they compile and work under our hipcc (verify with
`__has_builtin(__hip_atomic_load)`, as HIP's own headers do). Legally-grey:
double-underscore names are a reserved/internal namespace with no stability
guarantee. In practice they have been stable since 2021, are the foundation
of every public `atomic*` function in the HIP headers, and are used directly
by RAJA/tilelang. Removal would break HIP itself, so the real risk is a
signature change, which `__has_builtin` + a thin wrapper localizes.

Public alternatives and why they don't cover this:
- `atomicExch`/`atomicCAS`/`atomicAdd(p,0)` (+ `_system` variants): RMW-only,
  and the classic ones are relaxed — they don't give a pure acquire load that
  invalidates L2. (An `_system` RMW does fence, but pays RMW cost and still
  isn't a plain load.)
- `volatile` + `__threadfence_system()`: release-only — proven insufficient
  (l2probe).
- C++ `std::atomic_ref` / `hip::atomic_ref`: not present in ROCm 7.1 device
  headers.

## Should we use them?

**Yes, for the multi-GCD allreduce — they are the missing piece.** The
p2p-allreduce study floored at ~80–100 µs/iter of host-API overhead
(hipEventRecord/hipStreamWaitEvent round trips between ranks), because every
cross-GCD handshake went through the host. System-scope atomic flags enable a
**fully device-side barrier**: ranks spin on `__hip_atomic_load(ACQUIRE,
SYSTEM)` of flag words (host-pinned or fine-grained) and set them with
`__hip_atomic_store(RELEASE, SYSTEM)` — correct coherency (proven above) with
zero host API calls. That opens the single-kernel/persistent-kernel allreduce
that the study concluded was out of reach.

Cautions:
- `buffer_invl2` invalidates the whole 8 MB L2 — fine once per allreduce,
  ruinous per-element. Flags only; bulk data reads can stay normal loads
  (they're covered by the acquire's invalidate).
- Writer still needs the release half (release store or
  `__threadfence_system()` before the flag store) so its data is written
  back before the flag is visible.
- Keep usage behind a small wrapper with `__has_builtin` guards and a
  volatile+event fallback; note it in `docs/amd-gpu-notes.md`.

## gfx90a memory-hierarchy details (why the protocol is shaped this way)

Two refinements from microarchitecture, both probe/SASS-verified
(`fence_sass.hip`, `l2alloc.hip`):

1. **Fabric reads are served by HBM, not by the owner's L2.** gfx90a's L2 is
   a shader-side writeback cache, NOT a memory-side coherence point (unlike
   NVIDIA, whose L2 is associated with the memory controllers and serves
   NVLink reads). Consequence: a peer's remote read of my buffer goes to my
   **HBM** — so the writer's release must actually **write back** dirty L2
   lines, not just order them. SASS proof: `__threadfence_system()` emits
   **`buffer_wbl2`** (explicit L2→HBM writeback); the system-scope release
   store path also emits `buffer_wbl2`. This is why the release half is
   load-bearing on AMD in a way it isn't on NVIDIA.
2. **The reader-side invalidate covers 4 SE-private L2 slices + per-CU L1s —
   by broadcast, not by pooling.** Each GCD has 4 shader engines with their
   own L2 slices plus per-CU L1s. Probes (`l2pool.hip`, `l2alloc.hip`):
   - `l2pool.hip`: a random-permutation pointer chase from ONE CU (fully
     serialized dependent loads = true latency) over warmed buffers, with an
     evicted HBM control. Result: warm latency climbs with working-set size
     (1 MiB 188.6, 4 MiB 242.6, 6 MiB 265.2, 8 MiB 281.4 vs HBM 283 ns) —
     **one CU effectively sees only ~2 MiB of L2** (one SE's slice), NOT the
     full 8 MiB. The naive "L2 is pooled across SEs" hypothesis is REFUTED
     for allocation/latency purposes. (First version with a regular 16 KB
     stride was defeated by an apparent stride prefetcher; the random
     Hamiltonian cycle is load-bearing.)
   - `l2alloc.hip`: peer reads DO allocate in the reader's L2 (cold-L1 blocks
     on other CUs read at L2 speed after one block warms, 167–184 ns vs
     ~500–900 ns fabric). So L2 invalidation is genuinely required.
   - **Invalidation is broadcast even though allocation is SE-local**: the
     pipeline experiment (64 gather blocks across all 4 SEs, 4 MB × 100
     iters, 0 mismatches) had stale peer lines in every SE's slice (each
     block allocates into its own slice), yet one thread's `buffer_invl2`
     cleared them all. Cache-control ops broadcast to all slices; they are
     not address-routed lookups. This — not cache pooling — is what makes a
     single-thread acquire sufficient on this GCD.
   - Per-CU L1s are cleared at kernel launch (consistent with kbprobe:
     no-fence events leave the WRITER's dirty L2 unwritten-back, so the
     reader misses to stale HBM — the observed "garbage" was exactly the
     0x64646464 memset pattern).
   - Practical capacity note: budget ~2 MiB of L2 working set per CU, not
     8 MiB.

   The complete pipeline chain, every link now evidence-backed: fill (dirty
   L2) -> flagbar release (`buffer_wbl2`, device-wide -> HBM; flag store)
   -> peer poll sees flag -> peer acquire (`buffer_invl2`, device-wide) ->
   gather launch (per-CU L1s invalidated) -> gather reads miss all the way
   to producer HBM (fresh).

## Verification artifacts

- `atomtest.hip` — codegen probe (the SASS table above).
- `atomicprobe.hip` — runtime probe (release/acquire flag protocol vs stale
  L2; the l2probe scenario with system-scope atomic flags).

## Device-side barrier: fine-grained device flags vs host-pinned (`flagprobe.hip`)

`flagprobe.hip` implements the full protocol as a reusable device-side
barrier + write-scatter allreduce, with the flag words in **fine-grained
device VRAM** (each rank's flag in its own GCD,
`hipExtMallocWithFlags(..., hipDeviceMallocFinegrained)`) and a `host` mode
(`hipHostMalloc`) kept as a reference to verify the objection that
host-pinned flags are the wrong pattern here (2 CPUs: GCDs 0–3 on NUMA
node 0, GCDs 4–7 on node 1 — a host page is local-node for only half the
GCDs, and every poll is a PCIe round trip).

Protocol per barrier: writer `RELEASE/SYSTEM` store to its own flag; readers
`RELAXED/SYSTEM` glc-poll all peers' flags (no per-poll `buffer_invl2`), then
ONE `ACQUIRE/SYSTEM` load after all arrive, then plain local data loads. The
allreduce uses two barriers per epoch (produce + consume) so peers can't
overwrite slots being read.

Measured (gfx90a, 32 KB/rank, 150–300 iters):

| phase | fine (XGMI) | host (PCIe) |
|---|---|---|
| solo poll, ns/load | local 162–179, remote 494–920 | 1334 (ws=4) – 2047 (ws=8), PCIe-dominated |
| pingpong ws=4, µs/RTT | 3.1–3.6 | 8.0–8.1 |
| pingpong ws=8, µs/RTT | 3.2–5.4 (uniform) | 8.4–10.2, bimodal — and the mode **flips run to run**: the pinned page landed on node 1 even under `numactl --cpunodebind=0` (driver controls placement, not the process) |
| barrier ws=2 / 4 / 8, µs | 1.81 / 3.90 / 10.2 | — / 7.44 / 22.0 |
| allreduce ws=2 / 4 / 8, µs | **34.6** / 69.5 / 148 | — / 78.5 / 163–174 |
| correctness (all ws) | **FRESH, 0 mismatches** | FRESH, 0 mismatches |

4. **Read-gather (pull) beats write-scatter (push) — after vectorization.**
   Three dataflows measured in `flagprobe.hip` (fine flags, 32 KB/rank):

| µs/allreduce | ws=2 | ws=4 | ws=8 | ws=2 4MB | ws=4 4MB |
|---|---|---|---|---|---|
| write-scatter (push, scalar) | 36.1 | 69.4 | 143 | 4080 | 8247 |
| read-gather (pull, scalar) | 55.7 | 119.8 | 247.8 | 7073 | 15561 |
| read-gather, float4, 256 thr | 14.7 | 32.8 | 70.4 | 712 | 1520 |
| **read-gather, float4, 1024 thr** | **8.1** | **17.9** | **38.7** | — | — |
| RCCL (cited, 32 KB) | 45.7 | 45.7 | — | ~815 GB/s | — |

   The user's proposal (sync -> every rank remote-READS all others' contribs,
   reduce in registers, sync, store locally) is the best shape:
   - Producers only write locally; consumers do all remote traffic as reads.
     No slot scratch — the reduce is fused into the read pass.
   - Scalar pull is *slower* than scalar push — the fabric is
     **transaction-granularity-bound, not byte-bound**: 4B remote loads waste
     3/4 of each transaction. float4 (16B) loads give ~4x. Vectorize or lose.
   - Block width is the second knob: 1024 threads hides remote-load latency
     (2x over 256). The flag protocol is block-size-independent (thread 0 +
     `__syncthreads`).
   - Result at decode size (32 KB/rank): **8.1 µs ws=2 / 17.9 µs ws=4 —
     2.6-5.7x faster than RCCL's 45.7 µs**, correct (FRESH, 0 mismatches) at
     every ws.
   - Large-payload regime: single-block is one-CU-memory-pipe-bound and
     read-gather is O(ws*n) regardless — see the multi-block results below.

5. **Push/pull convergence, multi-block scaling, and in-harness RCCL**
   (all measured in `flagprobe.hip`; RCCL = AMD-overlay librccl via
   `LD_LIBRARY_PATH=~/scratch/rocm-debs/amd-7.1/opt/rocm-7.1.0/lib`,
   serial HIP setup + threaded `ncclCommInitRank`):

| 32 KB/rank, µs | ws=2 | ws=4 | ws=8 |
|---|---|---|---|
| write-scatter f4, 1024 thr (fused) | 8.32 | 17.74 | 34.15 |
| read-gather f4, 1024 thr (fused) | 8.49 | 17.87 | 38.12 |
| read-gather f4, 3-kernel pipe (8 blk) | 15.8 | 21.3 | 36.1 |
| RCCL sync-per-call | 41.6 | 53.3 | 75.0 |
| RCCL pipelined | 13.8 | 22.4 | 36.3 |

| 4 MB/rank, µs | ws=2 | ws=4 |
|---|---|---|
| write-scatter f4, fused 1-block | 576 | 1088 |
| read-gather f4, fused 1-block | 708 | 1520 |
| **read-gather f4, 3-kernel pipe (64 blk)** | **61.0** | 310.8 |
| RCCL sync-per-call | 81.5 | 143.0 |
| RCCL pipelined | 56.6 | 114.3 |

   - **Push vs pull is a wash once vectorized** (ws-f4 17.74 vs rg-f4 17.87
     at ws=4): the earlier difference was transaction granularity + thread
     count, not the dataflow direction.
   - **Multi-block scaling works**: the 3-kernel pipeline (fill / 1-block
     flag-barrier / wide gather / flag-barrier, plain stream ordering + the
     flag kernel's release-acquire across GCDs) goes 708 -> 61 µs at ws=2
     4 MB (11.6x) and 1520 -> 311 at ws=4 (4.9x). LSU bandwidth scales with
     blocks as expected; correctness holds (the barrier kernel's acquire
     invalidates the shared L2; kernel launch resets per-CU L1s).
   - At 32 KB the 4 launches/epoch (~8-12 µs overhead) make the fused
     single-block kernel the winner; at 4 MB the pipe is 5-12x better.
     Rule: fused for the latency regime, pipe for the bandwidth regime.
   - **RCCL in-harness**: sync-per-call 41.6/53.3/75.0 µs at ws=2/4/8
     (32 KB). Our fused f4 kernels beat RCCL sync-per-call 2.2-5x at every
     ws, and beat RCCL *pipelined* at ws=2/4. (The engine's decode path is
     sync-per-call-shaped: each allreduce is serialized by forward compute.)
   - RCCL's ring still wins big at ws>=4 + large payload (114 vs 311 µs at
     ws=4 4 MB) — O(n) beats O(ws*n) when bytes dominate, as expected.

Conclusions:

1. **Fine-grained device flags are the right pattern; host-pinned is out —
   verified.** Every host-mode poll is a ~1.3–2 µs PCIe round trip (vs
   0.16–0.9 µs over XGMI), the page's NUMA placement is driver-controlled
   and shifts between runs (so which half of the GCDs pays the cross-socket
   tax is unobservable/uncontrollable), and all flag traffic pollutes PCIe.
   Fine mode wins every axis and is 2× better on the barrier (3.9 vs 7.4 µs
   at ws=4; 10.2 vs 22.0 at ws=8).
2. **The protocol is correct at ws=2/4/8** — the write-scatter allreduce
   (remote writes into my VRAM, then my local reads — exactly the l2probe
   hazard) reports zero mismatches at every size: the acquire's
   `buffer_invl2` fixes the stale-L2 problem.
3. **The host-API floor is broken.** The study floored at ~80–100 µs/iter
   of host-event overhead; this device-side barrier costs **1.8 µs (ws=2) /
   3.9 µs (ws=4)**. The naive single-kernel write-scatter allreduce (1 block,
   scalar stores — payload path unoptimized) already hits **34.6 µs at ws=2
   (beats RCCL's 45.7 µs)** and 69.5 µs at ws=4 (RCCL's ring still wins
   there, but this is with an unoptimized payload path and two barriers per
   epoch; a parity-buffer single-barrier variant + vectorized stores is the
   obvious next step).
