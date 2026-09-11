# Study: single-process multi-rank RCCL + P2P fallback (2026-09-11)

Question (user): does single-process, multi-rank RCCL work on this machine?
If not, TP would have to go the multi-process SPMD route.

## Verdict

**RCCL is broken on this stack for ANY multi-rank configuration — in both
process models.** The single-process multi-rank *model* is sound (init
works, see below); it is Ubuntu's RCCL build that cannot execute a
collective. A hand-rolled P2P allreduce is fully viable and is now the
recommended first TP path.

## What was tested

`study.cpp` (g++, host-only, build.sh): one process, N std::threads, one
RCCL rank per thread/GCD; correctness allreduce (exact fp32 pattern) +
latency sweep 8 KB–16 MB.

Findings, in order:

1. Concurrent first-touch HIP context creation from 8 threads → **segfault**
   inside libamdhip64. Fix: create contexts/streams/buffers serially in the
   owner thread first (mirrors the engine: the model object owns device
   state; threads only drive collectives).
2. Threads then failed `ncclCommInitRank` with `ncclInvalidUsage`: the
   *current device is per-thread state* — each rank thread must
   `hipSetDevice(rank)` itself; RCCL validates it against the rank.
3. With both fixed: **in-process multi-rank init succeeds** (4 ranks, all
   "Init COMPLETE", 2.34 s, connections 1.26 s). The model works.
4. First `ncclAllReduce` → **segfault** in libamdhip64's legacy
   `hipHccModuleLaunchKernel` path, called from RCCL's hipified launch stub.
   Faulting instruction `mov 0x20(%rdi),%rax` with rdi = 0x43000 — a size in
   a pointer slot: the launch-path argument layout in Ubuntu's hipified
   RCCL build does not match the installed HIP runtime. A build bug, not
   configuration.
5. Vendor corroboration: `/usr/libexec/rocm/librccl1-tests/rccl-UnitTests`
   passes SP/MP **1-rank** AllReduce, then "Child 0 pipe closed
   unexpectedly" at the first **2-rank** test in BOTH single-process and
   multi-process modes. Multi-rank RCCL is broken here, period — SPMD
   would not have saved us.
6. Env knobs tried, all crash identically: `NCCL_P2P_DISABLE`,
   `NCCL_IB_DISABLE`, `NCCL_LAUNCH_MODE=PARALLEL`, `NCCL_ALGO=RING`,
   `NCCL_PROTO=LL`, `NCCL_NCHANNELS=1`.

Package facts: `librccl1` = `libamdhip64-7` = 7.1.0-0ubuntu2 (same Ubuntu
"resolute" ROCm build; RCCL reports NCCL API 2.27.7). gfx90a present in
`.hip_fatbin` (48 MB embedded device code). RCCL's own warnings on this
box: `numa_balancing=1` (perf variance) and **missing `iommu=pt`** from
the kernel cmdline ("can lead to system instability or hang") — neither
was the crash cause here (P2P_DISABLE didn't help), but `iommu=pt` is
worth adding at the next maintenance window regardless.

## P2P fallback probe (p2p.cpp, hipcc, no RCCL)

- Peer access: **full 8×8 mesh, every pair enableable**.
- `hipMemcpyPeer` (DMA): 50.7 GB/s on the weight-15 pairs (0↔1, 2↔3,
  4↔5, 6↔7 — the intra-card GCD pairs), ~37 GB/s on weight-30, ~31 on
  weight-45.
- **Direct remote reads from a device kernel** (the allreduce primitive):
  **134.7 GB/s** intra-card, **71.9 GB/s** on the 30-weight cross-card
  links, 36.8 GB/s on the 45-weight links. SM remote loads beat the DMA
  engines up to 2.7× on the fat links — a hand-rolled allreduce should
  reduce intra-card first, then cross-card (hierarchical, topology-aware).

TP feasibility (70B, d=8192, fp32 stream row = 32 KB): a flat
"every-rank-reads-every-rank" allreduce at decode moves 7×32 KB remote
per rank ≈ 6 µs at the worst link class; hierarchical is better. ~160
allreduces/token at 80 layers ⇒ ~1 ms against a ~22 ms TP=4 decode
budget (~4%). Viable with margin.

## Recommendations

- **First TP path: hand-rolled P2P allreduce behind a `CrossDeviceReduce`
  seam** (single process, model owns devices; threads only for the
  collective, or fused into block epilogues later). No RCCL dependency.
- **RCCL: report the Ubuntu bug** (hipified 7.1.0-0ubuntu2 build segfaults
  on any ≥2-rank collective on gfx90a) and revisit only if the user
  installs AMD's matched RCCL (amdgpu repo, needs sudo) or Ubuntu fixes
  the package. RCCL remains the right answer for complex collectives
  (all-to-all if MoE expert-parallel arrives).
- Add `iommu=pt` to the kernel cmdline at the next maintenance window
  (user action; RCCL explicitly warns about hang/instability without it).

## Reproduce

```sh
cd research/rccl-single-process
./build.sh && ./study 4            # RCCL probe (segfaults at first allreduce)
hipcc -O2 p2p.cpp -o p2p && ./p2p  # P2P probe (works; prints matrices)
NCCL_DEBUG=INFO ./study 4          # init completes; crash is in the launch
/usr/libexec/rocm/librccl1-tests/rccl-UnitTests --gtest_filter='*AllReduce*'
```
