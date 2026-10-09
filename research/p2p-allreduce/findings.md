# End-of-round synthesis

All studies are standalone; production is not a dependency. Sources, raw
logs, compiler/binary hashes and the detailed tables are documented in
README.md / temp/p2p-allreduce/. Numbers are native O3, CPU+memory NUMA node 0,
graph mode unless explicitly marked eager. No attention context is modeled;
32 KiB / 1 MiB / 4 MiB correspond to 70B S=1 / 32 / 128.

## Conclusions worth carrying forward

1. Immutable per-collective inputs + separate output remove the consume
   barrier. Sources may be reused only after the all-rank forward boundary.
2. An ordinary wide kernel can publish readiness because the whole producer
   is a preceding kernel. Cooperative grids are not needed for this design.
   Cooperative launch itself is expensive here: the identical-kernel
   coop-wide control loses even without grid.sync(). This is not a universal
   conclusion about cooperative kernels or fused producers.
3. More CTAs are not automatically better. Same-kernel 32x256 pipeline tests
   beat the inherited 220x256 at 1/4 MiB. Do not ascribe that comparison to
   different reduction implementations or assume a precise fabric cause.
4. The gather was already coalesced float4. Pointer-table lookup and payload
   waits serialized the inner loop. Hoisting peer addresses into 64 B of LDS
   is a strong bulk improvement. Cyclic dephasing helps some workloads, not
   uniformly; pointer-only controls avoid calling every gain a topology win.
5. Two/four-source lookahead is geometry-dependent, not a blanket speedup.
   Verify codegen: outer unrolling initially turned depth two into eight, and
   interleaved LDS/flat waits could collapse the intended overlap. The final
   variants have bounded issue and no spills. More MLP can increase congestion.
6. Memory-backed peer-major has explicit local partial-output traffic; its
   losses do not alone condemn the loop order. Register tiles 1/4/8/32 were
   checked, including full-tile fast paths. They are not general bulk winners.
   Tile-four decode has a narrow same-geometry gain; data-major at its useful
   wide geometry remains the selected simple decode reference. Retain unusual
   peer-major points only as targeted experiments, not a Cartesian search.
7. The 32-vector tile fits only with the intended register budget: <=256
   threads, gfx90a one-wave/EU launch bound, 204 VGPRs, zero spills. Capacity
   and occupancy are distinct. 16 MiB full-chunk register storage is excluded.
8. Working AMD-overlay RCCL changes the bulk conclusion. Out-of-place captured
   RCCL at ws=4/8: 32 KiB 23.67/39.09 us, 1 MiB 63.89/79.05, 4 MiB
   116.12/123.38. Matched P2P references: 15.42/26.03, 80.86/122.68,
   296.21/558.52. Keep P2P decode; RCCL is the prefill integration candidate.
   Version number is not proof of a working build: verify the actual library
   path and use the documented AMD 7.1 overlay, never Canonical's broken one.

## Limits and implementation pitfalls

- Reported latency includes a common GPU producer. Do not subtract controls
  as though overlap costs were additive, or multiply us/op into a promised
  engine speedup. Real GEMV scratch/KV dirty-cache behavior differs.
- Timing events disable system fences. Otherwise the timer can repair a broken
  publication protocol. GPU elapsed is max-rank per-device delta, not sums
  across GPUs or cross-card wall_clock64 differences.
- The own-flag acquire is a gfx90a cache-invalidation idiom, not a portable
  peer release/acquire happens-before proof. NVIDIA/L2 fabric arguments remain
  untested here. Carry this explicitly into native-backend documentation.
- Dephased peer order changes FP32 summation order and can differ across ranks.
  Exact small integer/32 operands validate addresses/coherency, not arbitrary
  floating-point replica equivalence. Real canonical gates are mandatory.
- RCCL stop events must be after GroupEnd; all streams must be capturing before
  grouped multi-rank recording. Warm a collective with every rank submitted,
  never one rank alone. Destroy graphs before normal communicator destruction.
- Initialization is outside inference timing. Reuse a communicator set across
  payloads/modes rather than pay 15-22 seconds per research point.
- Source-level arrays are not evidence of register storage. Inspect metadata
  for scratch/spills and actual instructions before naming a register kernel.
- Keep the search bounded. Default traversal is 30 cases (~50 s), lookahead
  32 cases (~54 s), selected register comparisons 40 (~78 s). Poor variants
  remain in history or explicit targeted commands, not every new sweep.

## Production integration decision for this round

Use the captured decode reference: a produced barrier followed by a
pointer-hoisted, cyclically dephased data-major gather (2x1024 at the 32 KiB
70B shape). Use out-of-place RCCL for S>1. Preserve the immutable slots and
explicit all-rank completion boundary, and validate canonical numerics plus
short/long-context end-to-end performance before claiming production gains.
This is a workload-specific hybrid, not a universal allreduce algorithm.
