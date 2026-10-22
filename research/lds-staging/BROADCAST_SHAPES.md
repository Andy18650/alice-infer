# Shared A/B panel shapes:4x4,4x8,8x8,13x8

User-directed extension of the8x8 result. Keep four waves/CTA,512 KiB A/B
panels,1 MiB logical source/CTA,32-slot DWORDx4 ring and no periodic waits.
Test only the three requested shapes plus a fresh8x8 control, each matched
against its own disjoint control. Earlier measured source/binaries preserved.

## Generalized map / controls

For R rows,C columns, CTA c corresponds to(i=c/C,j=c%C):

```text
A panels [0,R*512KiB), one512KiB panel per row
B panels [R*512KiB,(R+C)*512KiB), one512KiB panel per column
waves0/1 -> A_i +wave*256KiB
waves2/3 -> B_j +(wave%2)*256KiB
```

GPU source bases are recorded after timing and checked against the map for
every CTA/wave/launch. There are2R unique A-quarter regions, each referenced
C times, and2C unique B-quarter regions, each referenced R times. In the13x8
case, A reuse is8x while B reuse is13x—not uniformly8x. Disjoint control
uses4RC unique quarters, one per wave, without inter-CTA source sharing.

| Grid | CTAs | Active shared source | A/B panel reuse | Matched allocation |
|---|---:|---:|---:|---:|
|4x4 |16 |4 MiB |4x /4x |16 MiB |
|4x8 |32 |6 MiB |8x /4x |32 MiB |
|8x8 |64 |8 MiB |8x /8x |64 MiB |
|13x8 |104 |10.5 MiB |8x /13x |104 MiB |

Each shape allocates the full disjoint size in BOTH layout modes; sharing
reduces reachable addresses, not allocation size or requested bytes. L2
reported8 MiB, so this spans below/at/above nominal capacity.13x8 gives104
CTAs on a104-CU GCD; exact one-to-one CTA/CU placement is not recorded.

One kernel/executable for all shapes/layouts, dynamic row/column arguments
used before the clock bracket. The generated timed asm is identical to the
previous VGPR probes. Native loop remains53 instructions,32 x4-VGPR slots,
zero timed waits/consumers/copies/NOPs, one terminal VM0 outside issue interval.
Metadata139 VGPRs/25 SGPRs, zero LDS/scratch/lane spills; runtime occupancy
query3 CTAs/CU. Dynamic address setup/register allocation differs slightly
from the earlier fixed8x8 source, so8x8 is replayed as a fresh control.

## Measurement

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, block256,
no model/context/clock-policy overrides. Two counterbalanced process repeats,
nine checked launches each after200 ms selected-kernel warmup. Same source
pretraversal and post-warmup H2D output initialization as preceding studies.
Default256 MiB requested/CTA/sample,256 source traversals. Grid payload4/8/
16/26 GiB at16/32/64/104 CTAs. No uniformly-hot/cache-counter claim.

Core period: median CTA makespan /loads per wave each round, then median of
nine rounds. One all-wave step returns **4 KiB/CTA**. Aggregate requested GB/s
uses global steady-clock grid makespan, never mixed core timestamps across
CUs. Completion/full-event rates also retained. Every sampled CTA issue
interval overlaps; output/address-record stores are outside their CTA's own
bracket but can overlap other CTAs still loading.

## Shared-layout results

Process medians run1/run2, clock overhead included:

| Grid | Core cycles/four-wave x4 step | Requested grid GB/s | Matched disjoint GB/s | Throughput ratio |
|---|---:|---:|---:|---:|
|4x4 |68.663 /66.112 |1553.72 /1584.44 |729.80 /708.42 |2.13x /2.24x |
|4x8 |104.518 /104.830 |1972.34 /1974.23 |1257.07 /1256.31 |1.57x /1.57x |
|8x8 |136.972 /137.673 |3064.68 /3054.74 |1370.51 /1381.80 |2.24x /2.21x |
|13x8 |153.083 /151.863 |4348.38 /4383.01 |761.45 /747.80 |5.71x /5.86x |

Matched disjoint core periods:

| Grid | Disjoint cycles/step run1/run2 |
|---|---:|
|4x4 |146.470 /151.158 |
|4x8 |169.360 /169.936 |
|8x8 |310.559 /307.606 |
|13x8 |910.308 /928.415 |

-4x4's per-CTA period is close to the earlier single-CTA~64-cycle reference.
-More sharing/concurrent CTAs raises total requested throughput even as the
 per-CTA period grows; do not rank grids using only per-CTA periods.
-The13x8 shared case remains strong beyond nominal L2 footprint, with no
 capacity cliff at10.5 MiB here. Whole-source residency is not required for
 temporal reuse before eviction or coalescing of concurrent misses; those
 are consistent explanations, not counter-established mechanisms.
-The severe disjoint104-CTA regression is largely absent with this pattern.
 This is a useful limit on conclusions from the disjoint-streaming baseline,
 not a demonstrated real-GEMM/prefill speedup.

Core/steady ratios are~1646-1648 /1635-1636 /1604 /1562-1563 MHz for the
shared grids in increasing order, not fixed clock overrides. Wall-time and
core-cycle ratios need not be identical. Requested rates above HBM's physical
bandwidth reflect repeated source use;4.38 TB/s is NOT4.38 TB/s of HBM traffic.
No exact L2 residency, hit/miss/coalescing split or universal ceiling claim.

## Verification / records

**144 timing launches /7776 CTA records** plus16 fill/source-wrap grid smokes
pass. Every final ring slot/lane/DWORD and output guard checked; every GPU
source base/bound and family-specific reuse multiplicity verified. Periodic
payload remains an ambiguous last-iteration oracle, so address proof is
separate from data checking. No instruction/window/padding/skew tuning.

`results/broadcast-shapes/` preserves commands, per-CTA/per-wave clocks,
source maps, process medians, native/source SHA256, metadata and timed-loop
extracts. Distinct binary/full ISA/CDB: `temp/lds-staging/broadcast-shapes/`.

```sh
bash research/lds-staging/build-broadcast-shapes.sh
numactl --cpunodebind=0 --membind=0 temp/lds-staging/broadcast-shapes/probe \
  --rows 13 --cols 8 --layout outer --rounds 9 --warmup-ms 200
# --layout disjoint for matched control; rows/cols limited to listed shapes.
python3 research/lds-staging/run_broadcast_shapes.py --output <fresh-results-directory>
```

Research status appended to `research/research-log.md`. No earlier measured
source/binary, production/CMake/model/system/package/clock-policy changes.
