# gfx90a unloaded VMEM completion latency

User-directed first latency study on the same MI250X GCD as the GEMV/GEMM
research. Standalone C++20/HIP; no production, package, clock/power-policy or
system configuration changes. This measures latency, not streaming bandwidth.

## First results

Native O3, HIP 0 / PCI **0000:31:00.0**, gfx90a, 104 CUs; CPU+memory NUMA node
0, no model/attention context, automatic clocks. **One wave on one CU**, without
an intentionally concurrent GPU workload. Two process runs, different random
cycle seeds/order, nine rounds each: **18 samples/configuration**, 1,134 records
across nine allocation sizes and seven mode/policy/control combinations.
Measured effective core rate is about **1685 MHz**, not the ~1550 MHz busy GEMV
rate or ~1385 MHz GEMM rate. Both cycle and steady-counter results are retained.

For the **64-lane coalesced dword load** (256 B per wave instruction), default
cache policy:

| Allocation | Median raw single-load interval | Empty interval | Single minus empty, approximate | Dependent-chain step, cycles / ns |
|---|---:|---:|---:|---:|
| 4 KiB | 160 cycles | 44 | ~116 cycles | 192 / 114 |
| 64 KiB | 296 | 44 | ~252 | 319 / 189 |
| 4 MiB | 296 | 44 | ~252 | 311 / 184 |
| 16 MiB | 600 | 44 | ~556 | 684 / 406 |
| 64 MiB | 616 | 44 | ~572 | 692 / 411 |
| 256 MiB | 708 | 44 | ~664 | 791 / 469 |

The raw single intervals include timestamp/wait overhead. The empty subtraction
is an **estimate, not an exact isolation of memory-controller latency**. At the
observed clock, the large-footprint estimates correspond roughly to 340-400 ns,
whereas the complete dependent steps take 410-470 ns. Do not substitute the
step time or a raw interval for pure issue-to-data-ready latency without these
qualifications. No high-percentile latency claim is made from 18 first-request
samples.

The 1024 B wave `global_load_dwordx4` probe gives ~204 cycles/121 ns per chain
step at 4 KiB, ~368/218 at 4 MiB, ~675/400 at 64 MiB and ~744/441 at 256 MiB.
These modes match the payload/coalescing of a GEMV load instruction, **not the
actual GEMV address stream, prefetch depth or active-CU concurrency**.

## What is measured

`probe.hip` builds one randomized Hamiltonian cycle of node indices per
allocation/mode. GPU initialization fills every node; each lane's first word
contains the next node index. The host independently computes the expected
final index. No allocation, upload/download or host synchronization occurs
inside a measured interval.

Modes:

- `lane-dword`: only lane 0 active, one 32-bit load; nodes spaced 128 B apart.
- `wave-dword`: all 64 lanes load consecutive dwords; each node is 256 B.
- `wave-dwordx4`: all lanes load four adjacent dwords; each node is 1024 B.
  All four components of the final vector are checked independently per lane.
- `dependent-add`: same loop count with dependent VALU add and vmcnt wait,
  no load. It is a lower-bound/control sequence, **not a matched subtraction
  for all address, wait and scheduling costs**. Median is ~40 cycles/step.

Each read has a true next-address dependency on the preceding load, and an
explicit **`s_waitcnt vmcnt(0)`**. No second VMEM load instruction is issued
while the preceding one remains outstanding. One vector instruction can still
produce multiple memory transactions internally.

Each kernel traverses the **complete cycle once before timing, on that same
wave/CU**. This avoids warming L1 on a different CU and avoids mistaking a
repeated short, cache-fitting subchain for a large working set. Different rounds
start at different cycle positions. The sizes are 4/16/64/256 KiB and 1/4/16/64/
256 MiB. Reported sizes are allocation footprints: scalar mode touches one word
per 128 B node, so fetched bytes and cache-line/tag footprint differ. These
curves are not an exact cache-capacity measurement.

Two intervals follow the warmup:

1. **One pre-addressed load**, bracketed inside a single volatile inline-asm
   sequence by `s_memtime` and completion waits. Address generation occurs
   before the timestamp block. An identical empty timestamp/wait block is
   recorded immediately before it. No result-buffer store occurs between the
   load and its ending timestamp.
2. **65,536 dependent steps**, bracketed by core `s_memtime` and fixed-rate
   `s_memrealtime` counters. The latter runs at the device-reported 25 MHz.
   This includes address calculation, load, completion wait and loop control.
   Pure scalar arithmetic can be scheduled around the asm boundaries; a few
   setup/final instructions are included too, negligible at this loop count.

Only after both intervals do lanes write results. Hardware IDs before/after
measurement must agree. Every active lane's final index/payload is validated;
compiler/ISA checks independently establish that warmup and timed loads remain.

### Single-request caveat

The single interval is the **first request after full warmup plus timer
calibration**, not a sample collected in the middle of a continuously streaming
kernel. Timestamp instructions perturb instruction spacing. First-request
variation can be large, especially for the one-active-lane probe: at 4 KiB its
raw intervals span 156-1000 cycles despite a stable ~184-cycle chain mean.
This is retained evidence, not silently discarded as an outlier. Do not infer
precise scalar intrinsic latency or a tail distribution from that single-event
median. The wave-coalesced probe and long dependent sequences provide separate
cross-checks, not interchangeable measurements.

## Hardware-counter validation

An additional **single-pass** capture uses `cache-counters.yaml` on a short
4,096-step run. Each captured dispatch has exactly one `SQ_WAVES`. For every
read dispatch:

```
SQ_INSTS_VMEM = nodes + 1 + iterations + 4
               warm    one  chain        result stores
```

Controls have exactly four VMEM stores and no loads. All **63** records pass
these counts, proving the full-cycle warmup/single/chain load work is live.

For the 4 KiB wave-dword case, across the complete captured kernel:

| Policy | TCP cache accesses | TCP-to-L2 read requests |
|---|---:|---:|
| default | 16,708 | 64 |
| glc | 16,708 | 16,452 |

This directly supports the default-cache versus L1-bypass interpretation for
these instructions on this GCD. Default chain steps are ~192 cycles, glc ~312.
At 64 KiB and above, their latencies largely converge. The intermediate-size
measured chain is consistent with L2 service; large-footprint counters show
substantial external reads and near-total L2 misses.

**Counter collection covers warmup as well as measurement.** For a fitting L2
case, initial warmup misses can dominate that kernel's aggregate hit rate; do
not present the whole-kernel percentage as the timed subinterval's hit rate.
Counter payloads are not total physical wire traffic. The existing mixed-version
profiler's metadata limitations described in `../gemm-compute-bound/profiling.md`
still apply. No profiler timing is mixed into the unprofiled latency summaries.

The 256 MiB latency increase relative to 16/64 MiB can include translation,
page/row locality and cache effects; it is not isolated DRAM-cell latency or a
proven channel map.

## ISA and resource checks

`check-codegen.py` checks the extracted measured device object:

- All seven chase instantiations have **zero private bytes and zero LDS**.
- Each read instantiation has three live load sites: warmup, single request,
  and dependent loop. Each load is immediately followed by vmcnt(0).
- The single request has the expected core-timestamp/wait/load/wait/timestamp
  bracket with **no address arithmetic inserted into the interval**.
- Scalar/wide load instructions match the intended types; default and glc
  paths remain distinct. Timing asm is not merely a compiler scheduling fence.

The dependency and control paths differ, so subtracting 40 cycles from each
chain point is not promoted as an exact latency correction. Raw measurements
and empty intervals remain in the CSVs.

## Reproduce and retained artifacts

```sh
bash research/memory-latency/build.sh temp/memory-latency/new-build
bash research/memory-latency/run.sh temp/memory-latency/new-a \
  temp/memory-latency/new-build/probe --rounds 9 --iterations 65536 --seed 12345
bash research/memory-latency/run.sh temp/memory-latency/new-b \
  temp/memory-latency/new-build/probe --rounds 9 --iterations 65536 --seed 98765
python3 research/memory-latency/check-codegen.py temp/memory-latency/new-build \
  --out temp/memory-latency/new-codegen.json
python3 research/memory-latency/analyze.py \
  --runs temp/memory-latency/new-a temp/memory-latency/new-b \
  --out temp/memory-latency/new-summary
```

The active run helper now checks the build's executable/source hash manifest
and refuses nonempty output directories, matching the window helper. This
review-time safety change does not alter or relabel any retained measurements.

For cache verification, use the already-validated local profiler environment
(Python venv on PATH, SDK Python modules on PYTHONPATH, installed local AQL
library directory on LD_LIBRARY_PATH), then:

```sh
SDK=/home/andyhu/.cache/uv/archive-v0/2EEHNuXZame80Ad1/_rocm_sdk_core
HIP_VISIBLE_DEVICES=0 "$SDK/bin/rocprofv3" \
  -i research/memory-latency/cache-counters.yaml \
  -d "$PWD/temp/memory-latency/new-cache-profile" \
  --kernel-trace --output-format csv --kernel-include-regex chase \
  --kernel-iteration-range '[1-9]' -- \
  numactl --cpunodebind=0 --membind=0 temp/memory-latency/new-build/probe \
  --rounds 1 --iterations 4096 --seed 12345
```

`analyze.py` additionally accepts `--counter-log` (combined stdout/stderr from
that invocation) and `--counter-csv` (raw counter_collection.csv) to join and
validate all counters. It matches each kernel's per-size ordinal to stdout;
retain all nine filtered instances for this mapping.

Curated data: `results/{samples,medians,cache-counters}.csv`, executable/source
hashes and commands in `provenance.json`, ISA checks in `codegen.json`. Full
binary/object/ISA/source snapshots and raw logs are under durable repo-local
`temp/memory-latency/`: `initial` is the first chain-only smoke, `v2` is the
measured single-plus-chain build, `a/b` are the independent nine-round runs,
and `cache-profile` is the counter check. No stale initial-build result is
substituted into the v2 summary.

## Loaded-window follow-up

[window.md](window.md) now measures bandwidth against independent burst depth,
wave count and instruction width on a fixed 1040 MiB input. Native SDK
`VmemLatency` collection works (unlike the earlier blank Compute table), giving
same-pass instruction-lifetime/population estimates. At 104 CTAs, wide-load
depths 1/2/4 reach ~1.02/1.41/1.47 TB/s unprofiled, with 416/832/1664 KiB
nominal windows. The counter-pass lifetime grows to ~0.51-0.97 us near the
plateau. This is a shader instruction window, not physical HBM queue bytes.

The baseline here and that long-stream reader still do not explain the output
GEMV's full ~31 us producer time. Remaining controls include its actual stride,
working-set duration and a continuous rather than burst/drain pipeline. Do not
multiply serial latency by the GEMV load count: independent loads/waves overlap.
