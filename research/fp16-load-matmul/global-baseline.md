# Historical v1 global-load baseline

Subsequent [verification](VERIFICATION.md) checks these original device
kernels directly, including record-varying global input data.

This records the pre-LDS implementation and original measurements. Its old
binaries remain at `temp/fp16-load-matmul/dD-uU/`, and its source snapshot is
`temp/fp16-load-matmul/pre-lds/source/`. The commands below refer to that
version; see [README.md](README.md) for current build paths and LDS options.

# One-CTA FP16 matmul operand-pipeline probe

User-directed standalone **performance-only** probe. One CTA, **256 threads /
four wave64s**, native `v_mfma_f32_32x32x8f16`, FP16 operands and FP32
accumulators. One 16-FP32 AGPR accumulator bank per wave. No FP8 conversion,
LDS, vendor GEMM, production integration, numerical oracle or output readback.

## Build and run

```sh
# Build-time parameters: depth, unroll, optional output directory.
bash research/fp16-load-matmul/build.sh 8 32
python3 research/fp16-load-matmul/inspect_loops.py temp/fp16-load-matmul/d8-u32

# Runtime parameters: mode, number of loads per operand pair, partial waits.
numactl --cpunodebind=0 --membind=0 temp/fp16-load-matmul/d8-u32/probe \
  --mode hot --loads 1 --wait 0 --steps 4096 --rounds 9

# L2-sized window: records are per wave; total footprint = records*4096 B.
numactl --cpunodebind=0 --membind=0 temp/fp16-load-matmul/d8-u32/probe \
  --mode l2 --records 256 --loads 2 --wait 1

# Streaming: automatically allocate enough records for all lookahead loads.
numactl --cpunodebind=0 --membind=0 temp/fp16-load-matmul/d8-u32/probe \
  --mode stream --loads 4 --wait 0

# Depth and unroll need not divide each other.
bash research/fp16-load-matmul/build.sh 16 1
bash research/fp16-load-matmul/build.sh 3 8
```

Depth accepts any integer **1..16**, unroll any integer **1..64**. Steps must
be a positive multiple of unroll; default 4096 steps means **logical K32768**.
Other flags: `--device 0`, `--warmup-ms 200`, `--evict-mib N`, `--help`.

Each build emits all 18 mode/load/wait entries in its own executable, plus
`pipeline.hpp`, ISA, metadata, hashes, extracted loops and a compile database.
No engine/CMake changes or external dependencies beyond the existing toolchain.

## Load shapes and dummy-data layout

Each wave/fragment occupies **1024 bytes**: 64 lanes * (8 A bytes + 8 B bytes).
All four waves have private streams, including private dummy A tiles; this
avoids unequal cross-wave A sharing between load shapes. Each wave computes
one 32x32 tile with one accumulator bank. This is not a generalized GEMM API.
Data are finite nonzero FP16 +/-0.125, initialized on GPU; no repacker or
cross-layout mathematical-equivalence check is supplied.

Within one wave fragment, with `lane=0..63` and dwords A0,A1,B0,B1:

| `--loads` | Instructions per pair | Physical layout / address offsets |
|---:|---|---|
| 1 | one `global_load_dwordx4` | lane*16 B: `[A0,A1,B0,B1]` |
| 2 | two `global_load_dwordx2` | lane*8 B: `[A0,A1]`; +512 B: `[B0,B1]` |
| 4 | four `global_load_dword` | lane*4 B; planes +0/+256/+512/+768 B |

Every instruction is coalesced across adjacent lanes. Every shape requests
**4096 logical bytes per four-wave step**. Four raw-to-ready `v_mov_b32`
instructions per step are retained in all shapes, keeping preparation work
constant. No hidden dtype conversion or different amount of arithmetic.

## Address modes

- **hot:** one fragment per wave, 4 KiB total. Pointers are invariant inside
  the loop; no per-step address arithmetic. In-dispatch pre-read before timing.
- **l2:** cyclic configurable window, default **256 records/wave = 1 MiB
  total**. The entire window is pre-read in the same dispatch before timing.
  Each load group advances one fragment; offset wraps at the chosen window.
  This is **L2-sized/primed**, not hardware-counter proof of L2 residency.
- **stream:** no wrap and no full-window pre-read. Default records are
  `steps+depth+2` per wave, including lookahead padding. At defaults the total
  footprint is ~16 MiB, above full reported 8 MiB L2. A completed **256 MiB
  eviction reader** precedes each launch by default, outside timing. Override
  with `--evict-mib 0` if desired; initial two seed loads still precede timing.

A record is 1024 bytes per wave regardless of load shape. For non-hot modes,
per-step address construction adds two vector address-add instructions and
scalar offset update; L2 adds wrap compare/select. Thus hot/L2/stream are
not an isolated cache-latency subtraction. No cache-hit or physical HBM/NoC
traffic claims are made. `--records` controls the allocated stream pitch and
available window size, printed as `active_bytes` in CSV/logs. For streaming,
only `steps+depth+1` records/wave are actually fetched including seeds and
lookahead; overallocating `--records` does not cause the loop to read that
extra space. The default has one extra padding record. Do not interpret the
allocation/window field as an exact timed unique-traffic counter.

## Pipeline and independently configurable unroll

Depth D means **D raw complete A+B fragment slots**, plus two ready operand
banks. Before timing, seed ready steps 0/1 and issue raw steps 2..D. Each
logical step n does:

```text
MFMA using ready[n%2]
issue the pair for n+D+1 into raw[(n+D-1)%D]
optional vmcnt(loads*(D-1))
copy all four dwords from raw[n%D] into ready[n%2], for use at n+2
```

The wait counts **instructions**, not bytes. `--wait 0` removes just the
steady partial waits: raw reads are intentionally unsafe and values may be
stale. Initial completion, final VMEM drain and timer LGKM waits remain.
Disabling waits cannot remove hardware issue/backpressure or the final drain.

`generate.py` emits a single opaque asm block with explicitly clobbered private
VGPRs. This prevents compiler movement/reuse/copies of unresolved registers.
Depth is not coupled to unroll by forcing a divisibility restriction.

**Register-rotation phases:** with `L=lcm(D,2)`, the implementation needs
`P=L/gcd(L,U)` phase-specific U-step bodies. There are three steady scalar
control instructions and one taken conditional branch after **each U steps**;
phase branches select the proper fixed register slots. An extra branch is
executed only on exit when P>1. Static code contains **P*U MFMA slots**, which
is explicitly reported, not silently called U. E.g. D16/U1 has 16 one-step
phases; D3/U8 has three eight-step phases, and can exit midway through that
phase cycle. Code footprint consequently depends on both D and U. No pending
register rotation is lowered into compiler-generated moves or scratch.

## Timing and inspection

Both core and steady clocks are saved **absolutely for every wave**. Primary
CTA time is `max(end)-min(begin)`, not the median of wave durations. CSVs also
retain mean/min/max wave intervals, full-dispatch event time, core/steady
frequency estimate, active footprint, and all 16 raw timestamp values.
Reported logical B/core-cycle is `4096*steps/CTA_span`, not physical traffic.

Pre-read, initial ready operands and raw issue priming precede core timing;
remaining in-flight work can overlap the interval. The final full VMEM drain
is inside timing; output stores are outside. Event time includes everything
in the probe dispatch, including any pre-read. Warm-up is **per selected
configuration**, default 200 ms, not a favored subset of a mixed kernel list.

`inspect_loops.py` checks the actual code object for all 18 entries: exact
MFMA/load/copy counts, wait thresholds, mode-dependent address instructions,
phase control, zero scratch/LDS/lane spilling, and conservative FIFO register
lifetimes for waited entries. This is instruction inspection, not numerical
testing; no safety/equivalence claim is made for no-wait reads. `analyze.py`
independently reconstructs each CTA interval from raw timestamps.

## Initial measurements

Native HIP O3/C++20, gfx90a, HIP0 (AMD-SMI GPU2 / PCI31:00.0), CPU+memory
NUMA0, one CTA/four waves, no model/context or clock/power overrides. AMD-SMI
before/after main sweep reports no other GPU processes. D8/U32, 4096 steps,
two independently warmed randomized process-order sweeps, nine samples each.
No numerical tests/output readbacks. **CTA core cycles/step, run1 / run2:**

| Mode | Load shape | No partial wait | With partial wait |
|---|---|---:|---:|
| Hot | 1 x dwordx4 | **64.133 / 64.133** | 64.349 / 64.150 |
| Hot | 2 x dwordx2 | **128.234 / 128.234** | 128.234 / 128.234 |
| Hot | 4 x dword | **96.458 / 96.458** | 96.845 / 96.956 |
| 1 MiB window | 1 x dwordx4 | 64.903 / 64.900 | 65.022 / 65.020 |
| 1 MiB window | 2 x dwordx2 | 128.222 / 128.221 | 128.216 / 128.224 |
| 1 MiB window | 4 x dword | 104.524 / 104.523 | 108.380 / 108.389 |
| Streaming | 1 x dwordx4 | 153.927 / 150.458 | 156.827 / 155.964 |
| Streaming | 2 x dwordx2 | 176.655 / 186.685 | 163.201 / 182.195 |
| Streaming | 4 x dword | 154.542 / 155.128 | 150.206 / 157.492 |

**The hot one-load A+B path feeds MFMA near its ~64-cycle rate.** This is a
useful matched-total-payload comparison, not identification of a physical
port or proof of an optimal streaming configuration. Streaming is noisier;
no universal winner or latency decomposition is claimed. No depth/unroll
performance sweep is claimed: D16/U1 and D3/U8 have four short three-sample
configuration examples each, plus full 18-entry code-object inspections.

## Artifacts and batch runner

```sh
python3 research/fp16-load-matmul/run_matrix.py \
  --binary temp/fp16-load-matmul/d8-u32/probe \
  --output research/fp16-load-matmul/results/d8-u32
python3 research/fp16-load-matmul/analyze.py
```

`results/`: timing CSV/logs, manifests, hashes/flags, inventory, summaries.
`temp/fp16-load-matmul/dD-uU/`: binaries, generated header, ISA/metadata,
extracted loops, compile database. Previous studies are not overwritten.

For clangd, point it at that build's generated `compile_commands.json`:

```sh
clangd --check=research/fp16-load-matmul/probe.hip \
  --compile-commands-dir=temp/fp16-load-matmul/d8-u32
```

The scoped `.clangd` defaults to D8/U32 when no command-line database override
is supplied. The workspace VS Code arguments currently force `build/`, which
takes precedence; choose this generated database to avoid inferred googletest
flags or a missing `pipeline.hpp`. No global editor configuration is changed.
