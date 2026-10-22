# Per-load VMEM watermark versus streaming bandwidth

User-directed test of the proposed latency/throughput window estimate. Return
to the pure loading loop, insert one `s_waitcnt vmcnt(N)` after every load,
and measure at1 and8 CTAs. Private16 KiB LDS/wave,64 KiB/CTA; disjoint1 MiB
source/CTA. The earlier source/geometry and64-load unroll remain intact.
No MFMA, LDS consumers, periodic barriers or per-load timing instrumentation.

## Counter/window convention and controls

```asm
# Repeat64 times per outer iteration, with the old fixed-address grouping:
global_load_dword v[8:9], off ... lds
s_waitcnt vmcnt(N)
```

After each wait, at most N wave-level VMEM instructions remain outstanding.
The following load can temporarily raise the count to **N+1**. This bounds
architectural load instructions per wave, not internal cache-line/memory
transactions. `N=0` serializes each wave; all four waves/CTA still participate.
**N=63 disables the counter test**, not a tested cap of64. It retains one
extra decoded instruction/load, unlike the separate original no-wait control.

Thresholds0..32 (dense),40/48/56/62, disabled63, plus no-periodic-wait control:
39 profiles total. No sweep of other geometry/producer choices. All waited
variants have **145 native loop instructions** (64 loads,64 waits, original
17 address/control instructions); no-wait has81. Same terminal VM0 outside
the issue interval and clock handling as the original bandwidth study.

One template is compiled with63. Its64 wait immediates are patched to each
threshold, equal width, same PCs/branches/register allocations/metadata and
all other object/host bytes unchanged. LLVM-MC provides encodings, actual
embedded objects are re-extracted and disassembled, all64 next-after-load
sites audited, all byte differences bounded to those sites. Native manifests
and prelaunch executable SHA256 guard against wrong-binary comparisons.
No-wait is a separate compile with the same C++ source and no inserted waits.
These are numerically checked variants, not unsafe performance-only ablations.

## Measurement

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, block256,
no model/context/clock overrides. Device reports104 CUs/8 MiB L2. One or8
CTAs allocate64 KiB each; measured steady-clock body intervals overlap for
all8 CTAs in every timing launch. Source is1/8 MiB nominal;8 MiB is at the
reported L2 capacity before other accesses, not a guaranteed hot-cache case.

Two seeded-random/reversed process-order repeats, nine checked samples/process,
after at least200 ms selected-variant warmup.16384 outer groups/sample gives
1,048,576 loads/wave and1 GiB requested payload/CTA (8 GiB/grid at8 CTAs).
There are1024 source traversals within each launch, so this is long streaming
service rather than the latency recorder's first-source-pass/revisit mixture.
No cache counters certify its hit rate.

For each CTA, core cycles = max(wave issue-end)-min(wave begin); divide by
loads/wave. For8 CTAs, take the median CTA period each round, then median of
nine rounds. **Do not compare absolute core timestamps across CUs or divide
by8x as many instructions.** Aggregate requested GB/s instead uses the global
steady-clock grid makespan. Raw timestamps and per-CTA spread are preserved.
Each four-wave load step requests1024 B/CTA, not256 B. Full HIP event timing
includes initialization/output; body timing excludes final LDS copying.

## Results: measured knees, not assumed queue sizes

Core cycles/four-wave load step, two process medians. Selected rows below;
all39 profiles are in `results/windows/curve.csv` and full `summary.csv`.

| vmcnt N | Peak wave-level window bound | 1 CTA run1/run2 | 8 CTAs run1/run2 |
|---|---:|---:|---:|
| 0 | 1 | 239.459 /247.288 | 493.568 /497.217 |
| 1 | 2 | 129.280 /127.453 | 234.092 /236.560 |
| 7 | 8 | 33.640 /32.534 | 76.403 /75.794 |
| 11 | 12 | 22.295 /22.226 | 53.446 /53.435 |
| 15 | 16 | 18.203 /18.545 | 41.596 /41.592 |
| 16 | 17 | **17.399 /17.391** | 39.287 /40.795 |
| 17 | 18 | 17.388 /17.387 | 37.882 /39.701 |
| 18 | 19 | 17.389 /17.388 | 36.533 /36.823 |
| 19 | 20 | 17.394 /17.395 | **35.648 /35.723** |
| 20 | 21 | 17.395 /17.396 | **34.862 /35.114** |
| 31 | 32 | 17.395 /17.395 | 35.405 /35.369 |
| 62 | 63 | 17.396 /17.396 | 35.244 /35.103 |
| 63 | Disabled | 17.396 /17.395 | 35.429 /35.383 |
| No inserted wait | Natural backpressure | 17.365 /17.373 | 34.555 /36.975 |

**1 CTA:** vmcnt15 still constrains throughput; vmcnt16 reaches the disabled-
wait floor and all tested higher finite thresholds stay essentially flat.
A maximum17-instruction window/wave is sufficient here, consistent with the
user's16–17-load estimate. Inserting disabled waits adds only~0.03 cycles/load
relative to no-wait in this case; decoder cost does not explain the knee.

**8 CTAs:** vmcnt16 remains clearly constraining. The curve approaches the
~35–36-cycle plateau around vmcnt19–20, i.e. a20–21-instruction window/wave.
The knee is less sharp: vmcnt22/24/30 each have one run around37 cycles,
and the original no-wait control is34.55/36.98. Those nonmonotonic/high-window
results are retained, not averaged away or silently retuned. We do not claim
one exact stall-free threshold at8 CTAs.

Thus the empirical window shifts with the multi-CTA workload. Also, both the
serial period (vmcnt0) and saturated streaming period increase at8 CTAs.
A fixed unloaded latency / throughput ratio is not an exact loaded queue-
capacity model. The ~239–247-cycle serialized1-CTA period is not expected to
match the earlier~276–288 per-load recorder: the latter has two clock reads/
completion waits per interval; this loop has clocks only around the full body.
The serialized loop still includes its own address/control overhead.

**A plateau proves window sufficiency for throughput, not that every wait is
stall-free or that hardware has an exactly17/21-entry transaction queue.**
Memory/cache transport, offered-load latency, instruction issue and natural
hardware backpressure remain coupled. No physical bandwidth roofline claim.

## Checks and records

**1404 grid timing samples /6318 CTA records** pass all per-CTA64 KiB LDS
words and output guards. No spills, conversions, sample-loop barriers or
consumers; wait immediates are the only changes among waited variants. Eight
CTA issue intervals overlap in every sample. Full native manifests/encodings,
loop extracts, hashes before launch, source snapshots, commands, clocks and
all verified timing samples live in `results/windows/`. Actual programs/full
objects/ISA are `temp/lds-staging/windows/{template,nowait,variants/}`.
The raw data also records actual core/steady ratios (~1.68 GHz), not a fixed
frequency assumption. No old measured executable or source is overwritten.

```sh
bash research/lds-staging/build-window.sh                #63 template
bash research/lds-staging/build-window.sh temp/lds-staging/windows/nowait none
python3 research/lds-staging/run_windows.py               #fresh output/artifact paths only
# --build / --nowait / --output / --artifacts select fresh replay paths.
```

This round uses only private LDS destinations because earlier studies found
no shared/private difference; it does not rerun those axes or tune other
schedules. Research status goes to `research/research-log.md`, per user move,
not engine-design.md. No production/CMake/model/system/clock-policy changes.
