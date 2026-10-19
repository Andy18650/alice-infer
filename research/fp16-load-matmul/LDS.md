# LDS operand staging: two raw slots hide the main latency penalty

Historical timing-only study. Subsequent [full verification](VERIFICATION.md)
checks the original code objects exactly. The [single-ready diagnostic](SINGLE_READY.md)
is separate and fails varying inputs; the tables here use two ready banks.

User asks to replace VRAM operand loads with LDS, respecting 32 banks * 512
dwords (64 KiB), using the different completion counter, and finding the
pipeline depth required. **VRAM->LDS transfer is excluded entirely.** The
CTA directly fills dynamic shared memory from finite FP16 bit constants,
synchronizes, and then measures LDS reads + MFMA. No numerical tests.

## Main answer and stage definition

**For hot LDS, D=2 is sufficient for all three load shapes to reach the
~64-cycle MFMA rate.** Wide reads sweeping the full 64 KiB are also near
~65 cycles at D2. Four scalar sweep reads retain a smaller non-monotonic
~66–69-cycle overhead; increasing depth does not consistently remove it.

Here **D means raw complete A+B fragment slots, PLUS two ready operand
banks**. Thus “D2 is sufficient” does not mean just two total operand buffers.
It gives one intervening MFMA interval from issuing a raw fragment to copying
it into a ready bank, then one further MFMA interval before consumption.
D1 issues a raw fragment and immediately waits/copies it, exposing load latency.

## Timing and matrix

Native HIP O3/C++20, gfx90a, HIP0 (AMD-SMI GPU2 / BDF31:00.0), CPU+memory
NUMA0, one CTA/256 threads/four waves, one reused accumulator bank/wave,
logical K32768 / 4096 steps, unroll32. No model/context or clock/power changes;
no profiling counters. No other GPU processes in before/after snapshots.
Every configuration gets its own 200 ms warm-up. Two randomized process-order
runs of nine timed samples per case, with both waits enabled and disabled.

Depths 1/2/3/4/8, three DS shapes, hot/sweep, stride1. Stride4 conflict controls
at D1/D4. Total 1512 timing samples; no output readback or numerical oracle.
CTA timing uses absolute `max(end)-min(begin)` and saves all raw clocks.

**Waited, stride1, hot LDS: pooled median core cycles/step**

| Raw depth | 1 x ds_read_b128 | 2 x ds_read_b64 | 4 x ds_read_b32 |
|---:|---:|---:|---:|
| 1 | 89.269 | 88.896 | 98.893* |
| 2 | **64.035** | **64.033** | **64.285** |
| 3 | 64.042 | 64.040 | 64.337 |
| 4 | 64.042 | 64.039 | 64.297 |
| 8 | 64.060 | 64.071 | 64.332† |

**Waited, stride1, sweeping full 64 KiB LDS**

| Raw depth | 1 x ds_read_b128 | 2 x ds_read_b64 | 4 x ds_read_b32 |
|---:|---:|---:|---:|
| 1 | 97.022 | 97.024 | 109.023 |
| 2 | **64.656** | **64.904** | 69.030 |
| 3 | 64.667 | 64.792 | 69.046 |
| 4 | 64.669 | 64.795 | 65.928* |
| 8 | 64.698 | 64.827 | 69.467† |

\* Scalar4 run medians spread: D1 hot **96.893/100.892**; D4 sweep
**69.424/65.928**. Do not portray the pooled scalar numbers as an exact
monotonic latency curve. All individual samples/run medians are retained.

† D8/scalar4 requested wait28 is conservatively capped at14; see below.

With waits disabled, D1 already gives hot ~64.03/~64.03/~64.27 for the
three shapes. D2 no-wait sweep is ~64.53/~64.66/~65.03. This removes much of
the D1 waited penalty but does not establish safe arithmetic for no-wait reads.
The remaining sweep scalar overhead has not been uniquely assigned to issue,
wait/dependency scheduling, bank phasing or another specific mechanism.

## Bank layout and conflict control

Every wave has its own padded fragment and stream, including A; no cross-wave
broadcast/reuse is hidden in the comparison. For LDS stride S, byte addresses
within a wave record are:

```text
b128: lane*16*S                 (four adjacent dwords)
b64:  lane*8*S + {0,512*S}       (two adjacent dwords per instruction)
b32:  lane*4*S + {0,256*S,512*S,768*S}
```

Record span is 1024*S bytes/wave; four waves consume 4096*S bytes/record.
Sweep defaults to 16/S records/wave, exactly 64 KiB allocation. Fragment and
wave bases are multiples of 32 dwords, so sweeping changes bank rows, not the
bank-index pattern. The constant plane offsets are encoded directly in DS
instructions. LDS symbol base is included in the byte pointer, not assumed.

Under bank=`dword_address % 32`, for one complete four-wave A+B step:

| Layout | Distinct requested dwords | Active banks | Dwords per active bank |
|---|---:|---:|---:|
| Stride1 | 1024 | 32 | 32 |
| Stride4 | 1024 | 8 | 128 |

Addresses in this model are distinct: the repeated bank hits are different
rows, not same-address broadcasts. This is **aggregate bank balance**, not a
claim that every undocumented wide-read service phase has zero conflicts.
For scalar stride1, each 32-lane half-wave addresses all 32 banks once.
Near-64 MFMA timing can also hide some subdominant read-side cost; it is not
a bank-conflict counter measurement. `bank-model.json` records the arithmetic
for strides 1/2/4/8 and every shape.

**Deliberate stride4 control, hot LDS, waits enabled:**

| Depth | b128 | 2 x b64 | 4 x b32 |
|---:|---:|---:|---:|
| 1 | 128.661 | 128.296 | 128.292 |
| 4 | **128.125** | **128.126** | **128.125** |

D4 with waits disabled also remains ~128.125 for all three shapes.
Fourfold greater concentrated bank pressure makes the transfer budget about
128 cycles under the usual one-dword/bank/cycle service model; measured values
are consistent with it. Depth does not hide this throughput penalty. No
specific internal DS phase/port has been identified with counters.

Stride4 uses padded records; full-window sweep accesses a subset of bytes
spread across 64 KiB. The primary conflict comparison above uses hot reads,
with identical payload and no per-step cursor arithmetic in either layout.

## Counter rules and the timestamp trap

LDS uses **`s_waitcnt lgkmcnt(...)`**, never `vmcnt` for its reads. Each
fragment contains N=1/2/4 instructions; nominal partial threshold is N*(D-1).
On this target the LGKM immediate is four bits; **15 is the all-ones/no-wait
setting**. The generator conservatively uses `min(N*(D-1),14)` and prints
requested/effective values. This can be a stronger wait for deep rings; it is
not secretly treated as the nominal larger in-flight window. D1..4 all shapes,
and D8 wide shapes, fit without clamping. The minimum-depth conclusions use
uncapped thresholds. `--wait 0` still leaves seed/timer/exit completion waits.

**The timer instructions themselves also use LGKM.** Leaving raw LDS priming
before `s_memtime` followed by `lgkmcnt(0)` would drain it at the boundary.
For LDS the sequence is deliberately:

```text
initialize LDS, CTA barrier, load ready 0/1 and complete them
read beginning core/steady counters, finish their LGKM operations
issue raw priming reads 2..D                 <-- inside timed interval
MFMA + DS load ring + optional partial waits
lgkmcnt(0) final drain
read ending counters
```

There are no timer LGKM waits between raw priming and the first MFMA. The
code-object inspector follows that actual ordering. Priming and final drain
are therefore included/amortized, not claimed to be free. All initialization
and ready seeding precede core timing. Event launch time includes everything.
Global modes retain their previous VMEM priming placement.

## Implementation, evidence and reproduction

The common generator emits 30 entries per build (18 global + 12 LDS). It
keeps pending registers in one opaque asm, four raw-to-ready copies in all
shapes, fixed register-rotation phases and independent D/U. Actual ISA for
all seven measured builds (210 entries) is inspected for load/MFMA counts,
correct counter/threshold, no vector global loads in LDS entries, no scratch
or lane spill, and conservative waited FIFO lifetimes. Dynamic LDS is passed
at launch; static-LDS metadata is zero. Numerical testing remains OFF.

```sh
bash research/fp16-load-matmul/build.sh 2 32
python3 research/fp16-load-matmul/inspect_loops.py temp/fp16-load-matmul/d2-u32-lds1
python3 research/fp16-load-matmul/run_matrix.py --lds \
  --binary temp/fp16-load-matmul/d2-u32-lds1/probe \
  --output research/fp16-load-matmul/results/lds/d2-s1

# Intentional bank concentration:
bash research/fp16-load-matmul/build.sh 4 32 temp/fp16-load-matmul/d4-u32-lds4 4

python3 research/fp16-load-matmul/analyze.py research/fp16-load-matmul/results/lds
python3 research/fp16-load-matmul/analyze_lds.py
```

`results/lds/dD-sS/`: CSV/logs, full command manifests, AMD-SMI snapshots,
flags, hashes and inventories. `temp/fp16-load-matmul/dD-u32-ldsS/`: actual
executables/HSACO/ISA, generated header, metadata, extracted loops and compile
database. Previous v1 binaries are untouched; pre-extension source is archived
at `temp/fp16-load-matmul/pre-lds/source/`. No production integration.
