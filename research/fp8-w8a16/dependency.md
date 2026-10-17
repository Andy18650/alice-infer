# Four-bank dependency gap and staged VALU controls

User-directed follow-up. Two questions:

1. Does increasing the FP8 conversion operand ring from two to four banks
   remove the boundary bubble, before or after MFMA?
2. At what point does added work bring the bubble back in raw FP16: vector
   NOPs, unrelated work, source reads, or source writes?

**Findings:** four banks do not remove the bubble. More importantly, source
dependencies are **not necessary** to reproduce it: seven `v_nop`s or seven
independent integer VALUs per MFMA already produce a ~4-cycle/backedge cost;
eight produce ~8. Barriers-only and the shorter sampled groups do not.
Source-write experiments also expose compiler copies/NOPs and actual numerical
hazards, so their requested operation count must not be mistaken for actual ISA.

## Experiment / protocol

`dependency_probe.hip` is standalone; the original probe, NOP sweep and order
probe are unchanged. Native HIP O3/C++20, system ROCm 7.1 / Clang 21.1.8,
gfx90a; HIP logical device 0, 104-CU MI250X GCD. CPU and memory NUMA node 0,
**no model or attention context**, no clock/power overrides.

104 CTAs x 256 threads, 416 waves (nominal four waves/CU, not a measured
assignment trace). Register-resident activation/weights, one dependent FP32
accumulator chain, `v_mfma_f32_32x32x8f16`. Each wave repeats one 32x32 tile.
The same signed dyadic operands and independently indexed FP64 oracle as the
previous probes are used. At tested lengths 4096/8192 the expected outputs
are exactly representable through FP32 accumulation; **any nonzero error is
flagged invalid**, not allowed to disappear under a length-dependent tolerance.

Use U=4/8/32 MFMAs per backedge, all divisible by four so the four-bank ring
has no dynamic index/phase selection or extra backedge bank-copy machinery.
Warm every kernel, then condition with raw FP16 and both four-bank FP8 phases
for >=2 seconds. Two independent nine-round processes at 4096 MFMAs/wave,
seeds 1729/2718, plus three rounds at 8192 (31415). Each sample contains eight
HIP-event-timed launches; last-launch mean per-wave core-clock intervals give
cycles/MFMA. Output copies/reference checks are outside timing; all outputs
of the final launch in each sample are checked. GPU measurements are sequential.
The terminal timestamp is not an explicit final-MFMA completion fence.

Fit U=4 and U=32 to `C(U) = A + B/U`, holding U=8 out. B estimates exposed
cycles/backedge, while A absorbs common arithmetic and finite-length endpoint
costs. All valid fits predict the held-out point within 0.004 cycles/MFMA.
Both processes and the doubled-length control agree on the main thresholds
and boundary constants.

## 1. Two versus four banks

Each bank is four FP16 weights / two VGPRs per lane. All banks are explicitly
converted in the prologue, outside the core-clock body interval. For N banks:

```text
before, step j:
    bank[j % N] = convert(packed_weight)
    MFMA(bank[(j+1) % N])

after, step j:
    MFMA(bank[j % N])
    bank[j % N] = convert(packed_weight)
```

The earlyclobber constraints protect the ready source bank, and compiler
scheduling barriers preserve the phase. Actual register tracking verifies
**one intervening MFMA for two banks, three for four banks** between the last
conversion producer and its consumer, including across the backedge. No
hot-loop register-copy instructions are needed for these rings.

| Phase | Banks | U=8 cycles/MFMA, run 1 | Run 2 | Fitted B, run 1 / run 2 / 8192 control |
|---|---:|---:|---:|---:|
| Before | 2 | 65.530283 | 65.530276 | 12.000 / 12.000 / 12.000 |
| Before | 4 | 66.526250 | 66.526238 | 20.000 / 20.000 / 20.000 |
| After | 2 | 65.031252 | 65.031252 | 8.000 / 8.000 / 8.000 |
| After | 4 | 67.026247 | 67.026229 | 24.000 / 24.000 / 24.000 |

Four banks use 32 reported VGPRs rather than 28, with 16 AGPRs / 20 SGPRs
in both cases, zero private bytes and LDS. Their fitted steady-state floor
is still approximately 64 core cycles/MFMA. More separation does not fix the
bubble; the exposed per-loop cost increases in both phase orderings.

### Important codegen difference from the preceding order probe

Do **not** compare the new two-bank after constant of 8 directly with the
old order probe's 20 and attribute that change to ring depth. The generalized
ring prologue primes **all** banks through conversion and consumes the
activation as a compiler-lifetime input. That changes the load-wait/hazard
layout: these conversion loops have **no in-loop `s_waitcnt`**, but one
compiler-emitted `s_nop 0` per backedge, before their first MFMA. Before-phase
places it after the first conversion, after-phase before the first MFMA.
The previous order probe retains `s_waitcnt vmcnt(0)` at its target.

The matched two/four comparisons here share that new priming scheme and
recorded scalar layout. The bubble demonstrably exists without an in-loop
VMEM wait, but this does not uniquely identify a hardware latency or prove
which part of the older difference was caused by the wait. Original measured
sources/binaries are preserved; no benchmark history is silently replaced.

## 2. Staged controls on raw FP16

Requested operations per MFMA: **1, 2, 4, 5, 6, 7, 8**. Both before/after
orders are tested. The absent count 3 is not interpolated as a measured point.

| Kind | Work |
|---:|---|
| 0 | Natural raw MFMA, no scheduling barriers or extra instructions |
| 1 | Same compiler scheduling barriers as other variants, no emitted work |
| 2 | `v_nop`, vector-pipeline NOP, not the scalar NOP from the earlier sweep |
| 3 | Scratch `v_xor_b32`, its own dependent chain, independent of MFMA operands |
| 4 | Copy/read B halfwords into scratch using `v_xor_b32 scratch,0,Bword` |
| 5 | Value-preserving write B using `v_or_b32 Bword,0,Bword` |
| 6 | Same source-read control on A |
| 7 | Same identity-write control on A |
| 8 | Real paired sign flips of matching A/B halfword pairs; products unchanged |
| 9 | FP8 conversion ring described above |
| 10 | Kind 8 plus conservative `s_nop 3` after its work, a hazard control |

A = activation MFMA source, B = weight source. Reads/writes alternate the
operand's two dwords. The audit checks **physical register numbers**, not just
source-level expressions. Kinds 8/10 use counts 2/4/8: adjacent XORs flip
matching K positions in A and B, so the mathematical dot products stay the
same despite actual changes in the input bits.

### Bubble threshold

First sampled requested count with fitted B > 1 cycle/backedge, identical in
both nine-round processes:

| Added work | Before MFMA | After MFMA |
|---|---:|---:|
| `v_nop` | **7** | **7** |
| Independent scratch XOR | **7** | **7** |
| Read B | **7** | **7** |
| Identity-write B | **7** | **5*** |
| Read A | **6**† | **7** |
| Identity-write A | **6**† | **5*** |

For the three clean controls (`v_nop`, independent XOR, read B), both phases:

| Requested ops/MFMA | Fitted B, cycles/backedge | Approximate cycles/MFMA at U=8 |
|---:|---:|---:|
| 1, 2, 4, 5, 6 | ~0 | 64.12 |
| 7 | ~4 | 64.62 |
| 8 | ~8 | 65.12 |

Thus an MFMA-source dependency is **not required** for the boundary bubble.
Short source reads and identity writes also do not instantly introduce it.

The barrier-only controls move increment/compare to the loop tail but still
have B approximately zero. Therefore **tail placement of increment/compare
alone is not sufficient**, either. Natural raw MFMA keeps those instructions
near the first MFMA, as in the previous inspection.

† Before-A read/write has **two emitted in-loop waits**, `vmcnt(1)` before the
first source operation and `vmcnt(0)` after the first work group; the matching
before-B controls have one. The earlier threshold correlates with an extra
scalar instruction/layout difference, not necessarily a new intrinsic A-source
hazard. Do not label that four-cycle boundary difference as an established
register latency without isolating those waits.

*After-identity-write variants with >=2 requested operations contain **two extra
`v_mov_b32` copies per MFMA**, plus compiler-emitted `s_nop 1` on the first
U-1 intervals. Five requested ORs are therefore seven vector instructions,
not five. The copies rotate between two physical pairs even though the C++
source names a single operand. These are useful compiled implementations but
**not clean equal-instruction-count dependency experiments**. The audit
retains those copies, NOP wait sums and all scalar positions explicitly.

### Changed-data correctness hazard

Identity writes leave register bits unchanged, so passing their oracle does
not establish safe latency for arbitrary changed source data.

The paired real-flip test makes that distinction visible:

- **Before, no explicit spacing (kind 8): all nine variants fail exact-value
  checks**, at counts 2/4/8 and U=4/8/32. Some errors are small transient
  effects and some are large. The program records them as invalid, rather
  than fitting a misleading fast result or allowing an expanding relative
  tolerance to mask them at longer lengths.
- **After:** these variants pass, but actual code has compiler copies/NOPs:
  four extra copy MOVs per MFMA at counts 4/8, and `2 + 2/U` copies per MFMA
  at count 2. This is not evidence that the bare XOR sequence is safe in
  either ordering.
- **Conservatively spaced control (kind 10): all variants pass.** This adds
  `s_nop 3` (~16 nominal core cycles here) and is explicitly accounted as a
  different instruction stream, not treated as pure VALU-only work.

There are **6300 exact-valid samples** across all retained processes and
**189 invalid samples**, all from kind-8/before. Invalid timing rows remain
in raw CSVs for inspection but are excluded from every reported performance
fit. The driver prints `COMPLETE ... invalid_samples=...`, not a false claim
that all variants passed. CLI completion alone is not the correctness gate;
use the analyzer and per-row `valid`/`max_abs_error` fields.

## ISA / resource audit and conclusions

`audit_dependency_probe.py` checks all **309** executable entries: requested
work order, actual source-register relationships, physical two/four-bank
producer-consumer distance, prologue conversion counts, scalar positions,
extra bank-copy MOVs and implicit/explicit NOPs. All have zero private/LDS
bytes. Full metadata and individual loops are retained under `temp/`;
small machine-readable audit and raw/derived results are under `results/`.

The narrow supported conclusion is: **neither a short conversion dependency
gap nor MFMA-source writes are necessary for the bubble.** A sufficiently
long stream of even vector NOPs is enough in these finite loops. Compiler
wait placement and hazard handling change individual constants, so this is
not yet a unique branch/issue/accumulator-forwarding attribution. It does not
justify changing production or replacing the accepted arithmetic reference.

The subsequent user-directed [preheader + split-conversion experiment](splitting.md)
explicitly handles first-entry readiness and redistributes the boundary work.
It removes the fitted bubble in its primary two-bank case, without changing
or invalidating the preserved measurements above.

## Reproduce

```sh
bash research/fp8-w8a16/build-dependency-probe.sh
python3 research/fp8-w8a16/audit_dependency_probe.py
numactl --cpunodebind=0 --membind=0 temp/fp8-w8a16/dependency/dependency-probe 4096 9 1729
# Repeat: 4096 9 2718. Length control: 8192 3 31415.
python3 research/fp8-w8a16/analyze_dependency_probe.py
```

Stdout is CSV, stderr records device and invalid samples. Retained runs are
`results/dependency/run{1,2}.{csv,log}` and `longer.{csv,log}`. The analyzer
checks completeness, ensures all invalid rows belong to the known hazardous
kind/order, excludes them from fits, and reports held-out residuals. Hashes
bind the source/executable/extracted code object. No engine sources or CMake
files changed; engine tests were not rerun for this isolated experiment.
