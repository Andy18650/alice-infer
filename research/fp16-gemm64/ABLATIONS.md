# Performance-only single-class instruction ablations

User-authorized unsafe diagnostic, NOT a GEMM implementation or numerical
validation. The active verified winner, its source generator and four GPU
entries are unchanged. This opt-in script patches copies of the measured
native executable/code object, not the active binary or production code.

## Three changes, independently

Only the16 static sites in the steady-state16-step bulk are changed. For
K32768,4064 of4096 logical K8 steps execute that bulk; startup and32-record
tail retain original waits/barriers/MFMAs. Each variant starts from the same
original bytes; variants are never combined.

| Variant | Replacement | Unchanged candidate instructions |
|---|---|---|
| MFMA proxy | One8-byte v_pk_fma_f32 | VM24 and barrier unchanged |
| VM wait | s_waitcnt vmcnt(24) -> s_nop0 | MFMA and barrier unchanged |
| Barrier | s_barrier -> s_nop0 | MFMA and VM24 unchanged |

The MFMA proxy reads the same four distinct A/B VGPRs. For raw0:

```asm
v_pk_fma_f32 v[0:1], v[0:1], v[2:3], v[0:1]
```

Raw1 uses4:5 and6:7. Third source duplicates A. Destination is current A's
VGPR pair, which following LDS reads refill. This is NOT a simulation of
AGPR-C reads/writes, MFMA execution, accumulator dependencies or full register-
file pressure. Operands are interpreted as FP32 bit patterns, not unpacked
FP16 values. The discarded proxy result has no numerical meaning.

All replacements have exactly the original instruction width: no shifted
instructions/branch targets, kernel descriptors, resource allocations or
host changes. Source/native SHA checks, exact16-site manifests and byte-diff
bounds prove every nonselected code-object byte is unchanged. Re-extracting
the embedded object from each patched executable confirms the actual host
program contains the patched image; all copies are disassembled again.
S_NOP0 is a real issued instruction/wait state, not zero-cost deletion.

## Results

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, one CTA /
256 threads, K32768, raw2/16 LDS slots. No model/context/clock overrides;
input packing/upload outside timing. Two randomized process-order repeats,
nine timing samples/process,200 ms warmup. Medians, run1/run2:

| Variant | Core cycles/logical K8 step | Entire body us | Core/steady MHz |
|---|---:|---:|---:|
| Baseline | 93.325 / 93.043 | 227.360 / 226.560 | 1682.146 / 1681.980 |
| MFMA -> packed FMA | 87.074 / 87.163 | 212.000 / 212.320 | 1682.382 / 1682.468 |
| VM wait -> NOP | 93.427 / 93.229 | 227.680 / 226.880 | 1682.064 / 1681.843 |
| Barrier -> NOP | **79.573 / 79.086** | **193.920 / 192.640** | 1680.901 / 1681.727 |

Divisor stays4096 logical iterations/wave, not the number of surviving MFMAs
or aggregate four-wave instruction count. Timing field `cycles_per_mfma` is
retained from the probe schema but means logical-step cost for these runs.
Event intervals and raw per-wave clocks remain in CSV. The inherited
`logical_GFLOPs` field is not actual achieved GEMM throughput for ablations,
especially when the matrix instructions have been removed.

## Interpretation and limitations

- Barrier removal gives the largest observable reduction:13.75/13.96 cycles
  per logical step. It removes rendezvous/skew constraints and may alter
  wave interleaving, throughput and memory races. This is NOT an isolated
  measurement of the hardware barrier instruction's latency.
- VM wait removal gives no improvement (slightly slower). The explicit VM24
  wait is not visibly limiting this workload/ring in isolation. This does
  not prove memory latency is irrelevant or the original wait costs zero:
  the replacement NOP has its own cost, and remaining LGKM waits, rendezvous,
  issue queues or other dependencies can mask a removed wait.
- MFMA proxy saves6.25/5.88 cycles, far less than the ~64-cycle arithmetic
  reference. Much matrix execution can overlap other work. The changed
  execution unit, removed accumulator dependency/traffic and proxy's own
  VGPR write mean this is not MFMA's intrinsic latency measurement.
- Similar core/steady ratios do not explain these changes as simple core-
  frequency shifts. No unique cache/NoC/port/issue bottleneck is established.
- Savings are nonadditive; no combined ablation or further tuning was tried.

Verification is explicitly OFF for all72 timing samples, including baseline
for protocol parity (`valid=-1`). The original baseline was separately
verified once before comparison. No ablation is certified numerically valid;
VM/barrier removals race shared data, and proxy removes the matrix work.
No claims that every unsafe variant was observed failing: numerical checks
were deliberately not run on them. These are performance experiments only.

## Files / reproduction

```sh
# Requires an existing current verified build. Never modifies it.
python3 research/fp16-gemm64/ablate.py
```

Defaults require fresh `temp/fp16-gemm64/ablations/` artifacts and
`results/ablations/` records; choose fresh --artifacts/--output for replays.
Patches, GPU PCs, replacements/encodings, hashes, ELF sections, bulk ISA,
commands, raw clocks and summaries: `results/ablations/`.
Full patched ISA/executables: `temp/fp16-gemm64/ablations/<variant>/`.
The three variants remain outside active code generation/default runners.
No package, compiler, system, production/CMake/model or clock-policy changes.
