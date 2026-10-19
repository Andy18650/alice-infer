# One-CTA FP16 operand pipeline — direct raw ring only

**Active implementation:** one CTA, 256 threads/four waves, native gfx90a
FP16 x FP16 -> FP32 MFMA, one AGPR accumulator bank per wave. No ready
buffers, operand copies, wait-free specialization or original-module replay.
Exact output verification is enabled by default.

See **[MATRIX.md](MATRIX.md)** for the complete remaining configuration matrix.
For fixed depth/unroll/LDS-stride, a build contains **15 kernels: nine global
and six LDS**. The only operand-consumption schedule is:

```text
prime raw slots 0..D-2
issue operand n+D-1
wait for completion of raw[n%D]
MFMA directly consumes raw[n%D]
```

D is now the **total operand-slot count**, not raw slots plus ready banks.

## Build and run

```sh
# depth, unroll, optional output directory, optional LDS bank stride
bash research/fp16-load-matmul/build.sh 2 32
python3 research/fp16-load-matmul/inspect_loops.py temp/fp16-load-matmul/d2-u32-lds1-direct
numactl --cpunodebind=0 --membind=0 temp/fp16-load-matmul/d2-u32-lds1-direct/probe \
  --mode lds-sweep --loads 1 --pattern hash --steps 4096 --rounds 9

# Deliberately concentrated bank layout
bash research/fp16-load-matmul/build.sh 4 32 temp/fp16-load-matmul/d4-u32-lds4-direct 4
```

Build-time controls: depth 1..16, unroll 1..64 independently, LDS stride
1/2/4/8. There is no fifth ready-bank argument and no generator ready selector.
D/U rotation uses P=D/gcd(D,U) phase bodies: U MFMAs between taken conditional
branches, P*U static slots, no pending-register copies.

Runtime controls:

- `--mode hot|l2|stream|lds-hot|lds-sweep`.
- `--loads 1|2|4`: one 128-bit, two 64-bit, or four 32-bit reads per A+B pair.
- `--steps`, `--records`, `--rounds`, `--warmup-ms`, `--device`, `--evict-mib`.
- `--pattern legacy|hash`, `--seed-a`, `--seed-b`, `--verify 0|1`.

Steps must be divisible by unroll; default 4096 means K32768. Verification
supports steps<=8192. `--verify 0` explicitly skips host checks, not GPU waits.
The legacy `--wait 1` spelling is accepted only for safe old command manifests;
other values are rejected. `--module` and `--symbol` are removed.

## Sources / layouts

| Mode | Address progression |
|---|---|
| hot | Global, one fixed fragment/wave, 4 KiB total |
| l2 | Global cyclic window; default 1 MiB total, pre-read before timing |
| stream | Global no-wrap; default steps+D records/wave, completed eviction before launch |
| lds-hot | Fixed local fragment/wave, directly initialized before timing |
| lds-sweep | Cyclic local window, default full 64 KiB LDS |

| Loads | Global | LDS |
|---:|---|---|
| 1 | global_load_dwordx4, A+B together | ds_read_b128 |
| 2 | two global_load_dwordx2, A/B planes | two ds_read_b64 |
| 4 | four global_load_dword, word planes | four ds_read_b32 |

All shapes request 4096 logical payload bytes per four-wave step. LDS has
32 banks * 512 dwords; stride1 is aggregate bank-balanced, larger strides
are bank-concentration controls. No VRAM->LDS fill is measured: the CTA
initializes local memory directly. Cache names are working-set descriptions,
not counter-proven residency. Allocated bytes are not physical traffic.

## Correctness / timing

The independent exact integer/dyadic reference reconstructs all 4096 MFMA
outputs; outputs are poisoned with leading/trailing guards. Hash data vary
records/lanes/halves with independent A/B seeds. Failures exit nonzero.

Completion waits are mandatory: vmcnt for globals, lgkmcnt for LDS, with
reported conservative LGKM cap14. LDS raw priming follows start-clock
completion waits so timers cannot drain it. Final drain is inside timing;
accumulator-harvest guard, output stores and verification are outside it.
The primary timing is **max(wave_end)-min(wave_begin)**; raw clocks are saved.

```sh
python3 research/fp16-load-matmul/run_matrix.py --lds \
  --binary temp/fp16-load-matmul/d2-u32-lds1-direct/probe \
  --output temp/fp16-load-matmul/my-lds-sweep
python3 research/fp16-load-matmul/check_active.py
```

The matrix runner selects nine global cases by default or six LDS cases with
`--lds`, independently warming every selected case. Current regression tools
inspect actual ISA/register lifetimes and reconstruct source operand sequences;
they are not hardware timing simulators. `check_active.py` checks full outputs,
removed-option rejection and surviving-loop equivalence.

Current build names use `dD-uU-ldsS-direct`. A compile database is generated
there for clangd. No engine/CMake/production files or global editor settings
are changed.

## Historical evidence, not active branches

[Global baseline](global-baseline.md), [LDS staging](LDS.md),
[verification](VERIFICATION.md), [copied single-ready failures](SINGLE_READY.md),
and [direct-raw validation](DIRECT_RAW.md) retain their original results.
Their checkpoint-specific replay drivers were removed; recover them from
`e260a39` or the earlier `3986b13` checkpoint in a separate worktree.
Raw data and older binaries are preserved. Analysis tools retain historical
CSV support; current `ready_banks`/`wait` fields are fixed at 0/1 for schema
compatibility, not selectable axes. [CULLING.md](CULLING.md) records pruning.
