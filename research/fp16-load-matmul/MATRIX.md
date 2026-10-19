# Active research matrix

The active implementation is **direct raw-ring consumption only**. There is
no ready-buffer/copy branch, no wait-free branch and no saved-module replay.

## Fixed properties

- gfx90a native FP16 x FP16 -> FP32 `v_mfma_f32_32x32x8f16`.
- One CTA, 256 threads, four wave64s; one AGPR accumulator bank per wave.
- Four private dummy 32x32 output tiles; no production/general GEMM API.
- Each step requests 4096 logical operand bytes across the four waves.
- D raw A+B slots are the entire operand pipeline: prime 0..D-2, issue n+D-1,
  wait, MFMA directly reads raw[n%D]. Zero operand copy instructions.
- Completion waits mandatory, final drain retained. Harvest guard is after
  ending timestamps. No additional arrival-lag stage.

## Structural axes

| Axis | Supported values | Configuration |
|---|---|---|
| Address mode | global `hot`, `l2`, `stream`; `lds-hot`, `lds-sweep` | runtime `--mode` |
| Instructions per A+B pair | 1 wide128, 2 wide64, 4 scalar32 | runtime `--loads 1|2|4` |
| Raw ring depth | any integer 1..16 | build argument 1 |
| Logical unroll | any integer 1..64, independent of depth | build argument 2 |
| LDS bank stride | 1,2,4,8; does not alter global modes | build argument 4 |

One fixed D/U/stride build contains **15 kernels: 9 global + 6 LDS**. The
batch runner selects 9 global cases by default, or 6 LDS cases with `--lds`.
If all bank strides are explored, globals are not four new cases each: there
are 9 unique global + 24 LDS mode/shape/stride combinations per D/U.

| Loads | Global opcode shape | LDS opcode shape |
|---:|---|---|
| 1 | one global_load_dwordx4 containing A+B | one ds_read_b128 |
| 2 | two global_load_dwordx2, A and B planes | two ds_read_b64 |
| 4 | four global_load_dword, four word planes | four ds_read_b32 |

D/U rotation uses P=D/gcd(D,U) phase bodies. Each U steps has a taken
conditional branch, but static code contains P*U MFMA slots. There is no
copying or dynamic indexing of unresolved registers. LDS partial waits cap
at 14 (all-ones 15 is avoided), so deep requested/effective thresholds can
differ; both are reported. This cap remains a constraint on interpreting
nominal depth, not an unsafe branch.

## Runtime sizes, data and measurement controls

- `--steps`: positive multiple of U; default 4096, logical K=8*steps.
- `hot`: one 1024-byte fragment per wave, 4 KiB total.
- `l2`: cyclic `--records`, default 256/wave = 1 MiB total, pre-read before timing.
- `stream`: no wrap, default steps+D records/wave including one padding record;
  minimum steps+D-1. Default completed 256 MiB eviction, outside timing.
- LDS allocation <=64 KiB: hot one padded record/wave; sweep default 16/stride
  records/wave. Direct local initialization/barrier outside timing, no VRAM
  transfer to LDS in the measurement. Stride1 is bank-balanced; other strides
  are explicit bank-concentration controls.
- `--pattern legacy|hash`, `--seed-a`, `--seed-b`: constructed finite FP16 data.
  Hash varies records/lanes/halves with independent A/B seeds.
- Exact full-output verification defaults ON, supports steps<=8192, includes
  poisoned outputs and guards. `--verify 0` is an explicit host-check opt-out,
  not a way to remove GPU waits. It permits longer timing-only runs.
- `--rounds`, `--warmup-ms`, `--device`, `--evict-mib`: measurement controls.
  Matrix runner binds CPU+memory NUMA0 and individually warms each case.

The primary timing is CTA max(end)-min(begin), with absolute wave timestamps
saved. Global cache names are working-set descriptions, not proof of hit rate;
logical payload bytes/cycle are not physical cache/NoC/HBM traffic.

## Removed axes

Ready-bank count, raw->ready copies, R1/R2 comparisons, optional completion
waits, and original-code-object/module dispatch are not part of the active
matrix. `--wait 1` is a legacy alias only; `--wait 0` errors. The constant-zero
`ready_banks` CSV column remains solely for historical schema compatibility.

Current build interface:

```sh
bash research/fp16-load-matmul/build.sh DEPTH UNROLL [OUTPUT] [LDS_STRIDE]
# Example: depth2, unroll32, bank-balanced LDS.
bash research/fp16-load-matmul/build.sh 2 32
numactl --cpunodebind=0 --membind=0 temp/fp16-load-matmul/d2-u32-lds1-direct/probe \
  --mode lds-sweep --loads 1 --pattern hash
```
