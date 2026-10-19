# One-CTA FP16 operand pipeline: global and LDS

Gfx90a research probe: **one CTA / 256 threads / four waves**, native
`v_mfma_f32_32x32x8f16`, FP16 inputs, FP32 accumulation, one AGPR bank/wave.
Four dummy 32x32 output tiles, not a generalized GEMM API.

**Exact result verification is now enabled by default.** The default **two
ready banks** pass the [verification suite](VERIFICATION.md). The optional
[single-ready diagnostic](SINGLE_READY.md) passes hot inputs but **fails
record-varying inputs even with waits enabled**; it is deliberately preserved
as a failing diagnostic. The new [direct-raw path](DIRECT_RAW.md), selected with R=0, eliminates both
ready banks and copies and passes the completed verification study.

## Build and run

```sh
# Build args: depth, unroll, optional output, LDS stride (default1), ready banks (default2).
# R=0: direct raw ring, no ready storage/copies.
bash research/fp16-load-matmul/build.sh 2 32 \
  temp/fp16-load-matmul/d2-u32-lds1-verified-r0 1 0
python3 research/fp16-load-matmul/inspect_loops.py temp/fp16-load-matmul/d2-u32-lds1-verified-r0

numactl --cpunodebind=0 --membind=0 \
  temp/fp16-load-matmul/d2-u32-lds1-verified-r0/probe \
  --mode lds-sweep --loads 1 --wait 1 --pattern hash --steps 4096 --rounds 9

# Single-ready diagnostic. Copying is retained, next MFMA immediately consumes it.
bash research/fp16-load-matmul/build.sh 2 32 \
  temp/fp16-load-matmul/d2-u32-lds1-verified-r1 1 1
```

Depth: 1..16. Unroll: 1..64, independently selectable. Steps must be divisible
by unroll; default4096 means K32768. Exact verification currently supports
steps<=8192. `--verify 0` explicitly opts out for performance-only runs.
Build names include `-verified-rR` to preserve all older measured binaries.

Flags include `--mode`, `--loads 1|2|4`, `--wait 0|1`, `--records`, `--steps`,
`--rounds`, `--warmup-ms 200`, `--device 0`, `--evict-mib`, `--pattern
legacy|hash`, `--seed-a`, `--seed-b`, `--verify`, `--help`. The legacy pattern
preserves historical +/-0.125 inputs; hash varies record/lane/half, magnitude
and sign, with independent A/B seeds. Every selected configuration is warmed
individually, not a favored subset.

## Modes and load shapes

| Mode | Source / progression |
|---|---|
| hot | Global, one invariant fragment/wave, 4 KiB total |
| l2 | Global cyclic window, default256 records/wave =1 MiB, pre-read before timing |
| stream | Global no-wrap; default `steps+depth+ready_banks` records/wave, completed256 MiB eviction before each launch |
| lds-hot | One local fragment/wave, initialized directly before timing |
| lds-sweep | Cyclic local window, default full64 KiB LDS, initialized directly before timing |

The global cache labels are working-set descriptions, not hit-rate proofs.
`active_bytes` reports the allocated/configured window including any padding,
not actual unique traffic. LDS modes do not fill from VRAM; direct initialization
and the CTA barrier precede timing. Each wave has a private A+B stream.

| Loads/pair | Global | LDS | Byte layout within wave record |
|---:|---|---|---|
| 1 | dwordx4 | b128 | lane*16*S, adjacent A0,A1,B0,B1 dwords |
| 2 | two dwordx2 | two b64 | lane*8*S; A at0, B at512*S |
| 4 | four dword | four b32 | lane*4*S; planes0/256*S/512*S/768*S |

S=1 globally, build-time LDS stride locally. Fragment span1024*S bytes/wave;
all forms request4096 payload bytes/four-wave step. R1/R2 retain four copy
VALUs; R0 has zero copies.
Stride1 distributes aggregate pressure over32 banks, stride4 over8; capacity
32*512 dwords. This is not a proof of every wide-read service phase. See
[LDS.md](LDS.md) for the bank model, conflict control and stage findings.

## Schedule, safety and timing

D means complete raw A+B slots; R is the ready-bank count.
- R1/R2 seed 0..R-1 and prime raw R..R+D-2. Step n consumes ready[n%R],
  issues n+D+R-1, waits, and copies raw n+R into ready[n%R]. R2 inserts one
  intervening MFMA before consumption. R1's varying-data failure is preserved.
- R0 primes raw 0..D-2. Each step issues n+D-1, waits for raw[n%D], and
  consumes that slot directly with MFMA. There is no separate ready storage,
  no copy, no operand-delay NOP and no extra arrival-lag stage.

Global partial waits: vmcnt(loads*(D-1)). LDS: lgkmcnt(min(loads*(D-1),14));
requested/effective values are printed, avoiding all-ones15. Initial/final
waits remain with `--wait 0`, and unsafe matches are not safety guarantees.
Mathematical/guard failure exits2; argument/execution error exits1.

LDS raw priming is inside timing **after** both start-clock completion waits,
so timer LGKM activity cannot drain it. Global raw priming retains its prior
before-timer placement. Core CTA span is max(end)-min(begin), not wave median;
all absolute core/steady wave clocks are saved. Final drain is timed, output
stores and verification are not. The current build adds a common conservative
accumulator-harvest guard after both ending clocks, not between load completion
and MFMA. Event timing includes initialization and that tail.

The opaque asm reserves raw/ready/address VGPRs explicitly. Even R1 keeps
v4..v7 reserved to isolate the diagnostic. R0 starts raw storage at v0 and
releases the eight reserved ready registers. With L=lcm(D,max(R,1)), P=L/gcd(L,U)
phase bodies give U MFMAs between taken conditional branches; static code has
P*U slots, reported explicitly. No compiler pending-register rotation.

`inspect_loops.py` inspects all30 entries for counts, register mapping, counter
choice and waited FIFO lifetimes; it is not a hardware hazard simulator.
`inspect_operand_sequence.py` checks source-level operand progression under
ideal load completion; it does not establish hardware readiness.

## Results / verification

- [Historical global baseline](global-baseline.md): hot one/two/four loads
  ~64/128/96 cycles. Original sources and binaries remain preserved.
- [LDS study](LDS.md): raw D2 +two ready banks reach ~64 hot; wide sweeping
  reads ~65. Intentional stride4 stays ~128 despite deeper queues.
- [Verification](VERIFICATION.md): 531 waited cases exact, including96 original
  code-object launches; 324 further checked timings exact. Fresh timings shift
  slightly upward; no bit-identical timing claim. 8/12 unsafe cases fail.
- [Single ready](SINGLE_READY.md): every tested varying-data case fails R1;
  R2 controls all pass. Hot data hide the issue. Copies remain in both.
- [Direct raw](DIRECT_RAW.md): 1224 checked positive launches pass; R1 negative
  controls still fail with the common harvest guard. D1 direct hot LDS is
  ~70 cycles, D2 ~65 with only two total operand slots. No fallback arrival
  stage was needed. Global hot one/two/four loads measure ~64/128/85 cycles.

The runners save raw CSV/logs, command manifests, hashes and inventories in
`results/`; generated headers, code objects/ISA and metadata live in each
`temp/fp16-load-matmul/dD-uU-ldsS-verified-rR/`. `single_ready.py` intentionally
exits nonzero for the failing diagnostic. Historical verification runners
still target their preserved `...-verified/` paths; explicit build output
arguments can reproduce those names with R2.

A generated compile database supports clangd; use its directory explicitly
if workspace arguments override the scoped default. No global editor settings,
engine, CMake, model fixtures or production blocks are changed.
