# Shallow K: deep loops still amortize entry/drain costs

User requests rerunning the performance model at much shallower K. **Numerical
correctness testing stays off.** This is a timing-only rerun of the exact
waited/unwaited executables from [waits.md](waits.md), not a new kernel.

## Setup

Single CTA, four N waves, fixed fragment-zero addresses, same depth variants
1/2/4/8/16 and 32-MFMA unroll. Native HIP O3/C++20, gfx90a, HIP 0,
CPU+memory NUMA node 0, no model/context or clock/power modifications.

Sweep 32/64/128/256/512/1024/4096 MFMA steps, i.e. **logical K = 256 through
32768** (eight K elements per instruction). These are repetition counts:
only fragment zero is fetched, not additional unique data as K grows.
32 is the minimum length supported by this unchanged unroll32 binary.

Two nine-round processes per length/mode, seeds 1729/2718. Length order and
waited/unwaited process order reverse between runs; case/cache order remains
randomized. Both in-dispatch pre-read and completed-eviction preparations run.
The existing two-second warm-up is unchanged. No output data are read back;
no numerical references, guards, fixtures, eviction checksum checks or
correctness gates run. Source/executable/code-object hashes match the earlier
wait-removal artifacts before measurement. No rebuild or kernel change.

## D8 result

Median over the two nine-round runs, **core cycles/MFMA**, in-dispatch pre-read:

| Steps | Logical K | FP8 waited | FP8 no wait | FP16 no wait |
|---:|---:|---:|---:|---:|
| 32 | 256 | 109.641 | **110.594** | **248.312** |
| 64 | 512 | 102.859 | 103.047 | 188.000 |
| 128 | 1024 | 99.391 | 99.590 | 156.074 |
| 256 | 2048 | 97.752 | 97.809 | 142.822 |
| 512 | 4096 | 96.799 | 96.871 | 132.980 |
| 1024 | 8192 | 96.424 | 96.452 | 132.306 |
| 4096 | 32768 | 96.106 | **96.113** | **128.563** |

For short FP16 cases, process medians have noticeable spread: at 32 steps
261.344/246.469, at 128 steps 158.992/154.711, at 256 steps
139.504/143.523. Do not interpret every decimal of the pooled median as a
stable architectural cost. All individual samples and run medians are kept.

## Revised model: include the finite-length term

The prior byte-rate model describes the asymptote, not the whole short loop:

```text
T_body(S) approximately T_entry/drain + S*T_steady
T_steady approximately max(64, four-wave logical request bytes / R_effective)
R_effective approximately 32 logical bytes/core-cycle (fitted, not a hardware counter)
```

For pre-read FP8 D8 without waits, the observed residual
`T_body - 96*S` is exceptionally consistent:

| Steps | Residual core cycles |
|---:|---:|
| 32 | 467.0 |
| 64 | 451.0 |
| 128 | 459.5 |
| 256 | 463.0 |
| 512 | 446.0 |
| 1024 | 463.0 |
| 4096 | 464.5 |

Thus a useful empirical model is **`T_body approximately 96*S + 460`**, or
**`cycles/MFMA approximately 96 + 460/S`**. The waited D8 path similarly has
roughly 430–450 cycles beyond `96*S`. Removing the wait is still not the
explanation of the 96-cycle steady slope.

**Yes, deep K amortizes a real finite-length cost. No, that does not explain
away the remaining 96-cycle steady-state floor.** FP16 also shows substantial
amortization, but its residual above `128*S` is less constant, approximately
2.3–4.4k cycles in this sweep; do not force it into one exact affine fit.

## Other depths and an important warm-up limitation

For unwaited FP8 D1/2/4/16, residuals above `96*S` are roughly **13–15k cycles**,
not the D8 ~460. At 32 steps those variants cost ~500–560 cycles/MFMA; at
4096 they are ~99–100. Much of their apparent long-run disadvantage versus
D8 is compatible with a larger amortized entry cost, not a different steady
slope. See `results/shallow-k/summary.txt` for all depths.

**The inherited warm-up favors FP8 D8 and the resident anchor.** Its exact
selection is `format<0 || (format==1 && depth==8)`; FP16 and other FP8 depths
are not individually prewarmed. This was retained to rerun the same binaries
and protocol, but it is a real limitation when comparing short-loop cases.
Do not attribute the 13–15k residual to queue depth or a specific hardware
latency. Code/cache residency, placement, and entry effects are possible and
have not been isolated. A future equally prewarmed comparison would be a
separate experiment, not something silently changed in this rerun.

The resident anchor itself has short-interval sensitivity, especially with
pre-read preparation. Its raw results are retained; the eviction-prepared
anchor trends from ~69 cycles at 32 steps to ~64.0 at 4096.

## Scope of the timing and interpretation

- Seed loads and raw-queue issue priming occur **before** the core timestamp;
  remaining in-flight work can still complete during timing. The final full
  load drain **is** inside it. The measured intercept is therefore not a
  measurement of the entire kernel prologue, nor a uniquely isolated drain.
- Event launch time is retained separately. It includes pre-read and launch/
  prologue/epilogue costs and is not substituted for core-body timing.
- Allocation sizes and packed N-stream pitch vary with `steps` in the existing
  harness, although addresses are fixed within each timed loop. This is not
  an independently fixed-address-placement sweep.
- No profiling counters, cache-hit proof or specific physical port assignment.
  The ~32 B/core-cycle rate remains a logical-request throughput model.
- No new warm-up policy, unroll, layout, instruction or correctness change.

## Artifacts / reproduction

`results/shallow-k/`: all 28 process CSV/log pairs, original mode-specific
hashes/flags, per-run and pooled timing summaries. Same binaries under
`temp/fp8-w8a16/wait/{waited,unwaited}/`.

```sh
for s in 32 64 128 256 512 1024 4096; do
  for mode in waited unwaited; do
    numactl --cpunodebind=0 --membind=0 \
      temp/fp8-w8a16/wait/$mode/wait-probe "$s" 9 1729
  done
done
# Second sweep seed 2718, reversed process/length order.
python3 research/fp8-w8a16/analyze_shallow_k.py
```

The analyzer consumes timestamps only. It never inspects computed output or
runs numerical correctness tests.
