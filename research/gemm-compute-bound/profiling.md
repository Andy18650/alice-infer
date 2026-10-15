# Actual GEMM hardware-counter profiling

## Scope and protocol

The existing profiler is usable once its existing components are configured
correctly. No installation, profiler-source patch, library copy, system setting,
clock/power setting or production change was needed. This supersedes the earlier
AQL-initialization blocker for the tested configuration, not for every profiler
feature/version combination.

Profiled actual wide CK V3 GEMM at **M=4096,N=6656,K=8192/32768**, BF16 inputs,
FP32 accumulation/output, 416 CTAs of 256 threads. One gfx90a GCD (HIP 0), native
O3 C++20, CPU+memory NUMA node 0, no model/attention context. Affine, B-panel and
AB-panel use the same logical inputs/shape/compute geometry. Packing is outside
the dispatch. There is no split-K. The executable's numerical/guard/repeat and
graph correctness checks passed in each profiler invocation.

Exact measured executable SHA256:
`d6c6998353ed3686efbbe7a21e68506599bbe7c271fb951471d9463d8c378080`.
Copies of the executable, relevant sources, compiler flags, commands, collector
configs, full traces/counters and analysis logs are retained in
`temp/gemm-counter-profile/`.

1. Two broad **12-pass** captures collect blocks 2/7/10/11/12/13/14/15/16/17/18,
   one capture per K, all three storage variants. This covers arithmetic,
   wavefronts, execution, LDS, instruction/scalar/vector caches, TA/TD and L2.
   Roofline microbenchmarks and PC sampling were deliberately excluded.
2. **Three independent single-pass runs per K** collect the eight counters in
   [profile-core.yaml](profile-core.yaml) together. This avoids estimating
   utilization/wait ratios from counters measured in different replays.
3. Three additional single-pass long-K runs select **m-fast** to control the
   previously observed affine n-fast regression. These profile all three formats,
   not just the favorable affine row. Broad memory metrics were not repeated
   for m-fast.

With `--rounds 1 --repeats 1`, per-kernel iteration **[3]** selects the third
n-fast warm-up dispatch; **[9]** selects the third m-fast warm-up. Raw traces
independently verify these ordinal selections for every single-pass run. These
are ordinary eager dispatches, not graph execution or full-operation batches.
The existing benchmark performs a preceding register-reference warmup, then
three kernel warmups, not the clock audit's two seconds of actual-GEMM warmup.
Consequently profiled durations are diagnostic only and do not replace the
previous repeated unprofiled rates. The full profiler's joined timestamps average
across replays; memory rates derived from them are approximate.

## Main results

Counter-derived MFMA utilization below follows AMD's gfx90a formula:

```
100 * SQ_VALU_MFMA_BUSY_CYCLES / (4 * 104 * GRBM_GUI_ACTIVE)
```

**Important:** AMD describes this busy-cycle counter as **emulated**, so this is
an instruction-cycle occupancy estimate against measured active core cycles,
not an independent transistor-level observation of MFMA busy/stall state.
Dependency waits use `100 * SQ_WAIT_ANY / SQ_WAVE_CYCLES`; both counters use
quad-cycle units, so their factors of four cancel. Entries are medians of three
same-pass samples.

| K | Order | Affine A/B MFMA util | Affine A, panel B | Panel A/B |
|---|---|---:|---:|---:|
| 8192 | n-fast | 92.67% | 87.69% | 94.92% |
| 32768 | n-fast | 86.88% | 90.01% | **97.70%** |
| 32768 | m-fast | **96.57%** | 88.39% | **97.61%** |

Long-K AB-panel n-fast spans **97.693-97.704%** across the three captures.
The GRBM-active-cycle / dispatch-time ratio is ~1389 MHz, consistent with the
previous ~1385 MHz wave-clock audit. This supports the previous conclusion:
actual GEMM can remain near arithmetic-bound in the core cycles it receives.
It is not a fixed 146 TFLOP/s hardware ceiling or a new unprofiled performance
record.

The affine m-fast recovery independently confirms that **zero spills alone is
not sufficient**: the same spill-free affine implementation changes from
86.88% to 96.57% when only CTA traversal changes. B-panel remains lower and is
not a spill-free comparison. No intervention removed its spill in isolation.

### Spill-related execution evidence

At long K, n-fast:

| Storage | Scratch bytes/thread | VMEM instructions | Dependency-wait wave cycles |
|---|---:|---:|---:|
| affine | 0 | 13,764,608 | 3.26% |
| b-panel | **16** | **14,623,232** | **4.93%** |
| ab-panel | 0 | 13,764,608 | 2.51% |

B-panel issues **6.24% more VMEM instructions** (6.20% at K=8192), consistent
with its already-audited prologue spill stores and K-loop private reload. This
shows live execution overhead rather than merely reserved private bytes.
The measured wait difference is compatible with extra dependencies, but is not
proof that spilling alone explains the entire performance difference.

The issue-wait metric is ~79-81% even for the good kernel. **That is not 80%
MFMA idle time.** These are wave-level inability-to-issue counts; the matrix
pipeline can remain occupied while a wave waits to issue its next instruction.
Do not sum overlapping wave/unit metrics into a kernel idle-time budget.

## LDS and cache/memory findings

All six broad-capture rows report **SQ_LDS_BANK_CONFLICT=0**. That is the observed
counter result, not a proof about every possible shape or LDS transaction.

For fixed n-fast:

| K | Storage | L2 lookup hit rate | L2-to-HBM read payload | Approx. read payload rate |
|---|---|---:|---:|---:|
| 8192 | affine | 83.55% | 0.507 GB | 157 GB/s |
| 8192 | b-panel | 83.84% | 0.505 GB | 149 GB/s |
| 8192 | ab-panel | 83.49% | 0.503 GB | 159 GB/s |
| 32768 | affine | 85.71% | 2.017 GB | 147 GB/s |
| 32768 | b-panel | 85.36% | 2.015 GB | 152 GB/s |
| 32768 | ab-panel | 85.03% | 2.013 GB | 163 GB/s |

L2 hit rate uses `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)`. Read payload uses
64 bytes per `TCC_EA_RDREQ_DRAM_sum`; the captured total 32-byte read request
counter is zero. This is **counter-based interface payload**, not all physical
HBM-wire traffic, ECC/protocol bytes or complete read+write bandwidth. These
rates are far below the separate ~1.4 TB/s streaming reference; the large GEMM
is not saturating the HBM read interface in these captures.

AB-panel payload also matches a simple resident-batch reuse model very closely.
With 104 CUs and one resident CTA/CU, 26 N tiles make a nominal batch cover four
M tiles; 16 M tiles require four such batches. Reading A once and B once per
batch predicts:

```
(4096 + 4 * 6656) * K * sizeof(bf16)
K=8192:   503,316,480 bytes; counter:   503,316,608 bytes
K=32768: 2,013,265,920 bytes; counter: 2,013,266,432 bytes
```

This is strong consistency with effective L2 sharing inside the resident batch,
not a measured timeline guaranteeing synchronized batch admission. It does not
prove a cache-index/channel map. Small differences among formats are retained;
B-panel does not have a dramatically larger HBM payload that would explain its
slower arithmetic rate by bulk bandwidth alone.

## Validation and profiler limitations

- Every captured format has exactly **1664 SQ_WAVES** and the expected BF16
  operations: `512 * SQ_INSTS_VALU_MFMA_MOPS_BF16 == 2*M*N*K`.
  These checks pass for every single-pass sample, not only the joined table.
- The executable is untouched; no instrumented kernel replaces it.
- Compute 3.3.0 is used with an existing SDK backend reporting 1.3.2 / ROCm
  7.14 build metadata, against system HIP/HSA 7.1. Successful selected counter
  checks do not validate all mixed-version features.
- Some Compute-derived fields are blank. Its wavefront UI uses SPI_CSN_WAVE,
  which reports zero here despite SQ_WAVES correctly reporting 1664. Do not use
  that row as the launch count. Raw counters and explicit checked formulas are
  the basis of the results above.
- Profiler SGPR and architected/accumulation-register metadata differ from the
  previously verified ELF/HIP resource fields. Keep ELF/HIP authoritative for
  register/residency analysis; do not blindly add or substitute those fields.
  Scratch 0/16/0 and LDS 64 KiB agree with the prior code-object audit.
- The tool labels this MI250X GCD as MI210 and lacks VBIOS metadata. Its nominal
  peak percentages are not the clock-adjusted reference. SPM reports unsupported
  on gfx90a; no PC sampling, GUI or multi-GPU profiler validation is claimed.
- Broad metrics span replayed runs. Only core metrics have three same-pass
  confirmations; do not assign statistical precision to one-pass memory rates.

## Reproduce

Existing installations (no package changes required):

```
Compute: /home/andyhu/.local/rocm-prof/bin/rocprof-compute
Python:  /home/andyhu/.local/rocm-prof-venv/bin/python
SDK:     /home/andyhu/.cache/uv/archive-v0/2EEHNuXZame80Ad1/_rocm_sdk_core
AQL lib: /home/andyhu/.local/rocm-prof/lib
```

For the single-pass core check (use a freshly built matching benchmark if the
retained snapshot is unavailable):

```sh
SDK=/home/andyhu/.cache/uv/archive-v0/2EEHNuXZame80Ad1/_rocm_sdk_core
export PATH="/home/andyhu/.local/rocm-prof-venv/bin:$SDK/bin:$PATH"
export PYTHONPATH="$SDK/lib/python3/site-packages"
export LD_LIBRARY_PATH=/home/andyhu/.local/rocm-prof/lib
HIP_VISIBLE_DEVICES=0 "$SDK/bin/rocprofv3" \
  -i research/gemm-compute-bound/profile-core.yaml \
  -d "$PWD/temp/new-gemm-core-profile" --kernel-trace --output-format csv \
  --kernel-include-regex 'compute_bound.*entry.*Storage' \
  --kernel-iteration-range '[3]' -- \
  numactl --cpunodebind=0 --membind=0 \
  "$PWD/temp/gemm-counter-profile/snapshot/bench" \
  --suite explicit --impl v3_256x256x64 --gm 16 --gn 26 --k 32768 \
  --rounds 1 --repeats 1
```

For the broad capture, the retained `temp/gemm-counter-profile/run.sh` invokes
Compute using `temp/profiler-usability/run-rocprof-compute.sh`; that wrapper
configures the same installed components, explicitly selects the SDK library,
and uses the supported ROCM_VER=7.1 metadata fallback for Ubuntu's /usr install
without `.info/version`. Exact single-pass commands/environment and executable
hash are in `results/profile-plan.json`. Curated data are `profile-summary.json`,
`profile-core-samples.csv` and the two `profile-k*-counters.csv` under `results/`.
Expanded raw runs and summarization scripts remain under durable repo-local
`temp/gemm-counter-profile/`. No subsequent commit is implied by profiling.
