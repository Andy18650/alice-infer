# One-CTA global DWORDx4 ->VGPR ring,1/2/4 waves

User-directed next step: measure global-to-VGPR loading at the previously
selected width4, separately from LDS staging. No width/CTA-count/window-depth
sweep or MFMA. Earlier measured kernels/binaries remain unchanged.

## Fixed workload

- One CTA with64/128/256 threads: **1/2/4 waves**.
- **32 slots x4 VGPRs/lane** of payload, i.e.128 payload registers/lane.
  Native kernel metadata reports136 VGPRs,24 SGPRs, zero LDS/scratch.
- Fixed **1 MiB global source total**, divided into disjoint1MiB/waves regions.
- Each `global_load_dwordx4` fully coalesces64 lanes x16 B =**1 KiB/wave**.
-32 loads per outer group; source offset advances32 KiB and wraps within
  each wave's region. Four immediate offsets0/1024/2048/3072 per4 KiB group;
  eight such groups per outer iteration. Source mask is explicitly scalarized
  before timing. No M0 or LDS operations.
- Opaque volatile asm owns all32 output vector tuples and source/control
  state across the loop. **No periodic wait, consumer, checksum, payload copy,
  barrier or NOP in the loop.** One terminal VM0 outside the issue interval
  makes final register capture safe.
- Native audit finds32 disjoint4-register result tuples, not overlapping
  source-address v8:v9;53 steady instructions =32 loads +16 address +5 control.

Ring32 is an initial measured choice, not a demonstrated optimal depth or
hardware limit. Result registers are continually overwritten after32 loads;
there is no per-slot readiness/retirement policy. Payload pattern repeats
by1024-byte fragment, so repeated writes to each slot have identical bits,
including reordered late returns. Nothing consumes intermediate values.
This is a synthetic loading probe, not a proof for arbitrary changing data
or a replacement for operand dependency management in a real GEMM pipeline.

## Timing / verification

Native HIP O3/C++20 gfx90a, HIP0/BDF31:00.0, CPU+memory NUMA0, one CTA,
no model/context/clock-policy overrides. Same fixed1 MiB source for all wave
counts; no cache counters certify residency or physical traffic.

Default outer groups32768/waves keep **1 GiB requested bytes/CTA/sample**
in all cases,1024 source traversals. Loads/wave are1,048,576 /524,288 /262,144
for1/2/4 waves. An unmeasured full-source traversal plus200 ms selected-kernel
warmup precedes sampling. As in prior studies, H2D output initialization is
after warmup; any initial cold source pass is included in the long stream.

CTA issue cycles =max(wave issue-end)-min(wave begin). Completion uses the
same first begin and latest completion-end, after terminal VM0. Steady-clock
makespans supply us; measured core/steady ratios ~1681.6–1681.9 MHz, not an
assumed core clock. Final ring output stores are outside both intervals.
Full HIP event additionally includes kernel setup/harvest/output/completion.

**One all-wave load step contains waves native load instructions**, returning
waves x1 KiB. `cycles_per_wave_group_load` divides the CTA makespan by loads/
wave, NOT aggregate instructions or bytes. Requested B/core-cycle uses the
full1 GiB divided by that same makespan. Do not compare this period directly
with a four-wave DWORD-to-LDS step without accounting for its different bytes.

Final output checks all32 slots, every active lane and all four DWORDs plus
32 output guards:32/64/128 KiB of ring data at1/2/4 waves. Repeated payload
is not a source-stride/last-iteration oracle. ISA audits the address grouping
and ring ownership. Six fill/source-wrap smoke checks plus54 timing samples
all pass. No spills/lane transfers or compiler-generated payload moves inside
timing. Outputs contain every final slot, not just a checksum or first word.

## Results

Two counterbalanced process repetitions (1/2/4, then4/2/1), nine checked
samples each after at least200 ms selected-kernel warmup. Process medians,
run1/run2:

| Waves | Threads | Bytes/all-wave step | Core cycles/step | Requested B/core-cycle | Requested GB/s | Issue body us |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 1024 | 16.0226 /16.0236 | 63.9097 /63.9058 | 107.4911 /107.4688 | 9989.12 /9991.20 |
| 2 | 128 | 2048 | 32.0446 /32.0446 | 63.9108 /63.9110 | 107.4911 /107.4808 | 9989.12 /9990.08 |
| 4 | 256 | 4096 | 64.0783 /64.0788 | 63.9218 /63.9213 | 107.4911 /107.5066 | 9989.12 /9987.68 |

The aggregate requested rate stays ~**63.9 B/core-cycle/~107.5 GB/s**: one
wave already matches the rate observed with four. More waves increase the
all-wave step period in proportion to its payload, not the total bandwidth.
This is evidence for a shared limit under this configuration, not proof of
which return path/cache/register resource limits it or a universal64 B/cycle
architectural roofline. Earlier tiny L1-hot/vector-load controls need not
share this1 MiB streaming source behavior. No further tuning is attempted.

The earlier1-CTA pure DWORD-to-LDS stream was ~58.9 requested B/core-cycle;
this VGPR case is somewhat faster, but width, destination and instruction
mix differ. That difference is not an isolated cost of LDS writes or proof
that a register-staged GEMM producer would win.

## Files / replay

```sh
bash research/lds-staging/build-vgpr.sh                 #optional fresh build path
python3 research/lds-staging/inspect_vgpr.py temp/lds-staging/vgpr-ring
numactl --cpunodebind=0 --membind=0 temp/lds-staging/vgpr-ring/probe \
  --waves 1 --rounds 9 --warmup-ms 200
#--waves2/4 for the other two cases; --groups overrides equal-payload default.
python3 research/lds-staging/run_vgpr.py --output <fresh-results-directory>
```

Exact commands, raw per-wave clocks, summary, source/native hashes and native
loop/metadata: `results/vgpr-ring/`. Actual binary/full ISA and CDB:
`temp/lds-staging/vgpr-ring/`. Build uses standalone HIP O3, no CMake or
production/model/system/package/clock-policy changes. Status is appended to
`research/research-log.md`; earlier sources, measurements and binaries intact.
