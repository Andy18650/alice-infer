# Four body calls versus four kernel launches: net 4 GiB cost

User request: test the intermediate control and count the whole-grid/kernel
boundary latency, rather than inferring practical value from isolated-window
bandwidth. **Splitting is not a universal latency win.** It gives a small
repeatable streaming-throughput benefit for A, but single-operation latency
is less robust. Repeating the body inside one grid generally does not recover
A's isolated 1 GiB-window rate.

## Three equal-payload paths

- `full`: one kernel, one read_body over 4 GiB, one checksum per wave.
- `body4`: one kernel, four read_body calls over successive 1 GiB windows,
  four separate checksum regions. No grid or CTA barrier; waves can move
  to the next window independently. Runtime outer loop, pragma unroll 1.
- `split4`: four 1 GiB kernels on the same stream, four checksum regions.
  Stream ordering enforces whole-grid completion between kernels. No CPU
  synchronize call is inserted between those four timed kernels.

Total useful bytes are 4 GiB in every case. All launches, output reductions,
checksum stores and grid-boundary costs lie inside the GPU event interval.
CPU wall time additionally includes event submission, host launch calls,
final synchronization and event-time retrieval. Validation/D2H, logging and
the two case warmups are outside both intervals; warmups explicitly finish
before wall timing starts. These are benchmark operation costs, not an
engine/model inference latency claim.

## Conditions

Native O3 HIP/C++20, gfx90a, HIP GPU 0 / PCI 0000:31:00.0, CPU+memory NUMA
node 0, no model/context or clock overrides. 104×256, phase-only scheduling
barriers. Six cases: A/chunks4, B/chunks4, B/chunks16, both chunk orders.
One 4 GiB allocation per process, all three paths jointly shuffled across
nine rounds, following brief and 30 s sustained warmups.

Streaming control: twelve complete 4 GiB operations per timed batch, with
two rich-clock process runs and one basic-monitor control. Latency control:
one complete 4 GiB operation per measurement, two basic-monitor runs and
one rich-clock run. Iteration counts are stored and summarized separately;
no pooling across the one/twelve-operation protocols.

## Streaming result: small net A benefit survives boundary costs

Sustained-stage **CPU wall ms per 4 GiB**, amortized over twelve operations,
process 1 / process 2 (rich telemetry):

| Reader/layout/chunks | Full | Four bodies, one kernel | Four kernels |
|---|---:|---:|---:|
| A/grid/4 | 3.0597 / 3.0565 | 3.0799 / 3.0753 | **3.0171 / 3.0192** |
| A/adjacent/4 | 3.0336 / 3.0303 | 3.0478 / 3.0420 | **2.9820 / 2.9778** |
| B/grid/4 | 3.2121 / 3.2099 | 3.2164 / 3.2169 | **3.1562 / 3.1481** |
| B/adjacent/4 | 3.0677 / 3.0663 | 3.0924 / 3.0911 | 3.0691 / 3.0545 |
| B/grid/16 | **3.0945 / 3.0950** | 3.1040 / 3.1008 | 3.2084 / 3.2180 |
| B/adjacent/16 | 3.0724 / 3.0750 | **3.0579 / 3.0551** | 3.0842 / 3.0802 |

For adjacent A, the difference of process medians is ~1.7%; paired-by-round
median wall savings are 39/55 us, winning 7/9 and 9/9 rounds. Basic polling
also retains a gain (8/9 rounds, median paired saving ~50 us). Grid A wins
9/9, 9/9 and 8/9 across those three processes. This is a modest real net
streaming gain, not the much larger inference one might make from isolated
1 GiB-window rates. B results are layout/chunk dependent.

## Single-operation latency: benefit is smaller and irregular

Sustained-stage **CPU wall ms for one complete 4 GiB operation**, three
runs (basic / rich / basic):

| Reader/layout/chunks | Full | Four bodies, one kernel | Four kernels |
|---|---:|---:|---:|
| A/grid/4 | 2.9360 / 2.9176 / 2.9239 | 2.9552 / 2.9532 / 2.9577 | 2.8978 / 2.9069 / 2.9004 |
| A/adjacent/4 | 2.9126 / 2.9000 / 2.9082 | 2.9512 / 2.9199 / 2.8981 | 2.8640 / 2.9112 / 2.8941 |
| B/grid/4 | 3.0914 / 3.1121 / 3.0667 | 3.1190 / 3.1225 / 3.1011 | 2.9922 / 3.0675 / 2.9633 |
| B/adjacent/4 | 2.9341 / 2.9102 / 2.9090 | 2.9210 / 2.9386 / 2.9099 | 2.9161 / 2.9429 / 2.9579 |
| B/grid/16 | 3.0387 / 3.0442 / 3.0457 | 3.0915 / 3.0592 / 3.0554 | 3.1648 / 3.1978 / 3.1971 |
| B/adjacent/16 | 3.0188 / 3.0395 / 3.0351 | 2.9960 / 2.9962 / 2.9865 | 3.0622 / 3.0268 / 3.0477 |

Do not inflate the benefit using only differences of medians. Adjacent A's
four-kernel path wins just 5/9, 3/9 and 4/9 paired rounds, with median paired
wall savings +38/-2/-2 us. Grid A wins 6/9, 4/9, 5/9 (+26/~0/+9 us).
One-grid/four-body A generally loses; B/adjacent/16 benefits from that path,
while B/grid/16 strongly disfavors four launches.

Absolute rates differ between streaming and single-operation protocols,
which also have separate process allocations and workload cadence. These
runs are not a controlled isolation of iteration count alone. The useful
comparison is the three paths matched **within each process/protocol**.
No production kernel fission is justified as a universal optimization by
these data.

## Assembly and correctness caveats

- Six multi-body entries retain an actual scalar outer-loop comparison
  against four, with backward control flow, not four fully unrolled copies.
- No s_barrier or device function-call instruction is emitted; read_body
  is inlined, and no grid/CTA synchronization was added.
- Complete 4-wide or 16/64-scalar load batches precede the first wait;
  no old load/wait-zero/consume serialization or changed cache flags.
- Inlining into the outer loop changes generated code/resources. Reported
  registers, full → body4: A/grid 48→46, A/adjacent 48→62, B/grid/4 40→49,
  B/adjacent/4 70→112, B/grid/16 137→150, B/adjacent/16 262→400.
  The last retains capacity one CTA/CU and zero scratch, but full-kernel
  accumulator moves rise to 143 reads / 145 writes. B/grid/16's captured
  SGPR lane-spill segment also grows. It is not a spill-free or purely
  scheduling-only control.
- Thus the result does not uniquely prove which part of kernel restart,
  grid admission, traversal reset, memory-controller phase or codegen is
  responsible. Merely writing four calls is not instruction-equivalent to
  four separate kernels.
- All four checksum outputs are consumed by the host, preventing dead-store
  elimination of earlier traversals. Empty and partial-window fixtures,
  shifted affine checksums and every timed output pass in all six runs.
  Operand extents are checked for the full four-window operation.

## Reproduce / artifacts

```sh
research/vram-coalescing/build.sh
# Streaming-throughput control, twelve operations/batch:
research/vram-coalescing/run.sh temp/vram-coalescing/new-body-control \
  default body-control 9 phase rich 12
# Single-operation latency control:
research/vram-coalescing/run.sh temp/vram-coalescing/new-body-latency \
  default body-control 9 phase basic 1
```

Tracked: `results/body-control-{a,b,basic}.csv`,
`body-latency-{a,b,c}.csv`, `body-control-paired.csv`, resource and ISA reports.
Full source/binary snapshots, raw rounds, telemetry and each multi-body
function's assembly are under matching `temp/vram-coalescing/` folders.
Follow-up summarizer snapshots preserve original reports and record iteration
counts explicitly. Original baseline and production code remain unchanged.
