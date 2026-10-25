# One B representation for decode and prefill

User-approved follow-up to the [row-major-A study](README.md). Compare simple
weight representations with **both** a row-major-A GEMM and an FP8 adaptation
of the existing column-coalesced split-K GEMV strategy. Do not redesign small-M
GEMM or tune each format independently. No production/model-format change.

**First result:** plain N-contiguous B is the leading *common-format candidate*,
not yet a performance-neutral replacement. It gives the best decode rates on
six of eight 70B shard shapes; blocked N128 wins the two QKV cases by small
margins. MFMA-specific packing wins all 32 practical GEMM cases. Blocked N128
recovers a little GEMM performance but can substantially hurt decode. Keep all
four controls; do not standardize a permanent weight format from this alone.

## Formats and unchanged semantics

Logical multiplication is `A[M,K] * B[K,N]`. A is ordinary row-major FP16,
stride K. B is finite E4M3FN, scales are FP32 `s[ceil(K/128)][ceil(N/128)]`:
`B_real[k,n]=decode(B[k,n])*s[k/128,n/128]`. Accumulate raw K128 partials,
then ordered FP32 scale FMAs. No per-invocation A or B pack; B preparation,
upload, references and FP32->FP16 activation conversion are outside events.

|Name|Physical FP8 bytes, outer to inner|
|---|---|
|native|Existing `[K32 stage][N128 tile][K8 slice][lane][16 bytes]`, MFMA-specific permutation|
|plain|`[k][n]`, N contiguous, row stride padded to128 columns|
|block128|`[K128 group][N128 tile][k_inside][n_inside]`, N contiguous within 16 KiB tiles|
|block256|`[K128 group][N256 tile][k_inside][n_inside]`, N contiguous within 32 KiB tiles|

Native/plain pad K to32; blocked formats pad K to128. Initial supported logical
K is multiple-of32 or zero; M/N tails supported. **All timed 70B shard dimensions
are already multiples of128 and256 in N**, so their formats have equal payload
sizes; padding differences appear only in correctness cases.

One execution-weight allocation is passed unchanged to both consumers in every
process. Full GPU unpack-to-logical roundtrip before execution, GEMM + decode
interoperability gates, and full byte-for-byte B immutability after execution.
The harness retains extra validation buffers; it is not a production peak-memory
measurement or a proposal to keep two execution weight copies.

## Kernels: deliberately bounded adaptations

**GEMM:** keep the previous 128x128x32/four-wave/16 KiB LDS/scaled-bank core,
row-major A loader, MFMA/scale sequence and waits. Native's complete 26,716
instruction bytes exactly match the previous row-major-A baseline; see
`results/common-b/checks/native-baseline-identity.json`.

Common formats fetch four N4 DWORD rows rather than the native single x4
packet. An eight-permutation4x4 byte transpose feeds the unchanged FP8 expansion.
Four B b64 scatters replace two b128 stores. v41 is a temporary while consumer
addresses are dead, restored before publication/raw reads. Reuse s20:21 for
transient row bases; no extra persistent VGPRs. All variants256 VGPR/0 AGPR,
private0, cap2; SGPR59 native versus66 common. The adapter is correct, not a
claim that four narrow loads/scatters are the best possible common-B staging.
The inherited q0 balance is not retuned for the additional producer work.

**Decode:**64-thread CTAs, N4 outputs/lane, four independent K accumulation
streams, K128 split units. Store raw partials; a second kernel applies scales
and combines in ascending K-group order. Native packing needs an in-register
transpose to yield the same N4 records; common formats do not. This reuses the
production CM strategy, **not its BF16 binary or bandwidth guarantee**: FP8
expansion, already-FP16 x, K128 splits and ordered scaled combine differ.
Partial kernels are spill-free,54 VGPR/30 SGPR native,58/24 plain,54/26 blocked;
combine is spill-free too. No decoding/CTA sweep or fused combine tuning.

Throughput comparisons cannot be read as proof of a physical HBM bottleneck.
Conversion/issue work, split count and locality can limit this FP8 prototype.
Its maximum observed useful weight rate is about839 GB/s, below the previously
measured BF16 roofline results; those are different data paths/byte counts.

## Shapes and measurement

70B means **TP4/8 local projections**, never full-width single-GCD model results:

|Projection|TP4 `(N,K)`|TP8 `(N,K)`|
|---|---|---|
|QKV|2560,8192|1280,8192|
|O|8192,2048|8192,1024|
|Gate/up|14336,8192|7168,8192|
|Down|8192,7168|8192,3584|

Decode M1; GEMM M16/128/512/2048. Also retain104/208/312 CTAs at K8192/32768
and the1664-CTA large regression case. These are **synthetic, per-rank kernel
benchmarks**: no model/attention context, collective, or full-model TP claim.
One gfx90a GCD HIP0/BDF0000:31:00.0, CPU+memory NUMA0, automatic clocks, HIP
O3/C++20, no BLAS/CK reference. Two paired seeds, reversed case/format order
in repeat2,200 ms selected-path warmup and nine checked eager samples/process.
Unlike the previous pack/no-pack comparison, only one path is timed per process;
interop/primitive checks happen before warmup, not between timed samples.

GEMM useful TFLOP/s=`2*M*N*K/full-event-time`. Decode useful weight GB/s=
`N*K/full-event-time`: includes **both partial and scale/combine kernels**,
not activation traffic/scratch/scales in the numerator. Requested padded weight
GB/s is also recorded for decode, but neither is a physical-HBM counter
measurement or hot/cold-cache certification. For GEMM the CSV's two weight-GB/s
fields are only one weight-matrix volume/time, not its total repeated CTA reads. Decode raw clocks cover the partial kernel only;
GEMM raw clocks cover its body, while event denominators cover complete work.

## Results

Means of two process median rates; all eight decode shapes:

|70B shard|Native GB/s|Plain GB/s|Block128 GB/s|Block256 GB/s|
|---|---:|---:|---:|---:|
|TP4 QKV|377.190|391.263|396.588|336.517|
|TP4 O|369.878|399.471|397.943|372.497|
|TP4 gate/up|678.708|838.862|701.757|674.947|
|TP4 down|583.846|707.132|523.569|499.664|
|TP8 QKV|206.090|218.456|219.186|214.172|
|TP8 O|205.606|235.111|231.475|227.458|
|TP8 gate/up|587.203|685.344|522.052|497.292|
|TP8 down|504.825|542.101|528.825|453.649|

GEMM at M2048, useful TFLOP/s:

|70B shard|Native|Plain|Block128|Block256|
|---|---:|---:|---:|---:|
|TP4 gate/up|124.376|110.449|113.748|110.498|
|TP4 down|126.381|114.672|115.857|113.912|
|TP8 gate/up|124.714|112.409|113.658|112.400|
|TP8 down|121.533|109.566|111.294|108.923|

Full tables: [decode.md](results/common-b/decode.md), [gemm.md](results/common-b/gemm.md).

Interpretation:

- Plain improves decode over native packing by23.6%/21.1% at TP4 gate/up/down,
  and16.7%/7.4% at TP8. It loses about9–11% GEMM rate on those M2048 cases.
- Block128 gains about3% GEMM rate over plain on TP4 gate/up M2048, but loses
  16.3% decode rate there. Decode losses versus plain reach26.0% at TP4 down
  and23.8% at TP8 gate/up. Block256 offers no broad compensating win.
- Long-K non-model controls can favor Block128: large-grid93.1/85.5/104.1/89.1
  TFLOP/s for native/plain/block128/block256. Its two event medians are
  16.58/17.80 ms, so preserve the variance; do not extrapolate this into a
  precise universal benefit. Native312-CTA/K32768 also varies3.88/3.71 ms.
- Coalesced local loads do not guarantee good cross-block memory distribution.
  Large power-of-two macro-tile strides are a possible concern, not an isolated
  cause proved by this study. More producer work does not alone explain every
  shape either: blocking changes locality while keeping the conversion work.

**Next candidate:** plain N-contiguous weights, retaining native packing as a
control. Improve the GEMM's common-B load/LDS mapping before accepting its
current prefill tax. Keep the decode strategy as the small-M reference; any
FP8 decode optimization should be separated from a format comparison. Do not
make duplicated persistent B copies or runtime B repacking the workaround.
No additional tuning was performed after these results.

## Validation, replay and preservation

3552 checked timed matmul invocations +1088 checked interop matmuls across544
executable-hashed processes. Every process:65024 exact finite FP8 half outputs,
16256 byte transposes, full B roundtrip, both consumers, immutable B, logical
references, all-C NaN poison/finite/guards and scratch/scale/A guards. Small
references full, large sampled; integer/16 and short finite-code domains only,
not arbitrary-IEEE/model numerical certification.

`checks/summary.json`:4,661,248 layout/address cases,272 stage/full-tail control
cases (each with two N strides), B pending/copy/store/pointer bounds, and
unchanged generated/native compute/read/fence/scale projections. Native scale
step's *external* SGPR binding is normalized; fixed MFMA/FMA operands unchanged.
Zero through29 stages,63/64/65 and2048-stage traces. Actual ISA/resource checks
and3552 independent raw-clock/event-span checks,3552 low-word-duplication
negatives. Unsupported K/nonempty build paths rejected. Bounded checks, not a
formal ISA/memory-return model. Derived tables and CPU/ISA checks reproduce
exactly. HIP builds are warning-free and clangd main-file check passes. Standalone
GPU-header check has no compiler/AST errors but24 interactive DefineOutline
failures and generic ODR warnings for four single-TU kernel definitions; recorded
as tooling limitations, not misreported as a clean header-tool exit.

```sh
for format in native plain block128 block256; do
  bash research/practical-gemm/build_common.sh "$format" <fresh-root>/"$format"
done
python3 research/practical-gemm/run_common.py --build-root <fresh-root> --output <fresh-results>
python3 research/practical-gemm/verify_common.py <results>
# Derived-report replay: copy samples.csv to a fresh directory first.
python3 research/practical-gemm/run_common.py --analyze-only --output <copied-results>
```

Measured binaries/objects and pilot checks remain under
`temp/practical-gemm/common-b-v1/`. Recorded source/native/commands/raw results
and one `checks/` set live in `results/common-b/`. Compilation-source mirrors
preserve relative imports/includes; rebuild historical code with its snapshot
script into a fresh path, never overwrite or relabel old measured binaries.
No production/CMake/model/session/system/package/clock-policy changes.
