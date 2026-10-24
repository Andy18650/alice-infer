# Balanced all-wave FP8 B producer

User-directed experiment: FP16 A, finite E4M3FN B, FP32 accumulation/output.
Packing is an external pre-GEMM contract and may change for both operands.
This is the historical first experimental variant, **not the FP16 default**.
Later user-directed [copy/early-load and MFMA-interleaving changes](FP8_SCHEDULE.md)
are measured separately; none of the original data below is overwritten.

**Result:** correct, but slower than the FP16 MFMA-first kernel at104/208/312
CTAs with the first conversion-burst schedule. Large-grid useful throughput is
essentially unchanged. No follow-up scheduling optimization was silently tried.

## Distribution and packing

Selected128x128x32 only. Every wave produces one K8 slice of BOTH operands:
wave0 K0..7, wave1 K8..15, wave2 K16..23, wave3 K24..31, within each K32 stage.
This also redistributes A rather than keeping the original two A-producing
waves. The balanced scheme has no wave-role branch and uses the same16 fixed
producer VGPRs as FP16. Its two source pointer pairs use v40:43; producer A
uses v44:51, packed B uses v52:55 and expands in place to v52:59. Raw operand
slots v0:39 and the four64x64 output rectangles stay unchanged.

Physical layouts (FP16 A and FP8 B are separate allocations):

```text
A: [K32 stage][128-row tile][K8 q][64-row half][lane][8 FP16 values]
B: [K32 stage][128-col tile][K8 q][lane][16 FP8 values]
```

For wave `w`, lane `l`, define `t=l%32`, `k=stage*32+w*8+(l/32)*4`.
Coordinates below add the CTA's tile-row/tile-column origin:

```text
A first x4:  rows t,t+32; each has K=k..k+3        (16 bytes/lane)
A second x4: rows t+64,t+96; each has K=k..k+3     (16 bytes/lane)
B only x4:   cols t,t+32,t+64,t+96; K=k..k+3      (16 bytes/lane)
```

Each global instruction accesses adjacent16-byte vectors across lanes; the
two A packets are1024 bytes apart. Per wave/stage: two A DWORDx4 loads plus
one B DWORDx4. Per CTA/stage:8 KiB A+4 KiB B, versus8+8 KiB for FP16. These
are requested input bytes, not physical HBM traffic or a cache-reuse multiplier.

Global A byte base is `tile_row*8192+w*2048+l*16`; B is
`tile_column*4096+w*1024+l*16`. Their stage strides are respectively
`padded_M*64` and `padded_N*32`. All M/N/K padding is zero.

Each wave writes its K8 A slice to LDS at `w*2048+l*16` and +1024;
its expanded B slice goes to `8192+w*2048+l*16` and +1024. The same16 KiB
LDS layout and the same2D-wave consumer addresses/read offsets are preserved.
Both operands are still reused between two compute waves, then within each
wave's four MFMAs. No duplicate B conversion is done by the consumers.

## Conversion and first schedule

The exact earlier finite E4M3FN bias7 sequence is reused:

1. `v_perm_b32`, selectors0x01040004/0x03040204, insert zero low bytes.
2. `v_pk_ashrrev_i16`, shift1 with explicit `op_sel_hi:[0,1]`.
3. AND0xbfffbfff.
4. `v_pk_mul_f16` by packed256 (0x5c005c00).

Four instructions per two output values; eight per packed FP8 dword. Four
source dwords expand to eight FP16 dwords using32 VALUs/wave/stage. Expand
source words3,2,1,0, with high-output conversion first, to avoid clobbering
unread packed input. Nonzero restored FP8 subnormals are normal FP16 values;
intermediate FP16 subnormals require IEEE denormal mode (same explicit build
flag as the baseline). No quantization scale or output compensation. The two
NaNs0x7f/0xff and FNUZ semantics are outside this finite conversion contract.

The existing first current-q0/C00 MFMA remains immediately after head VM0.
Then: two A LDS stores, the B conversion burst, two B LDS stores, three future
global loads/pointer advances, and the remaining three current-q0 MFMAs.
Publication/retirement waits/barriers, rolling raw reads, q3 parity and peeled
loop/tails are retained. Conversion has not yet been interleaved with the
remaining current matrix work.

Native two-stage loop:147 instructions versus81;32 MFMA,6 global x4,
8 LDS b128 writes,16 reads,16 EACH perm/shift/AND/multiply, four barriers,
six waits and three loop-control instructions. Extra second-pointer steps
are also counted. No spills/NOPs/AGPR transfers in the loop.

Metadata:140 VGPR/64 AGPR/46 SGPR versus136/64/46 for FP16;16 KiB LDS and
zero private segment unchanged. Runtime reported capacity remains3 CTAs/CU.
The extra address inputs cost four metadata VGPRs despite unchanged fixed
producer payload. This is a format+producer+packing experiment, NOT an isolated
measurement of conversion cost: pointer/address/register/locality changes
also belong to the measured difference.

## Validation

- Independent logical-coordinate CPU packing walk:524288 padded elements,
  default and finite-code fixtures, M129/N131 and K boundaries0..129.
- GPU in-place expansion uses the very same generated fixed-register expansion
  sequence as the producer:4064 input vectors/65024 output half values. All254
  finite FP8 encodings occupy every one of16 byte positions, with independently
  varying finite neighbors. Exact half bits, including negative zero and all
  signed subnormals, match the separate integer FP8-to-half oracle.
-18 full-reference GEMM launches at K1/7/8/9/31/32/33/63/64, M129/N131,
  two rounds each, B spanning all254 finite codes and A randomly signed1/16.
  K<=64 guarantees all partial sums remain exactly representable in FP32;
  reference sums independently decoded FP8 quanta in int64, not packed decoding.
-312 matched checked GEMM launches:30 FP16 functional/seed/boundary cases,
  30 FP8 cases and252 timing launches. Default A/B[-4..4]/16 inputs are exactly
  representable in E4M3FN, so both variants reference the same original logical
  matrices with no quantization-error tolerance. Large references remain
  sampled; every C is scanned finite after NaN poison and guards checked.
- Generated compute/raw-read/fence projection is identical at29 stage counts;
  actual native MFMA operands/order identical, raw LDS destinations/offsets/order
  identical after normalizing changed physical address-input VGPRs. Loads,
  stores and conversion deliberately differ and are NOT covered by an equality
  claim. Projection is not hardware emulation; packing/GPU gates are separate.
- New FP16 build with the shared host/preprocessor bridge has a whole `.text`
  byte-identical to the preserved measured FP16 executable. Normal default
  still builds the FP16 MFMA-first pipeline, not this experimental variant.

## Matched measurements

Native HIP O3/C++20 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0,block256,
no model/context/clock override. Full useful FLOPs divided by full kernel HIP
event time, packing/H2D/reference excluded. Same shared eager protocol,
200 ms selected-kernel warmup, two counterbalanced nine-sample process repeats.

Useful TFLOP/s, run1/run2:

|CTAs|K|FP16 before|FP8 B after|FP8 clock-adjusted global efficiency|
|---:|---:|---:|---:|---:|
|104|8192|119.920/120.085|113.448/113.522|72.76/72.80%|
|104|32768|127.407/127.476|120.750/120.562|77.74/77.64%|
|208|8192|132.586/132.636|129.104/129.295|84.51/84.59%|
|208|32768|137.972/137.918|134.038/134.037|88.10/88.09%|
|312|8192|137.208/137.750|135.398/135.503|89.40/89.48%|
|312|32768|141.693/141.416|139.021/138.966|92.14/92.14%|
|1664|32768|146.486/146.513|146.473/146.455|97.40/97.42%|

M/N are1664/1024,3328/1024,4992/1024 and4096/6656 respectively. Lower-grid
losses persist despite lower requested input bytes; larger scheduling coverage
largely hides the extra producer work. This is consistent with a longer producer
critical path, not proof of one unique bottleneck, intrinsic conversion cost,
or a verdict on a separately optimized W8A16 schedule. Grid counts do not prove
uniform CTA placement per CU.

## Reproduction and provenance

```sh
# To rebuild THIS historical algorithm, use its immutable compilation sources:
bash research/native-gemm/results/fp8-b/after/source/build_fp8.sh <fresh-build-directory>
numactl --cpunodebind=0 --membind=0 <build>/verify-fp8
python3 research/native-gemm/audit_fp8.py --output <fresh-audit-directory>
python3 research/native-gemm/run_fp8.py --after-build <build> --output <fresh-A-B-directory>
python3 research/native-gemm/verify_artifacts.py
```

Recorded FP8 builds: `temp/native-gemm/fp8-b/s32-t128`, results `results/fp8-b/`.
Current build_fp8.sh defaults to a later schedule in its own namespace;
the historical run_fp8.py/audit_fp8.py pin these original source snapshots.
Only128x128x32 ordinary GEMM supported; no claim of256/StageK64 integration.
Normal build defaults move to `temp/native-gemm/fp16-bridge/s32-t{128,256}`;
both .text payloads match the old measured FP16 code exactly.
The shared driver explicitly allows and records resource differences for
this format test (not for historical scheduling-only tests), retains104/208/
312 and the large grid, snapshots actual sources/drivers and checks executable
hashes before88 processes. Extra finite-code/primitive gates are separate from
those312 launches. Data, commands, resources, clocks, source/native fingerprints,
gate logs and FP16 bridge proof are retained; normal FP16 remains the default.

Artifact handling note: an ELF section-dump command without an output argument
rewrote the temporary FP16 baseline and bridge `.hsaco` containers. No measured
executable was modified. Both original code objects were re-extracted from the
untouched executables and verified against their already-recorded SHA256 before
restoring them. No rebuild or manifest relabeling; archived/native hashes all
pass. The compared `.text` payloads were identical. Future dumps must use a
separate output container or a non-mutating ELF reader.

HIP O3 builds are warning-free; Python/shell syntax, current/archive hashes
and LSP diagnostics pass. clangd check-mode on the conversion test reports one
interactive ExtractFunction-tweak failure for a selected continue statement,
not a compiler/AST diagnostic; normal kernel check-mode reports0 errors.

No production/CMake/model/session/system/package/clock changes or library GEMM.
