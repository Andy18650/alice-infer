# Retained accumulator-chain and loop-shape probes

These two standalone sources were left untracked during the earlier
compute-bound/clock research. They are retained as exploratory controls, not
new GEMM implementations or evidence of an automatic MFMA-to-VALU interlock.
They are independent of the main CK benchmark and need no CK headers.

- `mfma_chains.hip`: BF16 MFMA16x16x16 and32x32x8,1/2/3/4/8 accumulator
  chains,exact per-thread expected sums,per-wave core/steady counters.
- `mfma_loopshape.hip`: dependent/constant-C AGPR and architected-VGPR shapes,
  one MFMA/backedge,with an explicit conservative exit gap before VALU reads.

The source comments record hypotheses from that campaign. They are not a
validated hardware writeback model. In particular, the implemented nodep C
input is the invariant2.0 (some introductory wording says loop-varying), and
one104-CTA grid is not itself a hardware-ID proof of placement. Successful
exit-gap controls do not establish arbitrary MFMA-result read safety.

## Reproduce

Use a fresh output directory; do not overwrite `temp/raw-mfma/` historical
executables/logs:

```sh
mkdir -p temp/mfma-probes/<fresh-name>
for name in mfma_chains mfma_loopshape; do
  hipcc -O3 -std=c++20 --offload-arch=gfx90a -Wall -Wextra \
    research/gemm-compute-bound/$name.hip -o temp/mfma-probes/<fresh-name>/$name
  numactl --cpunodebind=0 --membind=0 temp/mfma-probes/<fresh-name>/$name
done
```

Each program has fixed4096 MFMA iterations,104x256 threads,two-second
selected-kernel warmups,and9 rounds of16 event-timed launches per case.
Forward/reverse case order is built in. Verification checks every thread;
no model/context,automatic clocks,no production/system configuration changes.
The constant low-magnitude operands are not a real-data sustained-GEMM clock
reference; see `clock-verification.md`.

Commit review built both sources into `temp/leftover-review/mfma/` and ran
all their built-in correctness gates successfully on HIP0/BDF31:00.0,
CPU+memoryNUMA0. Those fresh validation runs do not replace or augment the
older curated GEMM performance claims. Historical MFMA binaries/logs remain
under `temp/raw-mfma/`; no new unqualified performance conclusion is added.
