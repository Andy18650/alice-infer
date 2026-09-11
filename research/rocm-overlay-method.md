# Study note: the ROCm overlay method, and the 6.2.4 question (2026-09-11)

Context: Ubuntu 26.04 is a Canonical-native ROCm island (see
`rocm-tarball-feasibility-ubuntu2604.md`) — no AMD-official builds target
this distro, and Canonical's RCCL is broken (`rccl-single-process/README.md`).

## The overlay method (validated twice)

For evaluating or using an AMD-lineage ROCm library on this machine,
containers are unnecessary:

1. `curl` the deb from `repo.radeon.com/rocm/apt/<ver>/pool/main/...`
   (noble/24.04 builds; reachable from this box).
2. `dpkg -x` into `~/scratch/rocm-debs/amd-<ver>/` (no sudo).
3. Run with `LD_LIBRARY_PATH=<overlay>/opt/rocm-<ver>/lib`.
4. If the lib wants an older HIP soname (e.g. `libamdhip64.so.6`), symlink
   it to the system `.so.7` in a compat dir — **the HIP 7 runtime serves
   HIP 6 binaries correctly** (versioned symbols; AMD's promised direction
   is old-userland-on-newer-driver/runtime).

Validated uses:

| Library | Result |
|---|---|
| AMD RCCL 7.1 (2.27.7.70100, noble) | **Works fully** — 4/8-rank exact-match allreduce; replaces Canonical's broken build. See `rccl-single-process/README.md`. |
| AMD rocBLAS 6.2.4 (4.2.4, noble) | **Works** via the `.so.6`→`.so.7` symlink; benchmark below. |

Overlay debs live in `~/scratch/rocm-debs/` (not durable-critical —
re-downloadable; `/tmp` is tmpfs, home is not).

## Is ROCm 6.2.4 "the best for MI250X"? — measured: no, for our workload

Same binary (`alice_bench_lmhead_gpu` rebuilt against each), bf16
`gemm_ex` on gfx90a, lm_head shape (V=128256 × d=2048), same session:

| S | 6.2.4 (AMD) gemm ms | 7.1 (Canonical) gemm ms |
|---|---|---|
| 1 | 0.590 | 0.596 |
| 8 | 0.622 | 0.626 |
| 128 | 0.963 | 0.979 |

Dead tie (~2%, within run noise). gfx90a Tensile tuning was frozen long
before 6.2.4; the "6.2.4 is best" lore comes from the framework era
(PyTorch gfx90a wheels, MIOpen) and does not apply to direct rocBLAS
GEMM. **Lineage matters (AMD vs Canonical builds — the RCCL bug),
version does not (6.2.4 vs 7.1 — for this workload).**

## The docker question

- Docker works here (daemon 29.1.3, user in `docker` group), and a
  Ubuntu 24.04 + AMD ROCm container is the *standard* way to run an
  older ROCm userland: the KFD kernel driver is shared and
  backwards-compatible with older SDKs, passthrough is
  `--device=/dev/kfd --device=/dev/dri`.
- **But `docker.io` is unreachable from this machine** (registry
  connection times out). Would need a registry mirror (user's proxy
  setup) or `docker import` of a 24.04 rootfs tarball from a reachable
  source.
- Verdict: unnecessary for library evaluation (the overlay gives the
  same isolation with zero moving parts). Docker becomes interesting
  only if we ever want a fully AMD-consistent userland *runtime* for the
  engine (no Canonical mixing) — an operational choice, deferred.
