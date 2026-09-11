# Study: ROCm Tarball/Runfile Installation Feasibility on Ubuntu 26.04 (2026-09-11)

## Objective

Determine whether a stable, multi-rank-capable ROCm stack (specifically a matched
`librccl1` + `libamdhip64` build) can be installed into this project directory or
system via AMD's official tarball or runfile installer, bypassing the broken
Ubuntu-archive packages that cause the RCCL ABI segfault documented in
`rccl-single-process/`.

## System Context

- **Host:** Ubuntu 26.04.1 LTS (codename `resolute`)
- **Kernel:** 7.0.0-31-generic
- **GPU:** AMD Instinct MI250X (CDNA2, `gfx90a`)
- **Current ROCm:** Ubuntu-archive packages (`librccl1` 7.1.0-0ubuntu2,
  `libamdhip64-7` 7.1.0-0ubuntu2) — these exhibit an ABI mismatch that segfaults
  on any ≥2-rank RCCL collective.

## Methodology

1. Survey AMD official documentation for tarball/runfile installation methods.
2. Check ROCm compatibility matrices for Ubuntu 26.04 support across releases.
3. Inspect `repo.radeon.com` directory listings for available builds.
4. Review Canonical's public statements about ROCm packaging in Ubuntu 26.04.

## Key Findings

### 1. ROCm 6.x does NOT support Ubuntu 26.04 at all

The official AMD compatibility matrix (ROCm 6.4.1) lists supported operating
systems up to **Ubuntu 24.04.2/24.04.3** only. There is no Ubuntu 26.04 entry
for any ROCm 6.x release. Installing a 6.x runfile on this kernel would be
unsupported and likely fail at the KMD/UMD interface level.

### 2. AMD official repo has NO Ubuntu 26.04 builds for ROCm 7.1

Inspection of `https://repo.radeon.com/amdgpu-install/7.1/ubuntu/` shows only
two distribution directories:

- `jammy/`  → Ubuntu 22.04
- `noble/`  → Ubuntu 24.04

There is **no `resolute/` or 26.04 directory**. The ROCm 7.1 runfile installer
index (`repo.radeon.com/rocm/installer/rocm-runfile-installer/rocm-rel-7.1/`)
likewise contains only 22.04 and 24.04 subdirectories. Therefore:

- **Tarball:** Not available for ROCm 7.1 on Ubuntu 26.04.
- **Runfile:** Not available for ROCm 7.1 on Ubuntu 26.04.
- **amdgpu-install:** Not available for ROCm 7.1 on Ubuntu 26.04.

### 3. Ubuntu 26.04 is a "Canonical-native ROCm island"

Per the [Canonical blog post](https://canonical.com/blog/amd-rocm-on-ubuntu)
(2026-06-16):

> "Canonical and AMD teamed up last fall to package AMD ROCm AI/ML and HPC
> libraries right into the Ubuntu archive... We launched our first version of
> ROCm in Ubuntu 26.04 LTS!"

This means:
- Ubuntu 26.04 is the **first** Ubuntu release where ROCm ships natively in the
  archive (`apt install rocm`).
- The packages on this machine (`librccl1`, `libamdhip64-7`, etc.) are **built
  and packaged by Canonical**, not by AMD's unified build pipeline.
- AMD's own `repo.radeon.com` has not yet published 26.04-specific builds.

### 4. The RCCL segfault is likely a Canonical packaging defect

The crash analysis in `rccl-single-process/README.md` (launch-stub ABI mismatch:
`mov 0x20(%rdi),%rax` with `rdi = 0x43000`) points to a build-time mismatch
between the hipified RCCL and the HIP runtime. Because both libraries carry the
same version string (`7.1.0-0ubuntu2`) but were packaged by Canonical rather than
AMD, it is highly probable that the RCCL build used headers or hipify artifacts
that do not match the ABI of the shipped HIP runtime.

## Options Evaluated

| Option | Feasibility | Blocker |
|--------|-------------|---------|
| **Tarball/Runfile ROCm 6.4 → project dir** | ❌ Impossible | No Ubuntu 26.04 support in 6.x matrix |
| **Tarball/Runfile ROCm 7.1 → project dir** | ❌ Impossible | No `resolute` build on `repo.radeon.com` |
| **Downgrade system to ROCm 6.x** | ❌ Impossible | No 26.04 packages; kernel mismatch risk |
| **Wait for Canonical SRU** | ✅ Viable | Canonical is already working on ROCm 7.2 SRU |
| **Hand-rolled P2P allreduce** | ✅ Viable | Already validated in `p2p.cpp`; ~4% decode budget |
| **Ubuntu 24.04 container + AMD ROCm** | ⚠️ Complex | Docker overhead; needs full build env inside |
| **Build RCCL from source** | ⚠️ Fragile | Deep dependency chain; no guarantee of ABI match |

## Conclusion

**There is currently no supported way to install an AMD-official ROCm build
(tarball, runfile, or package repo) on this Ubuntu 26.04 machine.** The system
is locked to Canonical's archive packages until either:

1. Canonical publishes an SRU that fixes the RCCL build (ROCm 7.2+), or
2. AMD begins publishing `resolute` builds on `repo.radeon.com`.

For the `alice-infer` project, this means:

- **RCCL cannot be relied upon for TP in the near term.** The ABI bug is not
  configuration-related and cannot be bypassed with environment variables.
- **The hand-rolled P2P allreduce path (validated in `p2p.cpp`) remains the
  recommended first TP implementation.** It requires no external dependencies,
  no sudo, and has sufficient bandwidth margin for 70B decode at TP=4.
- Revisit RCCL only after a system-level ROCm update (`apt upgrade`) delivers
  a corrected build.

## References

- AMD ROCm 6.4.1 Compatibility Matrix:
  <https://rocm.docs.amd.com/en/docs-6.4.1/compatibility/compatibility-matrix.html>
- AMD ROCm 6.4.1 System Requirements:
  <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.1/reference/system-requirements.html>
- AMD Runfile Installer docs (7.1 era):
  <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.0/install/rocm-runfile-installer.html>
- Canonical Blog — "AMD ROCm on Ubuntu" (2026-06-16):
  <https://canonical.com/blog/amd-rocm-on-ubuntu>
- `repo.radeon.com` directory listing (verified 2026-09-11):
  `https://repo.radeon.com/amdgpu-install/7.1/ubuntu/` — contains `jammy/`, `noble/` only
- Prior study: `research/rccl-single-process/README.md` — RCCL ABI segfault analysis
