#!/usr/bin/env python3
"""Verify historical source snapshots and preserved native executables.

Recorded manifests refer to the live source paths used when they were built.
Those paths now contain newer code, so source/header entries must be checked
against the immutable snapshot beside each result, not today's source files.
Executable/code-object entries still refer to the original preserved binaries
in temp/native-gemm. Missing binaries are reported, never silently rebuilt.
"""

import hashlib
from pathlib import Path


RESULTS = Path(__file__).resolve().parent / 'results'
HISTORICAL_PHASES = ('initial', 'swizzle', 'late', 'early', 'stage64', 'resident', 'final')
SOURCE_FILES = {'probe.hip', 'data.hpp', 'generate.py'}


def verify_manifest(manifest, source_directory):
    for line in manifest.read_text().splitlines():
        expected_hash, recorded_path = line.split(maxsplit=1)
        path = Path(recorded_path)
        if path.name in SOURCE_FILES:
            path = source_directory / path.name
        elif path.name == 'compute.hpp':
            path = manifest.parent / path.name
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f'Fingerprint mismatch: {manifest}: {path}')


def verify_phase(phase):
    directory = RESULTS / phase
    if phase in ('final', 'readable'):
        manifests = list((directory / 'builds').glob('*/hashes.txt'))
    else:
        manifests = list(directory.glob('t*/hashes.txt'))
    if not manifests:
        raise RuntimeError(f'No build manifests for phase {phase}')
    for manifest in manifests:
        verify_manifest(manifest, directory / 'source')
    print(phase, 'source/header/original executable/code-object hashes pass')


def verify_bank_control():
    directory = RESULTS / 'bank'
    for line in (directory / 'hashes.txt').read_text().splitlines():
        expected_hash, recorded_path = line.split(maxsplit=1)
        path = Path(recorded_path)
        if path.name == 'bank-probe.hip':
            path = directory / path.name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f'Bank-control fingerprint mismatch: {path}')
    print('bank-control archived source/original native hashes pass')


def verify_dependencies(directory):
    for listing in directory.glob('ldd*.txt'):
        text = listing.read_text().lower()
        if any(name in text for name in ('rocblas', 'hipblas', 'libck')):
            raise RuntimeError(f'Unexpected vendor GEMM dependency: {listing}')


def verify_branch_comparison():
    directory = RESULTS / 'branch-clean'
    if not (directory / 'summary.csv').exists():
        return
    for variant in ('before', 'after', 'after256'):
        root = directory / variant
        manifests = list((root / 'builds').glob('*/hashes.txt'))
        if not manifests:
            raise RuntimeError(f'No branch-comparison manifest: {variant}')
        for manifest in manifests:
            verify_manifest(manifest, root / 'source')
        verify_dependencies(root)
    for manifest in (RESULTS / 'branch-flow-audit' / 'gpu64').glob('t*/hashes.txt'):
        verify_manifest(manifest, directory / 'after' / 'source')
    print('branch-only before/after and StageK64 source/native hashes pass')


def main():
    for phase in HISTORICAL_PHASES:
        verify_phase(phase)
    verify_bank_control()
    verify_dependencies(RESULTS / 'final')
    if (RESULTS / 'readable' / 'summary.csv').exists():
        verify_phase('readable')
        verify_dependencies(RESULTS / 'readable')
    verify_branch_comparison()
    print('Recorded dynamic dependencies contain no vendor GEMM libraries')


if __name__ == '__main__':
    main()
