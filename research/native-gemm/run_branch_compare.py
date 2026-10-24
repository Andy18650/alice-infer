#!/usr/bin/env python3
"""Control-flow-only A/B: retain104/208/312 CTA coverage and a large-grid check.

Uses the existing Case/run_case harness, references, clocks and eager timing
protocol. Before and after have separate immutable compilation-source/build
snapshots. The root driver directory records the actual shared runner used.
No MFMA/memory scheduling change is part of this experiment.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from run import Case, FUNCTIONAL_CASES, capture_smi, require, run_case, write_summary
from verify_artifacts import verify_manifest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBLEMS = (
    (1664, 1024, 8192), (1664, 1024, 32768),       #104 CTAs
    (3328, 1024, 8192), (3328, 1024, 32768),       #208 CTAs
    (4992, 1024, 8192), (4992, 1024, 32768),       #312 CTAs, retained
    (4096, 6656, 32768),                         #large-grid regression check
)
BOUNDARY_K = (95, 96, 97, 127, 128, 129, 191, 192, 193, 255, 256, 257)
SOURCE_FILES = ('probe.hip', 'data.hpp', 'generate.py', 'build.sh', 'inspect_code.py')
BUILD_FILES = ('compute.hpp', 'compiler-flags.txt', 'hashes.txt',
               'inventory.json', 'loop.txt', 'metadata.txt')


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def executable_hash(build):
    for line in (build / 'hashes.txt').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if Path(name).name == 'probe':
            require(fingerprint(build / 'probe') == expected, 'Executable fingerprint mismatch')
            return expected
    raise RuntimeError(f'No executable fingerprint in {build}')


def snapshot_variant(destination, build, source, extra_sources=()):
    for folder in ('source', 'builds', 'verification', 'timing', 'clocks'):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    for name in (*SOURCE_FILES, *extra_sources):
        shutil.copy2(source / name, destination / 'source' / name)
    build_snapshot = destination / 'builds' / build.name
    build_snapshot.mkdir()
    for name in BUILD_FILES:
        shutil.copy2(build / name, build_snapshot / name)


def functional_cases(tile):
    for m, n, k in FUNCTIONAL_CASES:
        yield Case(tile, m, n, k)
    for k in BOUNDARY_K:
        yield Case(tile, 129, 131, k)
    yield Case(tile, 129, 131, 65, seed_a=1730)
    yield Case(tile, 129, 131, 65, seed_b=2719)


def write_comparison(output, records, title='Branch-only A/B'):
    timing = [row for row in records if row['kind'] == 'timing']
    lines = [
        '# ' + title, '',
        'Native HIP O3 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0, no model/context/'
        'clock override. Full kernel event TFLOP/s; packing/reference excluded. '
        'Two counterbalanced nine-sample process medians.', '',
        '|M x N x K|CTAs|Before TFLOP/s|After TFLOP/s|Before global efficiency|After global efficiency|',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for m, n, k in PROBLEMS:
        columns = []
        for metric in ('useful_TFLOPs', 'global_clock_adjusted_efficiency'):
            for variant in ('before', 'after'):
                matches = sorted(
                    (row for row in timing
                     if row['variant'] == variant
                     and (row['m'], row['n'], row['k']) == (m, n, k)),
                    key=lambda row: row['repeat'],
                )
                scale = 100 if metric == 'global_clock_adjusted_efficiency' else 1
                columns.append('/'.join(f'{row[metric] * scale:.3f}' for row in matches))
        lines.append(f'|{m}x{n}x{k}|{(m // 128) * (n // 128)}|'
                     + '|'.join(columns) + '|')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')


def main(defaults=None):
    # The small MFMA-first entry point reuses this exact protocol, validators
    # and resource checks, overriding paths/title rather than copying a runner.
    parser = argparse.ArgumentParser(description=(defaults or {}).get('description', __doc__))
    parser.add_argument('--before-build', type=Path,
                        default=ROOT / 'temp/native-gemm/readable/s32-t128')
    parser.add_argument('--before-source', type=Path,
                        default=HERE / 'results/readable/source')
    parser.add_argument('--after-build', type=Path,
                        default=ROOT / 'temp/native-gemm/branch-clean/s32-t128')
    parser.add_argument('--after256', type=Path,
                        default=ROOT / 'temp/native-gemm/branch-clean/s32-t256')
    parser.add_argument('--after-source', type=Path,
                        default=HERE / 'results/branch-clean/after/source')
    parser.add_argument('--output', type=Path, default=HERE / 'results/branch-clean')
    parser.add_argument('--allow-resource-change', action='store_true')
    parser.set_defaults(title='Branch-only A/B', operation_change='control flow only',
                        entry_point='run_branch_compare.py', extra_sources=(), before_extra_sources=(),
                        functional_variants=(('after', 128), ('after256', 256)))
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('choose a fresh results directory')

    builds = {'before': args.before_build.resolve(), 'after': args.after_build.resolve()}
    sources = {'before': args.before_source.resolve(), 'after': args.after_source.resolve()}
    if args.after256 is not None:
        builds['after256'] = args.after256.resolve()
        sources['after256'] = args.after_source.resolve()
    expected_hashes = {}
    for variant, build in builds.items():
        verify_manifest(build / 'hashes.txt', sources[variant])
        expected_hashes[variant] = executable_hash(build)

    before_resources = json.loads((builds['before'] / 'inventory.json').read_text())
    after_resources = json.loads((builds['after'] / 'inventory.json').read_text())
    for field in ('vgpr_count', 'agpr_count', 'sgpr_count', 'lds_bytes',
                  'private_segment_fixed_size'):
        require(args.allow_resource_change or before_resources[field] == after_resources[field],
                f'Resource change would confound this test: {field}')

    for variant, build in builds.items():
        extra = args.extra_sources if variant != 'before' else args.before_extra_sources
        snapshot_variant(output / variant, build, sources[variant], extra)
    (output / 'driver').mkdir()
    for name in dict.fromkeys(('run_branch_compare.py', 'run.py', 'verify_artifacts.py',
                               args.entry_point)):
        shutil.copy2(HERE / name, output / 'driver' / name)
    (output / 'invocation.json').write_text(json.dumps(dict(
        builds={name: str(path) for name, path in builds.items()}, problems=PROBLEMS,
        tile=128, stage_k=32, placement='numactl CPU+memory node0',
        operation_change=args.operation_change, interleave_prefetch=False,
        title=args.title, allow_resource_change=args.allow_resource_change,
        before_resources=before_resources, after_resources=after_resources,
        functional_variants=args.functional_variants,
    ), indent=2) + '\n')
    capture_smi(output / 'amd-smi-before.txt')
    records = []
    launches = []

    with (output / 'commands.txt').open('w') as commands:
        def execute(variant, case):
            build = builds[variant]
            actual_hash = fingerprint(build / 'probe')
            require(actual_hash == expected_hashes[variant], 'Binary changed before launch')
            launches.append(dict(variant=variant, case=case.stem, sha256=actual_hash))
            commands.write(f'# {variant}\n')
            summary = run_case(case, {case.tile: build}, output / variant, commands)
            records.append(dict(variant=variant, **summary))
            write_summary(output / 'summary.csv', records)
            (output / 'launch-hashes.json').write_text(json.dumps(launches, indent=2) + '\n')
            if case.timing:
                write_comparison(output, records, args.title)

        for variant, tile in args.functional_variants:
            for case in functional_cases(tile):
                execute(variant, case)
        for repeat in (1, 2):
            problems = PROBLEMS if repeat == 1 else reversed(PROBLEMS)
            order = ('before', 'after') if repeat == 1 else ('after', 'before')
            for m, n, k in problems:
                for variant in order:
                    execute(variant, Case(128, m, n, k, repeat=repeat))

    capture_smi(output / 'amd-smi-after.txt')
    print(f"COMPLETE {sum(row['samples'] for row in records)} checked launches; "
          f'{args.title} comparison and bounded-tail correctness pass.')


if __name__ == '__main__':
    main()
