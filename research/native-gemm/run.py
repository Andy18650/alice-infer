#!/usr/bin/env python3
"""Focused correctness suite and matched full-problem tile comparison.

No tuning takes place here. The problem list retains the original comparisons
and adds 208/312-CTA coverage; seeds and timing protocol remain unchanged.
Every timed launch also returns raw per-wave clocks for independent checks.
"""

import argparse
import csv
import json
import shlex
import shutil
import statistics
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBLEMS = (
    (1664, 1024, 8192),
    (1664, 1024, 32768),
    (3328, 1024, 8192),  # 208 CTAs with the selected tile.
    (3328, 1024, 32768),
    (4992, 1024, 8192),  # Keep 312 CTAs as requested.
    (4992, 1024, 32768),
    (3328, 2048, 8192),
    (4096, 6656, 8192),
    (4096, 6656, 32768),
)
FUNCTIONAL_CASES = (
    (1, 1, 0), (1, 1, 1), (17, 23, 7), (31, 35, 8),
    (64, 64, 31), (127, 129, 32), (128, 128, 33),
    (129, 131, 63), (129, 131, 64), (129, 131, 65),
    (255, 257, 127), (257, 259, 129), (128, 128, 512),
    (257, 129, 1024), (17, 19, 65535), (17, 19, 65536),
)
SUMMARY_METRICS = (
    'useful_TFLOPs', 'executed_TFLOPs', 'event_us', 'grid_body_us',
    'median_core_MHz', 'median_cta_share_of_cu_peak',
    'global_clock_adjusted_efficiency', 'pack_ms',
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@dataclass(frozen=True)
class Case:
    tile: int
    m: int
    n: int
    k: int
    repeat: int = 0
    seed_a: int = 1729
    seed_b: int = 2718

    @property
    def timing(self):
        return self.repeat > 0

    @property
    def folder(self):
        return 'timing' if self.timing else 'verification'

    @property
    def stem(self):
        return (
            f't{self.tile}-m{self.m}-n{self.n}-k{self.k}-r{self.repeat}'
            f'-a{self.seed_a}-b{self.seed_b}'
        )


def capture_smi(destination):
    with destination.open('w') as stream:
        subprocess.run(['amd-smi'], stdout=stream, stderr=subprocess.STDOUT, check=False)


def copy_provenance(output, builds):
    for folder in ('source', 'builds', 'verification', 'timing', 'clocks'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    for name in ('probe.hip', 'data.hpp', 'generate.py', 'build.sh', 'inspect_code.py', 'run.py'):
        shutil.copy2(HERE / name, output / 'source' / name)
    for tile, build in builds.items():
        destination = output / 'builds' / str(tile)
        destination.mkdir()
        for name in ('compute.hpp', 'compiler-flags.txt', 'hashes.txt',
                     'inventory.json', 'loop.txt', 'metadata.txt'):
            shutil.copy2(build / name, destination / name)


def check_clock_dump(path, sample, tile):
    raw = path.read_bytes()
    timestamps = struct.unpack('<' + 'Q' * (len(raw) // 8), raw)
    blocks = int(sample['ctas'])
    require(len(timestamps) == blocks * 16, f'Unexpected timestamp count: {path}')

    # File order: [CTA][wave 0..3][core_begin,core_end,steady_begin,steady_end].
    grid_begin = min(timestamps[i * 16 + w * 4 + 2]
                     for i in range(blocks) for w in range(4))
    grid_end = max(timestamps[i * 16 + w * 4 + 3]
                   for i in range(blocks) for w in range(4))
    tick_us = float(sample['grid_body_us']) / (grid_end - grid_begin)
    core_rates = []
    cta_shares = []
    for block in range(blocks):
        values = timestamps[block * 16:(block + 1) * 16]
        cycles = (max(values[w * 4 + 1] for w in range(4))
                  - min(values[w * 4] for w in range(4)))
        steady_ticks = (max(values[w * 4 + 3] for w in range(4))
                        - min(values[w * 4 + 2] for w in range(4)))
        core_rates.append(cycles / (steady_ticks * tick_us))
        cta_shares.append((2 * tile * tile * int(sample['kp'])) / (cycles * 1024))

    median_rate = statistics.median(core_rates)
    require(abs(median_rate - float(sample['median_core_MHz'])) < 0.002,
            f'Core/steady-rate calculation mismatch: {path}')
    require(abs(statistics.median(cta_shares)
                - float(sample['median_cta_share_of_cu_peak'])) < 1e-8,
            f'Per-CTA share calculation mismatch: {path}')

    # Different from per-CTA share: normalize total useful event throughput
    # against all 104 CUs at the measured core/steady rate.
    efficiency = float(sample['useful_TFLOPs']) / (104 * 1024 * median_rate / 1e6)
    require(abs(efficiency - float(sample['global_clock_adjusted_efficiency'])) < 1e-6,
            f'Whole-GCD efficiency calculation mismatch: {path}')


def validate_samples(case, samples, output):
    require(len(samples) == (9 if case.timing else 1), f'Sample count: {case.stem}')
    for sample in samples:
        require(sample['kind'] == 'gemm', 'Resident diagnostic used as ordinary GEMM')
        require(int(sample['tile']) == case.tile
                and sample['stage_k'] == '32' and sample['swizzle'] == '0',
                f'Wrong kernel configuration: {case.stem}')
        require(sample['valid'] == '1'
                and sample['mismatches'] == sample['nonfinite'] == sample['guards'] == '0',
                f'Numerical/coverage/guard check failed: {case.stem}')
        expected_blocks = ((case.m + case.tile - 1) // case.tile
                           * ((case.n + case.tile - 1) // case.tile))
        require(int(sample['lds_bytes']) == case.tile * 128
                and int(sample['ctas']) == expected_blocks,
                f'Wrong resource/grid accounting: {case.stem}')
        if case.timing:
            path = output / 'clocks' / (case.stem + f"-r{sample['round']}.bin")
            check_clock_dump(path, sample, case.tile)


def write_summary(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run_case(case, builds, output, commands):
    command = [
        'numactl', '--cpunodebind=0', '--membind=0', str(builds[case.tile] / 'probe'),
        '--m', str(case.m), '--n', str(case.n), '--k', str(case.k),
        '--rounds', '9' if case.timing else '1',
        '--warmup-ms', '200' if case.timing else '0',
        '--swizzle', '0', '--seed-a', str(case.seed_a), '--seed-b', str(case.seed_b),
    ]
    if case.timing:
        command += ['--stamps-prefix', str(output / 'clocks' / case.stem)]
    commands.write(shlex.join(command) + '\n')
    commands.flush()

    csv_path = output / case.folder / (case.stem + '.csv')
    log_path = output / case.folder / (case.stem + '.log')
    with csv_path.open('w') as stdout, log_path.open('w') as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
    with csv_path.open() as stream:
        samples = list(csv.DictReader(stream))
    require('COMPLETE all reference outputs/finite C/guards exact' in log_path.read_text(),
            f'Incomplete probe: {case.stem}')
    validate_samples(case, samples, output)

    first = samples[0]
    row = dict(
        kind=case.folder, tile=case.tile, m=case.m, n=case.n, k=case.k,
        repeat=case.repeat, samples=len(samples),
        full_reference=first['full_reference'], reference_outputs=first['reference_outputs'],
        ctas=first['ctas'], resident_ctas_per_cu=first['resident_ctas_per_cu'],
    )
    for metric in SUMMARY_METRICS:
        row[metric] = statistics.median(float(sample[metric]) for sample in samples)
    if case.timing:
        print(
            f"{case.stem}: {row['useful_TFLOPs']:.3f} TFLOP/s, "
            f"clock={row['median_core_MHz']:.1f}MHz, "
            f"global efficiency={row['global_clock_adjusted_efficiency']:.3f}",
            flush=True,
        )
    return row


def case_sequence(builds):
    for tile in builds:
        for m, n, k in FUNCTIONAL_CASES:
            yield Case(tile, m, n, k)
        yield Case(tile, 129, 131, 65, seed_a=1730)
        yield Case(tile, 129, 131, 65, seed_b=2719)

    jobs = [(tile, m, n, k) for m, n, k in PROBLEMS for tile in builds]
    for repeat in (1, 2):
        order = jobs if repeat == 1 else reversed(jobs)
        for tile, m, n, k in order:
            yield Case(tile, m, n, k, repeat=repeat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/fp16-bridge-full')
    parser.add_argument('--build128', type=Path,
                        default=ROOT / 'temp/native-gemm/fp16-bridge/s32-t128')
    parser.add_argument('--build256', type=Path,
                        default=ROOT / 'temp/native-gemm/fp16-bridge/s32-t256')
    args = parser.parse_args()
    output = args.output.resolve()
    builds = {128: args.build128.resolve(), 256: args.build256.resolve()}
    if output.exists() and any(output.iterdir()):
        parser.error('choose a fresh results directory')
    for build in builds.values():
        subprocess.run(['sha256sum', '-c', str(build / 'hashes.txt')],
                       check=True, stdout=subprocess.DEVNULL)

    copy_provenance(output, builds)
    invocation = dict(
        builds={str(tile): str(path) for tile, path in builds.items()},
        problems=PROBLEMS, stage_k=32, swizzle=0,
        placement='numactl CPU+memory node0', libraries='none',
        packing='CPU outside GEMM timing',
    )
    (output / 'invocation.json').write_text(json.dumps(invocation, indent=2) + '\n')
    capture_smi(output / 'amd-smi-before.txt')
    rows = []
    with (output / 'commands.txt').open('w') as commands:
        for case in case_sequence(builds):
            rows.append(run_case(case, builds, output, commands))
            write_summary(output / 'summary.csv', rows)
    capture_smi(output / 'amd-smi-after.txt')
    print(
        f"COMPLETE {sum(row['samples'] for row in rows)} verified launches; "
        'small full references and large sampled references/finite all-C/guards pass.'
    )


if __name__ == '__main__':
    main()
