#!/usr/bin/env python3
"""Summarize measured rates and a representative CTA-body overlap proof.

Use --report-dir to audit a historical dataset without overwriting its derived
reports. This script never modifies raw CSV, clocks, source or build snapshots.
"""

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path


HERE = Path(__file__).resolve().parent


def summarize_rates(rows):
    timing_rows = [row for row in rows if row['kind'] == 'timing']
    problems = sorted({(int(row['m']), int(row['n']), int(row['k'])) for row in timing_rows})
    lines = [
        '# Native GEMM matched results',
        '',
        'Native HIP O3 gfx90a HIP0/BDF31:00.0, CPU+memory NUMA0, no model/context/clock overrides. CPU packing/H2D/reference excluded; full kernel HIP event used for TFLOP/s. Two counterbalanced process repeats, nine samples each.',
        '',
        '| M x N x K | Tile | Useful TFLOP/s run1/run2 | Effective core MHz | Global clock-adjusted efficiency |',
        '|---|---:|---:|---:|---:|',
    ]
    for m, n, k in problems:
        for tile in (128, 256):
            matches = sorted(
                (row for row in timing_rows
                 if (int(row['m']), int(row['n']), int(row['k']), int(row['tile']))
                 == (m, n, k, tile)),
                key=lambda row: int(row['repeat']),
            )
            if not matches:
                continue
            columns = []
            for metric, scale in (('useful_TFLOPs', 1), ('median_core_MHz', 1),
                                  ('global_clock_adjusted_efficiency', 100)):
                columns.append('/'.join(f'{float(row[metric]) * scale:.2f}'
                                        for row in matches))
            lines.append(f'|{m}x{n}x{k}|{tile}|' + ' | '.join(columns) + '%|')
    return lines


def overlap_proof(clock_file, tile):
    raw = clock_file.read_bytes()
    timestamps = struct.unpack('<' + 'Q' * (len(raw) // 8), raw)
    events = []
    for block in range(len(timestamps) // 16):
        values = timestamps[block * 16:(block + 1) * 16]
        begin = min(values[wave * 4 + 2] for wave in range(4))
        end = max(values[wave * 4 + 3] for wave in range(4))
        events.extend(((begin, 1), (end, -1)))

    # Half-open intervals: an end is processed before a start at the same tick.
    active = 0
    peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return dict(
        tile=tile,
        file=str(clock_file),
        sha256=hashlib.sha256(raw).hexdigest(),
        peak_overlapping_cta_bodies=peak,
        ctas=len(timestamps) // 16,
    )


def analyze(results, report_directory):
    with (results / 'summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    lines = summarize_rates(rows)

    # Retain the original representative launch selection, not a search for
    # whichever launch happens to give the largest overlap number.
    proofs = []
    for tile in (128, 256):
        filename = f't{tile}-m4096-n6656-k32768-r1-a1729-b2718-r4.bin'
        proofs.append(overlap_proof(results / 'clocks' / filename, tile))
    overlap_text = ', '.join(
        f"tile{proof['tile']}={proof['peak_overlapping_cta_bodies']}" for proof in proofs
    )
    lines.extend([
        '',
        'Observed maximum simultaneous CTA-body intervals in the representative '
        f'long-K launch: {overlap_text}. This is body-overlap evidence, not exact '
        'allocation timestamps or a CTA-to-CU map.',
        '',
        f"Checked launches: {sum(int(row['samples']) for row in rows)}. Large cases "
        'use sampled exact references plus all-C finite/poison and guard scans; '
        'do not relabel them full references.',
    ])
    report_directory.mkdir(parents=True, exist_ok=True)
    (report_directory / 'overlap.json').write_text(json.dumps(proofs, indent=2) + '\n')
    (report_directory / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='?', default=HERE / 'results/fp16-bridge-full')
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    report_directory = args.report_dir.resolve() if args.report_dir else results
    analyze(results, report_directory)


if __name__ == '__main__':
    main()
