#!/usr/bin/env python3
"""Report schedule-phase A/B and separate endpoint intercept from backedge cost."""
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results/order')
medians = {}
for run in ('run1', 'run2', 'longer'):
    rows = list(csv.DictReader((folder / f'{run}.csv').open()))
    assert len(rows) == 81
    seen = set()
    grouped = defaultdict(list)
    for row in rows:
        mode, unroll, round_ = (int(row[k]) for k in ('mode', 'unroll', 'round'))
        assert (mode, unroll, round_) not in seen
        seen.add((mode, unroll, round_))
        assert float(row['max_abs_error']) == 0
        grouped[mode, unroll].append(float(row['cycles_per_mfma']))
    assert len(grouped) == 9 and all(len(v) == 9 for v in grouped.values())
    for (mode, unroll), values in grouped.items():
        medians[run, mode, unroll] = median(values)
lines = ['Median core cycles/MFMA:', 'run | U | pure FP16 | convert-before | convert-after']
with (folder / 'summary.csv').open('w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['run', 'mode', 'unroll', 'median_cycles_per_mfma'])
    for run in ('run1', 'run2', 'longer'):
        for u in (2, 8, 32):
            values = [medians[run, mode, u] for mode in (0, 1, 2)]
            lines.append(f'{run} | {u} | ' + ' | '.join(f'{v:.6f}' for v in values))
            for mode in (0, 1, 2):
                writer.writerow([run, mode, u, medians[run, mode, u]])
lines += ['', 'C(U) = A + B/U; fit U=2 and U=32; U=8 held out:', 'run | mode | A (cycles/MFMA) | B (cycles/backedge) | U=8 residual']
fits = []
for run in ('run1', 'run2', 'longer'):
    for mode in (1, 2):
        c2, c32 = (medians[run, mode, u] for u in (2, 32))
        b = (c2 - c32) / (1 / 2 - 1 / 32)
        a = c32 - b / 32
        residual = medians[run, mode, 8] - (a + b / 8)
        lines.append(f'{run} | {mode} | {a:.6f} | {b:.6f} | {residual:.6f}')
        fits.append([run, mode, a, b, residual])
with (folder / 'fits.csv').open('w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['run', 'mode', 'intercept_cycles_per_mfma', 'boundary_cycles_per_loop', 'held_out_u8_residual'])
    writer.writerows(fits)
lines += ['', 'PASS: 243 samples, all output checks report zero absolute error.']
text = '\n'.join(lines) + '\n'
(folder / 'summary.txt').write_text(text)
print(text)
