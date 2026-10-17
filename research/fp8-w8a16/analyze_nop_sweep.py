#!/usr/bin/env python3
"""Summarize two retained runs and draw the measured overlap-window curves."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results/nop-sweep')
values = defaultdict(list)
errors = []
for run in (1, 2):
    rows = list(csv.DictReader((folder / f'run{run}.csv').open()))
    assert len(rows) == 135 * 9
    seen = set()
    for row in rows:
        delay, unroll, round_ = (int(row[k]) for k in ('delay', 'unroll', 'round'))
        assert (delay, unroll, round_) not in seen
        seen.add((delay, unroll, round_))
        values[run, delay, unroll].append(float(row['cycles_per_mfma']))
        errors.append(float(row['max_abs_error']))
assert max(errors) == 0
med = {key: median(v) for key, v in values.items()}
delays = sorted({d for _, d, _ in med})
with (folder / 'summary.csv').open('w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['delay_wait_units', 'nominal_added_core_cycles', 'unroll', 'run1_median_cycles_per_mfma', 'run2_median_cycles_per_mfma'])
    for delay in delays:
        for unroll in (2, 8, 32):
            writer.writerow([delay, max(delay, 0) * 4, unroll, med[1, delay, unroll], med[2, delay, unroll]])
lines = ['Median core cycles/MFMA (run 1 / run 2):', 'wait-units | added-core | U=2 | U=8 | U=32']
for delay in [-1, 0, 1, 2, 4, 6, 7, 8, 10, 12, 16, 24, 32, 64, 96]:
    cells = [f'{med[1, delay, u]:.3f}/{med[2, delay, u]:.3f}' for u in (2, 8, 32)]
    lines.append(f'{delay:>10} | {max(delay, 0)*4:>10} | ' + ' | '.join(cells))
lines += ['', 'Slope in core cycles/MFMA per added nominal core cycle:', 'U | run | initial (wait 0..6) | exposed (wait 8..16)']
for unroll in (2, 8, 32):
    for run in (1, 2):
        initial = (med[run, 6, unroll] - med[run, 0, unroll]) / 24
        exposed = (med[run, 16, unroll] - med[run, 8, unroll]) / 32
        lines.append(f'{unroll} | {run} | {initial:.6f} | {exposed:.6f}')
lines += ['', 'Independent NOP calibration: core cycles per added encoded wait unit']
for filename in ('nop-clock.csv', 'nop-clock-repeat.csv'):
    calibration = {(row['geometry'], int(row['wait_units'])):
                   float(row['median_core_cycles_per_delay'])
                   for row in csv.DictReader((folder / filename).open())}
    assert len(calibration) == 24
    for geometry in ('1x64', '104x256'):
        slope = (calibration[geometry, 64] - calibration[geometry, 1]) / 63
        assert 3.99 < slope < 4.01
        lines.append(f'{filename} {geometry}: {slope:.6f}')
lines += ['', f'PASS: all 2430 timed configurations, maximum observed absolute output error {max(errors)}']
text = '\n'.join(lines) + '\n'
(folder / 'summary.txt').write_text(text)
print(text)
# Focus the plot on the knee; long-delay points remain in summary.csv.
w, h = 840, 480
x0, y0, pw, ph = 75, 65, 700, 345
def x(v): return x0 + v / 64 * pw
def y(v): return y0 + ph - (v - 60) / 65 * ph
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>', '<g font-family="sans-serif" font-size="13">', '<text x="75" y="25" font-size="18">FP8 conversion + scalar NOP delay: measured overlap knee</text>', '<text x="75" y="46">HIP 0 gfx90a; CPU+memory NUMA 0; no model/context; two runs, nine rounds each</text>']
for val in range(0, 65, 8):
    svg += [f'<path d="M{x(val)},{y0} V{y0+ph}" stroke="#ddd"/>', f'<text x="{x(val)}" y="{y0+ph+20}" text-anchor="middle">{val}</text>']
for val in range(60, 126, 10):
    svg += [f'<path d="M{x0},{y(val)} H{x0+pw}" stroke="#ddd"/>', f'<text x="{x0-9}" y="{y(val)+4}" text-anchor="end">{val}</text>']
for u, color in [(2, '#bf3b25'), (8, '#2878b5'), (32, '#228833')]:
    for run in (1, 2):
        ds = [d for d in delays if 0 <= d <= 16]
        pts = ' '.join(f'{x(4*d):.3f},{y(med[run,d,u]):.3f}' for d in ds)
        svg += [f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" opacity="{1 if run==1 else .45}"/>']
    svg += [f'<text x="{x(49)}" y="{y(med[1,12,u])-8}" fill="{color}">U={u}</text>']
svg += [f'<path d="M{x(28)},{y0} V{y0+ph}" stroke="#555" stroke-dasharray="4 4"/>', '<text x="425" y="462" text-anchor="middle">Added nominal core cycles per conversion (4 × encoded NOP wait units)</text>', '<text transform="translate(20,245) rotate(-90)" text-anchor="middle">Mean per-wave core cycles / MFMA</text>', '</g></svg>']
(folder / 'knee.svg').write_text('\n'.join(svg) + '\n')
