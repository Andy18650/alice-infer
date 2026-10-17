#!/usr/bin/env python3
"""Summarize all preheaders/splits, including unfavorable variants."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results/split')
metadata = {tuple(row[k] for k in ('schedule', 'prelude', 'unroll')): row for row in json.loads((folder / 'audit.json').read_text())['kernels']}
assert len(metadata) == 168
medians = {}; fits = {}
fields = ('cycles_per_mfma', 'TFLOPs', 'core_MHz')
for run, rounds in [('run1', 9), ('run2', 9), ('longer', 3)]:
    rows = list(csv.DictReader((folder / f'{run}.csv').open()))
    assert len(rows) == 168 * rounds
    grouped = defaultdict(list); seen = set()
    for row in rows:
        key = tuple(int(row[k]) for k in ('schedule', 'prelude', 'unroll'))
        ident = (*key, int(row['round']))
        assert ident not in seen; seen.add(ident)
        assert int(row['valid']) == 1 and float(row['max_abs_error']) == 0, (run, key)
        grouped[key].append(row)
    assert set(grouped) == set(metadata)
    for key, samples in grouped.items():
        assert len(samples) == rounds
        medians[run, *key] = {field: median(float(row[field]) for row in samples) for field in fields}
    for schedule, prelude in sorted({key[:2] for key in grouped}):
        c4, c8, c32 = [medians[run, schedule, prelude, u]['cycles_per_mfma'] for u in (4, 8, 32)]
        b = (c4 - c32) / (1 / 4 - 1 / 32)
        a = c32 - b / 32
        fits[run, schedule, prelude] = (a, b, c8 - (a + b / 8))
with (folder / 'summary.csv').open('w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['run', 'schedule', 'prelude', 'unroll', *fields])
    for key, values in sorted(medians.items()): writer.writerow([*key, *(values[field] for field in fields)])
with (folder / 'fits.csv').open('w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['run', 'schedule', 'prelude', 'intercept', 'boundary_cycles_per_loop', 'held_out_u8_residual'])
    for key, values in sorted(fits.items()): writer.writerow([*key, *values])
lines = ['Primary two-bank results, U=8: C8 and fitted B cycles/backedge', 'run | prelude | schedule | C8 | B | TFLOPs | core_MHz']
for run in ('run1', 'run2', 'longer'):
    for prelude in (0, 1, 4):
        for schedule in (0, 1, 2, 3):
            v = medians[run, schedule, prelude, 8]
            lines.append(f'{run} | {prelude} | {schedule} | {v[fields[0]]:.6f} | {fits[run,schedule,prelude][1]:.6f} | {v[fields[1]]:.3f} | {v[fields[2]]:.1f}')
lines += ['', 'All baseline preheaders, process 1 fitted B:', 'prelude | raw | two-bank 8/8 | four-bank-lagged 8/8']
for prelude in range(8):
    lines.append(str(prelude) + ' | ' + ' | '.join(f'{fits["run1",s,prelude][1]:.6f}' for s in (0, 1, 6)))
lines += ['', 'All split variants, process 1 fitted B:', 'schedule | prelude0 | prelude1 | prelude4 | prelude7']
for schedule in range(1, 11):
    lines.append(str(schedule) + ' | ' + ' | '.join(f'{fits["run1",schedule,p][1]:.6f}' for p in (0, 1, 4, 7)))
lines += ['', 'Net event-throughput ratio, two-bank 10/6 vs matching raw FP16, prelude1:']
for run in ('run1', 'run2'):
    ratio = medians[run, 3, 1, 8]['TFLOPs'] / medians[run, 0, 1, 8]['TFLOPs']
    lines.append(f'{run}: {100*ratio:.6f}%')
lines += ['', f'PASS 3528 exact-valid samples; max held-out residual {max(abs(x[2]) for x in fits.values()):.6f} cycles/MFMA.']
text = '\n'.join(lines) + '\n'
(folder / 'summary.txt').write_text(text)
print(text)
