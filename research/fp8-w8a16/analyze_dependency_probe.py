#!/usr/bin/env python3
"""Summarize ring depth and staged VALU controls; exclude incorrect results."""
import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results/dependency')
labels = {0: 'raw', 1: 'barriers', 2: 'v_nop', 3: 'scratch_xor', 4: 'read_B',
          5: 'identity_write_B', 6: 'read_A', 7: 'identity_write_A',
          8: 'paired_flip_unsafe', 9: 'convert', 10: 'paired_flip_delayed'}
audit = json.loads((folder / 'audit.json').read_text())
metadata = {tuple(row[k] for k in ('kind', 'order', 'banks', 'ops', 'unroll')): row for row in audit['kernels']}
assert len(metadata) == 309
medians = {}; validity = {}; fits = {}; invalid = []
for run, rounds in [('run1', 9), ('run2', 9), ('longer', 3)]:
    rows = list(csv.DictReader((folder / f'{run}.csv').open()))
    assert len(rows) == 309 * rounds
    grouped = defaultdict(list); seen = set()
    for row in rows:
        key = tuple(int(row[k]) for k in ('kind', 'order', 'banks', 'ops', 'unroll'))
        identity = (*key, int(row['round']))
        assert identity not in seen; seen.add(identity)
        assert bool(int(row['valid'])) == (float(row['max_abs_error']) == 0)
        assert bool(int(row['valid'])) == (int(row['bad_outputs']) == 0)
        if not int(row['valid']):
            invalid.append([run, *key, int(row['round']), float(row['max_abs_error']), int(row['bad_outputs'])])
            # This exact-value test detects genuine data-changing source hazards.
            assert key[0] == 8 and key[1] == 0, (run, key)
        grouped[key].append(row)
    assert set(grouped) == set(metadata)
    for key, samples in grouped.items():
        assert len(samples) == rounds
        validity[run, *key] = all(int(row['valid']) for row in samples)
        medians[run, *key] = median(float(row['cycles_per_mfma']) for row in samples)
    shapes = sorted({key[:4] for key in grouped})
    for shape in shapes:
        values = [medians[run, *shape, u] for u in (4, 8, 32)]
        valid = all(validity[run, *shape, u] for u in (4, 8, 32))
        if valid:
            c4, c8, c32 = values
            b = (c4 - c32) / (1 / 4 - 1 / 32)
            a = c32 - b / 32
            fits[run, *shape] = (a, b, c8 - (a + b / 8))
with (folder / 'summary.csv').open('w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['run', 'kind', 'label', 'order', 'banks', 'requested_ops', 'unroll', 'valid_all_samples', 'median_core_cycles_per_mfma', 'extra_copy_movs_per_mfma', 'nop_wait_units_per_loop'])
    for (run, kind, order, banks, ops, u), value in medians.items():
        meta = metadata[kind, order, banks, ops, u]
        writer.writerow([run, kind, labels[kind], order, banks, ops, u, int(validity[run, kind, order, banks, ops, u]), value, meta['copy_movs_per_mfma'], meta['nop_wait_units_per_loop']])
with (folder / 'fits.csv').open('w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['run', 'kind', 'label', 'order', 'banks', 'requested_ops', 'intercept', 'boundary_cycles_per_loop', 'held_out_u8_residual'])
    for (run, kind, order, banks, ops), (a, b, residual) in fits.items():
        writer.writerow([run, kind, labels[kind], order, banks, ops, a, b, residual])
with (folder / 'invalid.csv').open('w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['run', 'kind', 'order', 'banks', 'ops', 'unroll', 'round', 'max_abs_error', 'bad_outputs'])
    writer.writerows(invalid)
lines = ['Ring results: cycles/MFMA at U=8; fitted boundary cost from U=4/32', 'run | order | banks | C8 | intercept | B | held-out residual']
for run in ('run1', 'run2', 'longer'):
    for order in (0, 1):
        for banks in (2, 4):
            a, b, residual = fits[run, 9, order, banks, 8]
            lines.append(f'{run} | {order} | {banks} | {medians[run,9,order,banks,8,8]:.6f} | {a:.6f} | {b:.6f} | {residual:.6f}')
lines += ['', 'Staged controls, process 1 fitted B cycles/backedge (before / after):', 'kind | count 1 | 2 | 4 | 5 | 6 | 7 | 8']
for kind in range(2, 8):
    vals = []
    for ops in (1, 2, 4, 5, 6, 7, 8):
        before, after = [fits['run1', kind, order, 1, ops][1] for order in (0, 1)]
        vals.append(f'{before:.3f}/{after:.3f}')
    lines.append(labels[kind] + ' | ' + ' | '.join(vals))
lines += ['', 'First B > 1 core cycle/backedge among sampled instruction counts, run1/run2:', 'kind | before | after']
for kind in range(2, 8):
    cells = []
    for order in (0, 1):
        vals = []
        for run in ('run1', 'run2'):
            first = next(ops for ops in (1, 2, 4, 5, 6, 7, 8) if fits[run, kind, order, 1, ops][1] > 1)
            vals.append(str(first))
        cells.append('/'.join(vals))
    lines.append(labels[kind] + ' | ' + ' | '.join(cells))
valid_count = sum(309 * n for n in (9, 9, 3)) - len(invalid)
lines += ['', f'{valid_count} exact-valid samples; {len(invalid)} invalid samples excluded from fits.',
          'All invalid samples are the undelayed paired sign-flip / before case.',
          f'Max held-out residual among valid fits: {max(abs(v[2]) for v in fits.values()):.6f} cycles/MFMA.']
text = '\n'.join(lines) + '\n'
(folder / 'summary.txt').write_text(text)
print(text)
