#!/usr/bin/env python3
"""Describe the untuned real-load baseline; no backedge fitting or tuning."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'results/load')
labels = ['resident_fp16', 'resident_fp8_8_8', 'stream_fp16_wait0', 'stream_fp8_wait0']
metadata = {tuple(row[k] for k in ('mode', 'unroll')): row for row in json.loads((folder / 'audit.json').read_text())['kernels']}
assert len(metadata) == 12
fields = ('cycles_per_mfma', 'ms', 'TFLOPs', 'useful_weight_GBs', 'requested_weight_GBs', 'core_MHz')
summary = []
for run, rounds, steps in [('run1', 9, 4096), ('run2', 9, 4096), ('longer', 3, 8192)]:
    rows = list(csv.DictReader((folder / f'{run}.csv').open()))
    assert len(rows) == 12 * rounds
    assert 'PASS 24 changed-input/poisoned-output cases and output guards' in (folder / f'{run}.log').read_text()
    grouped = defaultdict(list); seen = set()
    for row in rows:
        mode, unroll, round_ = (int(row[k]) for k in ('mode', 'unroll', 'round'))
        assert (mode, unroll, round_) not in seen
        seen.add((mode, unroll, round_))
        assert int(row['steps']) == steps
        assert int(row['valid']) == 1 and float(row['max_abs_error']) == 0
        if mode >= 2:
            intensity = 32 if mode == 2 else 64
            assert abs(float(row['useful_weight_GBs']) - float(row['TFLOPs']) * 1000 / intensity) < 0.001
            assert abs(float(row['requested_weight_GBs']) / float(row['useful_weight_GBs']) - (steps + 2) / steps) < 1e-6
        else:
            assert float(row['useful_weight_GBs']) == float(row['requested_weight_GBs']) == 0
        grouped[mode, unroll].append(row)
    assert set(grouped) == set(metadata)
    for (mode, unroll), samples in sorted(grouped.items()):
        assert len(samples) == rounds
        values = {field: median(float(row[field]) for row in samples) for field in fields}
        minimum = min(float(row['cycles_per_mfma']) for row in samples)
        maximum = max(float(row['cycles_per_mfma']) for row in samples)
        summary.append([run, steps, mode, labels[mode], unroll, *(values[f] for f in fields), minimum, maximum])
with (folder / 'summary.csv').open('w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['run', 'steps', 'mode', 'label', 'unroll', *fields, 'min_cycles_per_mfma', 'max_cycles_per_mfma'])
    writer.writerows(summary)
lines = ['Untuned real-load baseline; immediate wait per loaded future operand.',
         'run | U | mode | cycles/MFMA | TFLOP/s | useful weight GB/s | core MHz | min..max cycles']
for row in summary:
    run, steps, mode, label, unroll, cycles, ms, tf, useful, requested, mhz, minimum, maximum = row
    lines.append(f'{run} | {unroll} | {label} | {cycles:.3f} | {tf:.3f} | {useful:.3f} | {mhz:.1f} | {minimum:.3f}..{maximum:.3f}')
lines += ['', 'PASS all 252 timed samples exact against FP64; all 72 changed-input/poisoned-output cases and guards passed.',
          'No fits are used to assign a fixed backedge cost under variable memory stalls.',
          'No prefetch-depth, per-MFMA or backedge tuning has been performed.']
text = '\n'.join(lines) + '\n'
(folder / 'summary.txt').write_text(text)
print(text)
