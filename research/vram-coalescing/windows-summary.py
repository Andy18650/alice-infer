#!/usr/bin/env python3
"""Summarize address windows and warmup stages; never merge unlike workloads."""
import collections
import csv
from pathlib import Path
import statistics
import sys

out = Path(sys.argv[1])
fields = ['stage', 'pattern', 'chunks', 'grid', 'block', 'kind', 'bytes', 'offset_bytes', 'iters']
groups = collections.defaultdict(list)
for row in csv.DictReader((out / 'windows-raw.csv').open()):
    groups[tuple(row[k] for k in fields)].append(row)
with (out / 'windows-summary.csv').open('w') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(fields + ['samples', 'median_GB_s', 'min_GB_s', 'max_GB_s', 'median_ms', 'median_wall_ms'])
    for key, rows in sorted(groups.items()):
        values = [float(r['GB_s']) for r in rows]
        median = statistics.median(values)
        ms = statistics.median(float(r['ms']) for r in rows)
        wall = statistics.median(float(r['wall_ms']) for r in rows) if 'wall_ms' in rows[0] else 'NA'
        w.writerow(list(key) + [len(values), median, min(values), max(values), ms, wall])
        print(' '.join(key), f'{median:.1f} GB/s [{min(values):.1f}, {max(values):.1f}], {ms:.4f} ms')
