#!/usr/bin/env python3
import collections
import csv
from pathlib import Path
import statistics
import sys

out = Path(sys.argv[1])
fields = ["pattern", "chunks", "grid", "block", "bytes"]
groups = collections.defaultdict(list)
with (out / "raw.csv").open() as f:
    for row in csv.DictReader(f):
        groups[tuple(row[k] for k in fields)].append(float(row["GB_s"]))
with (out / "summary.csv").open("w") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(fields + ["samples", "median_GB_s", "min_GB_s", "max_GB_s"])
    for key, values in sorted(groups.items()):
        med = statistics.median(values)
        w.writerow(list(key) + [len(values), med, min(values), max(values)])
        print(" ".join(key), f"{med:.1f} GB/s [{min(values):.1f}, {max(values):.1f}]")
