#!/usr/bin/env python3
"""Preserve all source/layout/scheduling axes and aggregate paired rounds."""
import collections
import csv
from pathlib import Path
import statistics
import sys

out = Path(sys.argv[1])
fields = ["source", "pattern", "unroll", "policy", "grid", "block", "bytes", "offset"]
groups = collections.defaultdict(list)
with (out / "raw.csv").open() as f:
    for row in csv.DictReader(f):
        groups[tuple(row[k] for k in fields)].append(float(row["GB_s"]))
with (out / "summary.csv").open("w") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(fields + ["samples", "median_GB_s", "min_GB_s", "max_GB_s"])
    for key, values in sorted(groups.items()):
        w.writerow(list(key) + [len(values), statistics.median(values), min(values), max(values)])
print(f"summarized {len(groups)} configuration/size pairs")
