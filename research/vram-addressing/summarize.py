#!/usr/bin/env python3
"""Summarize randomized batch samples, keeping all geometry/address axes."""
import collections
import csv
from pathlib import Path
import statistics
import sys

out = Path(sys.argv[1])
fields = ["group", "tile", "streams", "unroll", "block", "grid", "stride", "base", "delta", "bytes", "span_per_stream", "plane_gap"]
groups = collections.defaultdict(list)
with (out / "raw.csv").open() as f:
    for row in csv.DictReader(f):
        row.setdefault("plane_gap", str(8 << 30))  # earlier fixed-gap runs
        groups[tuple(row[k] for k in fields)].append(float(row["GB_s"]))
with (out / "summary.csv").open("w") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(fields + ["samples", "median_GB_s", "min_GB_s", "max_GB_s"])
    for key, values in sorted(groups.items()):
        w.writerow(list(key) + [len(values), statistics.median(values), min(values), max(values)])
print(f"summarized {len(groups)} cases")
