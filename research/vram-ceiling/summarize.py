#!/usr/bin/env python3
"""Print screening winners and summarize randomized confirmation rounds."""
import collections
import csv
from pathlib import Path
import statistics
import sys

out = Path(sys.argv[1])
key_fields = ["pattern", "load_bytes", "unroll", "block", "grid", "bytes"]
with (out / "sweep.csv").open() as f:
    screening = list(csv.DictReader(f))
print("Screen top 8 (1 GiB, three batch samples):")
for row in sorted(screening, key=lambda r: float(r["median_GB_s"]), reverse=True)[:8]:
    print(" ".join(row[k] for k in key_fields), row["median_GB_s"], "GB/s")
groups = collections.defaultdict(list)
with (out / "confirm-raw.csv").open() as f:
    for row in csv.DictReader(f):
        groups[tuple(row[k] for k in key_fields)].append(float(row["GB_s"]))
with (out / "confirm.csv").open("w") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(key_fields + ["samples", "median_GB_s", "min_GB_s", "max_GB_s"])
    for key, values in sorted(groups.items(), key=lambda kv: (int(kv[0][-1]), -statistics.median(kv[1]))):
        med, low, high = statistics.median(values), min(values), max(values)
        w.writerow(list(key) + [len(values), med, low, high])
        print(" ".join(key), f"median={med:.1f} min={low:.1f} max={high:.1f} GB/s")
