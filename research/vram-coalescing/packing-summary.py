#!/usr/bin/env python3
"""Summarize assignment and simultaneous marked residence separately."""
import collections
import csv
from pathlib import Path
import re
import sys

out = Path(sys.argv[1])
rate = int(re.search(r"wall_clock_khz=(\d+)", (out / "device.txt").read_text()).group(1))
resources = {}
with (out / "resources.csv").open() as f:
    for row in csv.DictReader(f):
        resources[row["pattern"]] = row

def peak(intervals):
    events = [(start, 1) for start, end in intervals] + [(end, -1) for start, end in intervals]
    active = maximum = 0
    for tick, change in sorted(events):  # ends before starts at equal ticks
        active += change
        maximum = max(maximum, active)
    return maximum

columns = ["pattern", "grid", "sample", "original_CTAs_per_CU", "marked_CTAs_per_CU", "CUs_seen", "min_assigned_CTAs_used_CU", "max_assigned_CTAs_used_CU",
           "min_SIMDs_per_CTA", "max_SIMDs_per_CTA", "peak_CTAs_global", "peak_waves_global",
           "max_peak_CTAs_one_CU", "max_peak_waves_one_CU", "max_peak_waves_one_SIMD",
           "max_same_slot_overlap", "wave_entry_spread_us", "all_wave_common_interval_us", "mean_marked_waves_global"]
with (out / "manifest.csv").open() as f:
    manifest = list(csv.DictReader(f))
with (out / "summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for case in manifest:
        with (out / case["trace"]).open() as raw:
            rows = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(raw)]
        by_cta = collections.defaultdict(list)
        cu_waves = collections.defaultdict(list)
        simd_waves = collections.defaultdict(list)
        slot_waves = collections.defaultdict(list)
        for r in rows:
            interval = (r["start_tick"], r["end_tick"])
            by_cta[r["CTA"]].append(r)
            cu_waves[r["CU_key"]].append(interval)
            simd_waves[(r["CU_key"], r["SIMD"])].append(interval)
            slot_waves[(r["CU_key"], r["SIMD"], r["slot"])].append(interval)
        cu_ctas = collections.defaultdict(list)
        simds_per_cta = []
        for group in by_cta.values():
            cu_ctas[group[0]["CU_key"]].append((min(r["start_tick"] for r in group), max(r["end_tick"] for r in group)))
            simds_per_cta.append(len({r["SIMD"] for r in group}))
        waves = [interval for group in cu_waves.values() for interval in group]
        ctas = [interval for group in cu_ctas.values() for interval in group]
        first, last = min(s for s, e in waves), max(s for s, e in waves)
        earliest_end, latest_end = min(e for s, e in waves), max(e for s, e in waves)
        slot_overlap = max(peak(v) for v in slot_waves.values())
        if slot_overlap > 1:
            raise RuntimeError(f"Overlapping use of the same CU/SIMD/slot: {case}")
        resource = resources[case["pattern"]]
        result = dict(pattern=case["pattern"], grid=case["grid"], sample=case["sample"],
                      original_CTAs_per_CU=resource["original_CTAs_per_CU"], marked_CTAs_per_CU=resource["marked_CTAs_per_CU"], CUs_seen=len(cu_waves),
                      min_assigned_CTAs_used_CU=min(map(len, cu_ctas.values())), max_assigned_CTAs_used_CU=max(map(len, cu_ctas.values())),
                      min_SIMDs_per_CTA=min(simds_per_cta), max_SIMDs_per_CTA=max(simds_per_cta),
                      peak_CTAs_global=peak(ctas), peak_waves_global=peak(waves),
                      max_peak_CTAs_one_CU=max(peak(v) for v in cu_ctas.values()), max_peak_waves_one_CU=max(peak(v) for v in cu_waves.values()),
                      max_peak_waves_one_SIMD=max(peak(v) for v in simd_waves.values()), max_same_slot_overlap=slot_overlap,
                      wave_entry_spread_us=(last-first)*1000/rate,
                      all_wave_common_interval_us=max(0, earliest_end-last)*1000/rate,
                      mean_marked_waves_global=sum(e-s for s,e in waves)/(latest_end-first))
        writer.writerow(result)
print(f"summarized {len(manifest)} placement traces")
