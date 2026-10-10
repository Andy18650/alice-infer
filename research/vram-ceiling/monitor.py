#!/usr/bin/env python3
"""Best-effort 0.5s read-only telemetry for the HIP-selected PCI device."""
import csv
import os
from pathlib import Path
import re
import sys
import time

pid, out = int(sys.argv[1]), Path(sys.argv[2])
with (out / "telemetry.csv").open("w") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["unix_s", "pci", "sclk_Hz", "mclk_Hz", "edge_mC", "junction_mC", "memory_mC", "package_uW", "busy_percent"])
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        try:
            pci = re.search(r"PCI=(\S+)", (out / "device.txt").read_text()).group(1)
            base = Path("/sys/bus/pci/devices") / pci
            hwmon = next((base / "hwmon").glob("hwmon*"))
            def read(path):
                try:
                    return path.read_text().strip()
                except OSError:
                    return "NA"
            w.writerow([time.time(), pci] + [read(hwmon / p) for p in
                       ["freq1_input", "freq2_input", "temp1_input", "temp2_input", "temp3_input", "power1_average"]]
                       + [read(base / "gpu_busy_percent")])
            f.flush()
        except (OSError, AttributeError, StopIteration):
            pass  # initialization may not yet have published device metadata
        time.sleep(0.5)
