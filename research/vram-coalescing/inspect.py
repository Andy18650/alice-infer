#!/usr/bin/env python3
"""Check hot-load widths, not the masked scalar tail, before benchmarking."""
import csv
from pathlib import Path
import re
import sys

out, mode = Path(sys.argv[1]), sys.argv[2]
text = (out / "disassembly.txt").read_text()
functions = re.split(r"(?=^0[0-9a-f]+ <)", text, flags=re.MULTILINE)
with (out / "load-widths.csv").open("w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["reader", "chunks", "hot_loads_before_first_wait", "load_bytes", "cache_flags"])
    for tag in "ABC":
        for chunks in [1, 2, 4, 8]:
            code = next(c for c in functions if f"read_tiles<{tag}, {chunks}>" in c)
            first = code.index("\tglobal_load_")
            first_wait = code.index("\ts_waitcnt vmcnt", first)
            hot = code[first:first_wait]
            loads = re.findall(r"\bglobal_load_(dword(?:x[24])?)\b", hot)
            widths = {"dword": 4, "dwordx2": 8, "dwordx4": 16}
            flags = sorted(set(re.findall(r"\b(?:glc|slc|dlc)\b", hot)))
            expected = ["dwordx4"] * chunks if tag == "A" else ["dword"] * (4 * chunks)
            if mode == "controlled" or tag != "C":
                if loads != expected:
                    raise RuntimeError(f"Unexpected hot loads: {tag}/{chunks}: {loads}")
            if flags:
                raise RuntimeError(f"Unexpected cache-policy flags: {tag}/{chunks}: {flags}")
            if re.search(r"\bscratch_", code):
                raise RuntimeError(f"Scratch traffic in {tag}/{chunks}")
            writer.writerow([tag, chunks, len(loads), " ".join(str(widths[x]) for x in loads), "none"])
            print(f"{tag}/{chunks}: {len(loads)} loads, widths={sorted(set(widths[x] for x in loads))} bytes")
