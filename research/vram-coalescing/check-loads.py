#!/usr/bin/env python3
"""Inspect load batches; enforce old shapes only in the barrier control."""
import csv
from pathlib import Path
import re
import sys

out, mode = Path(sys.argv[1]), sys.argv[2]
barriers = sys.argv[3] if len(sys.argv) > 3 else "on"
text = (out / "disassembly.txt").read_text()
functions = re.split(r"(?=^0[0-9a-f]+ <)", text, flags=re.MULTILINE)
widths = {"dword": 4, "dwordx2": 8, "dwordx4": 16}
load_pattern = r"\bglobal_load_(dword(?:x[24])?)\b"
with (out / "load-widths.csv").open("w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["reader", "chunks", "hot_loads_before_first_wait", "load_bytes", "cache_flags",
                     "scheduling_barriers", "loop_loads", "loop_load_bytes", "loop_waits"])
    for tag in "ABC":
        for chunks in [1, 2, 4, 8]:
            code = next(c for c in functions if f"read_tiles<{tag}, {chunks}>" in c)
            first = code.index("\tglobal_load_")
            first_wait = code.index("\ts_waitcnt vmcnt", first)
            before_wait = code[first:first_wait]
            loads = re.findall(load_pattern, before_wait)
            # The first backward branch after the hot loads closes the outer
            # tile loop. Include its full issue/consume schedule, not the tail.
            loop_lines = []
            for line in code[first:].splitlines():
                loop_lines.append(line)
                branch = re.search(r"\bs_(?:cbranch_[a-z0-9]+|branch)\s+(-?(?:0x[0-9a-f]+|[0-9]+))", line)
                if branch:
                    displacement = int(branch.group(1), 0)
                    if displacement < 0 or displacement > 32767:
                        break
            else:
                raise RuntimeError(f"No backward tile-loop branch: {tag}/{chunks}")
            loop = "\n".join(loop_lines)
            loop_loads = re.findall(load_pattern, loop)
            waits = re.findall(r"s_waitcnt vmcnt\((\d+)\)", loop)
            flags = sorted(set(re.findall(r"\b(?:glc|slc|dlc)\b", loop)))
            if barriers == "on":
                if sum(widths[x] for x in loads) != 16 * chunks:
                    raise RuntimeError(f"Unexpected bytes per hot batch: {tag}/{chunks}: {loads}")
                if tag == "A":
                    if any(x not in ["dwordx2", "dwordx4"] for x in loads):
                        raise RuntimeError(f"Scalarized vector reader: A/{chunks}: {loads}")
                elif tag == "B" or mode == "controlled":
                    if loads != ["dword"] * (4 * chunks):
                        raise RuntimeError(f"Unexpected scalar reader: {tag}/{chunks}: {loads}")
            # Removing barriers deliberately permits a different issue/use
            # schedule and load merging. Observe it rather than rejecting it.
            if flags:
                raise RuntimeError(f"Unexpected cache-policy flags: {tag}/{chunks}: {flags}")
            if re.search(r"\bscratch_", code):
                raise RuntimeError(f"Scratch traffic in {tag}/{chunks}")
            writer.writerow([tag, chunks, len(loads), " ".join(str(widths[x]) for x in loads), "none", barriers,
                             len(loop_loads), " ".join(str(widths[x]) for x in loop_loads), " ".join(waits)])
            print(f"{tag}/{chunks}: {len(loads)} loads before first wait; {len(loop_loads)} in loop; widths={sorted(set(widths[x] for x in loop_loads))} bytes")
