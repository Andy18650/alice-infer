#!/usr/bin/env python3
"""Inspect load batches, ordering controls, waits and register-lane moves."""
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
kernels = [(tag, chunks, "grid", f"read_tiles<{tag}, {chunks}>") for tag in "ABC" for chunks in [1, 2, 4, 8, 16]]
kernels += [(tag, chunks, "adjacent", f"read_adjacent_tiles<{tag}, {chunks}>") for tag in "AB" for chunks in [4, 8, 16]]
kernels += [(tag, chunks, order+"_body4", f"read_window_bodies<{tag}, {chunks}, {adjacent}>")
            for tag, chunks in [("A", 4), ("B", 4), ("B", 16)]
            for order, adjacent in [("grid", "false"), ("adjacent", "true")]]
with (out / "load-widths.csv").open("w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["reader", "chunks", "hot_loads_before_first_wait", "load_bytes", "cache_flags",
                     "scheduling_barriers", "loop_loads", "loop_load_bytes", "loop_waits",
                     "loads_to_branch_lines", "loads_to_branch_lane_reads", "loads_to_branch_lane_writes",
                     "chunk_order", "kernel_accvgpr_reads", "kernel_accvgpr_writes"])
    for tag, chunks, order, signature in kernels:
        code = next((c for c in functions if signature in c), None)
        if code is None:
            if order != "grid" or chunks == 16:  # Optional newer experiments.
                continue
            raise RuntimeError(f"Missing kernel: {tag}/{chunks}/{order}")
        first = code.index("\tglobal_load_")
        first_wait = code.index("\ts_waitcnt vmcnt", first)
        loads = re.findall(load_pattern, code[first:first_wait])
        # Segment from first hot load through the backward outer-loop branch;
        # excludes pre-load setup and masked tails, even for wide readers.
        loop_lines = []
        for line in code[first:].splitlines():
            loop_lines.append(line)
            branch = re.search(r"\bs_(?:cbranch_[a-z0-9]+|branch)\s+(-?(?:0x[0-9a-f]+|[0-9]+))", line)
            if branch:
                displacement = int(branch.group(1), 0)
                if displacement < 0 or displacement > 32767:
                    break
        else:
            raise RuntimeError(f"No backward tile-loop branch: {tag}/{chunks}/{order}")
        loop = "\n".join(loop_lines)
        loop_loads = re.findall(load_pattern, loop)
        waits = re.findall(r"s_waitcnt vmcnt\((\d+)\)", loop)
        flags = sorted(set(re.findall(r"\b(?:glc|slc|dlc)\b", loop)))
        if barriers != "off":
            if sum(widths[x] for x in loads) != 16 * chunks:
                raise RuntimeError(f"Unexpected bytes per hot batch: {tag}/{chunks}/{order}: {loads}")
            if tag == "A":
                if any(x not in ["dwordx2", "dwordx4"] for x in loads):
                    raise RuntimeError(f"Scalarized vector reader: A/{chunks}/{order}: {loads}")
            elif tag == "B" or mode == "controlled":
                if loads != ["dword"] * (4 * chunks):
                    raise RuntimeError(f"Unexpected scalar reader: {tag}/{chunks}/{order}: {loads}")
        # Barrier-free runs deliberately permit scheduling/load-merging changes.
        if flags:
            raise RuntimeError(f"Unexpected cache-policy flags: {tag}/{chunks}/{order}: {flags}")
        if re.search(r"\bscratch_", code):
            raise RuntimeError(f"Scratch traffic in {tag}/{chunks}/{order}")
        reads, writes = loop.count("v_readlane_b32"), loop.count("v_writelane_b32")
        writer.writerow([tag, chunks, len(loads), " ".join(str(widths[x]) for x in loads), "none", barriers,
                         len(loop_loads), " ".join(str(widths[x]) for x in loop_loads), " ".join(waits),
                         len(loop_lines), reads, writes, order,
                         code.count("v_accvgpr_read_b32"), code.count("v_accvgpr_write_b32")])
        print(f"{tag}/{chunks}/{order}: {len(loads)} loads before first wait; {len(loop_loads)} in loop; "
              f"widths={sorted(set(widths[x] for x in loop_loads))} bytes; lane reads/writes={reads}/{writes}")
