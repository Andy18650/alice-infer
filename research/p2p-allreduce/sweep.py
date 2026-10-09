#!/usr/bin/env python3
"""Small, serial traversal experiment. No broad Cartesian grid by default.

Default: 32 KiB and 1 MiB at two useful geometries each, ws=4/8,
and three surviving gather paths (30 cases). Fill/fused controls run once.
Peer-major and few-lane points remain available explicitly, not in the default.
--suite prefetch runs 32 cases at two geometries with enough per-thread work
for two/four-vector lookahead. --suite register runs 40 selected comparisons,
through 4 MiB only. Custom requires explicit --bytes and --points.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "temp/p2p-allreduce/study"
FIELDS = "backend,mode,ws,bytes,blocks,threads,slots,samples,gpu_median_us,gpu_min_us,gpu_p90_us,wall_median_us,output_GBs,correct".split(",")
PROFILES = {
    32768: [(2, 1024), (8, 256)],
    1048576: [(8, 256), (32, 256)],
}


def integers(text):
    return [int(x) for x in text.split(",")]


def points(text):
    return [tuple(map(int, x.split("x"))) for x in text.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["traversal", "prefetch", "register", "custom"], default="traversal")
    p.add_argument("--worlds", type=integers, default=[4, 8])
    p.add_argument("--bytes", type=integers)
    p.add_argument("--points", type=points)
    p.add_argument("--backends")
    p.add_argument("--slots", type=int, default=64)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--device-base", type=int, default=0)
    p.add_argument("--mode", choices=["graph", "eager"], default="graph")
    p.add_argument("--timeout", type=int, default=45)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.suite == "custom" and (not args.bytes or not args.points):
        p.error("custom runs require --bytes and --points; no implicit large grid")
    if args.suite != "custom" and (args.bytes or args.points):
        p.error("use --suite custom to override payloads/geometries")
    cases = []
    if args.suite == "register":
        # Small static accumulator budgets, not a Cartesian tile-size sweep.
        # 32-vector tiles use <=256 threads; 16 MiB is intentionally excluded.
        recipes = [
            (32768, 2, 1024, "phase,peer-major,peer-reg1"),
            (32768, 2, 256, "phase,peer-major,peer-reg4"),
            (1048576, 8, 256, "hoisted,phase,prefetch4,peer-major,peer-reg8,peer-reg32"),
            (1048576, 32, 256, "phase,prefetch2,peer-major,peer-reg8"),
            (4194304, 32, 256, "phase,peer-major,peer-reg8,peer-reg32"),
        ]
        for ws in args.worlds:
            for size, blocks, threads, names in recipes:
                for name in (args.backends or names).split(","):
                    cases.append((ws, size, name, blocks, threads))
    else:
        if args.suite == "prefetch":
            profiles = {32768: [(2, 256)], 1048576: [(8, 256)]}
            defaults = "hoisted,phase,hoist-prefetch2,prefetch2,prefetch4,peer-major,peer-prefetch2,peer-prefetch4"
        else:
            profiles = {size: args.points for size in args.bytes} if args.suite == "custom" else PROFILES
            defaults = "pipe,chunk,phase"
        names = (args.backends or defaults).split(",")
        for ws in args.worlds:
            for size, geometries in profiles.items():
                for blocks, threads in geometries:
                    for name in names:
                        cases.append((ws, size, name, blocks, threads))
            if args.suite == "traversal":
                cases.extend((ws, size, "fill", 0, 256) for size in profiles)
                cases.append((ws, 32768, "fused", 1, 1024))
    args.out.mkdir(parents=True, exist_ok=True)
    if not BIN.exists():
        p.error("run research/p2p-allreduce/build.sh first")
    meta = vars(args).copy()
    meta["out"] = str(args.out)
    meta["cases"] = len(cases)
    meta["placement"] = "numactl --cpunodebind=0 --membind=0"
    meta["context"] = "no attention/model; bytes=4*S*hidden"
    source = Path(__file__).resolve().parent
    meta["sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [BIN, source / "study.cpp", source / "kernels.hip", source / "kernels.hpp"]
    }
    meta["git"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    meta["compiler"] = subprocess.check_output(["hipcc", "--version"], text=True).strip()
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    skipped = []
    start = time.monotonic()
    with (args.out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for index, (ws, size, name, blocks, threads) in enumerate(cases, 1):
            tag = f"{name}-ws{ws}-n{size}-b{blocks}-t{threads}-{args.mode}"
            command = ["numactl", "--cpunodebind=0", "--membind=0", str(BIN),
                       name, str(ws), str(size), str(blocks), str(threads),
                       str(args.slots), str(args.samples), str(args.device_base), args.mode]
            print(f"[{index}/{len(cases)}; {time.monotonic()-start:.1f}s] {tag}", flush=True)
            try:
                result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                        timeout=args.timeout)
            except subprocess.TimeoutExpired as e:
                (args.out / f"{tag}.timeout").write_text(str(e) + "\n")
                sys.exit("timeout: stop; do not continue with potentially spinning GPUs")
            (args.out / f"{tag}.stdout").write_text(result.stdout)
            (args.out / f"{tag}.stderr").write_text(result.stderr)
            if result.returncode == 3:
                skipped.append({"case": tag, "reason": result.stderr.strip()})
                continue
            if result.returncode:
                sys.exit(f"failed {tag}:\n{result.stderr}")
            rows = list(csv.DictReader(result.stdout.splitlines()))
            if len(rows) != 1 or rows[0]["correct"] != "OK":
                sys.exit(f"invalid result for {tag}: {result.stdout}")
            writer.writerow(rows[0])
            f.flush()
    (args.out / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n")
    print(f"saved {args.out / 'results.csv'}; {time.monotonic()-start:.1f}s; {len(skipped)} skipped")


if __name__ == "__main__":
    main()
