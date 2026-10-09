#!/usr/bin/env python3
"""Run serial, NUMA-pinned sweeps; keep raw output/errors and machine metadata.

No profiler and no simultaneous experiments. Exit 3 means explicitly skipped
(e.g. cooperative residency limit), never substituted with a different kernel.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "temp/p2p-allreduce/study"
FIELDS = "backend,mode,ws,bytes,blocks,threads,coop_limit,slots,samples,gpu_median_us,gpu_min_us,gpu_p90_us,wall_median_us,output_GBs,correct".split(",")


def integers(text):
    return [int(x) for x in text.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["baseline", "geometry"], default="baseline")
    p.add_argument("--worlds", type=integers, default=[2, 4, 8])
    p.add_argument("--bytes", type=integers, default=[16384, 32768, 65536, 131072, 1048576, 4194304])
    p.add_argument("--backends", default=None)
    p.add_argument("--blocks", type=integers, default=[1, 2, 4, 8, 16, 32, 64, 128, 220])
    p.add_argument("--threads", type=integers, default=[256, 1024])
    p.add_argument("--slots", type=int, default=160)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--device-base", type=int, default=0)
    p.add_argument("--mode", choices=["graph", "eager"], default="graph")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not BIN.exists():
        p.error("run research/p2p-allreduce/build.sh first")
    meta = vars(args).copy()
    meta["out"] = str(args.out)
    meta["placement"] = "numactl --cpunodebind=0 --membind=0"
    meta["context"] = "no attention/model; bytes=4*S*hidden, slots=collectives per forward"
    source = Path(__file__).resolve().parent
    meta["sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [BIN, source / "study.cpp", source / "kernels.hip",
                     source / "kernels.hpp", source / "build.sh"]
    }
    for key, command in [("git", ["git", "rev-parse", "HEAD"]),
                         ("compiler", ["hipcc", "--version"]),
                         ("numa", ["numactl", "--hardware"])]:
        meta[key] = subprocess.check_output(command, cwd=ROOT, text=True).strip()
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    names = args.backends.split(",") if args.backends else (
        ["fill", "barrier", "fused", "pipe", "wide", "elected", "coop", "coop-wide"]
        if args.suite == "baseline" else ["pipe", "wide", "elected", "coop"])
    geometry = [(0, 256)] if args.suite == "baseline" else list(itertools.product(args.blocks, args.threads))
    cases = list(itertools.product(args.worlds, args.bytes, names, geometry))
    skipped = []
    with (args.out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for index, (ws, size, name, (blocks, threads)) in enumerate(cases, 1):
            tag = f"{name}-ws{ws}-n{size}-b{blocks}-t{threads}-{args.mode}"
            command = ["numactl", "--cpunodebind=0", "--membind=0", str(BIN),
                       name, str(ws), str(size), str(blocks), str(threads),
                       str(args.slots), str(args.samples), str(args.device_base), args.mode]
            print(f"[{index}/{len(cases)}] {tag}", flush=True)
            try:
                result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                        timeout=args.timeout)
            except subprocess.TimeoutExpired as e:
                (args.out / f"{tag}.timeout").write_text(str(e) + "\n")
                sys.exit("timeout: stop the sweep; do not continue with potentially spinning GPUs")
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
    print(f"saved {args.out / 'results.csv'} ({len(skipped)} skipped)")


if __name__ == "__main__":
    main()
