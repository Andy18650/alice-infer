#!/usr/bin/env python3
"""Pre-flight audit of an alice manifest against safetensors shard HEADERS.

Mirrors the engine's two-way audit (manifest.validate) without loading any
weight data: every claimed tensor must exist with matching shape/dtype,
every tensor in every shard must be claimed, and no name may appear in two
shards. Cheap enough for the 70B (17 header reads, no 141 GB pread).

Usage: check_manifest.py <model_dir> <manifest.json>
"""
import glob
import json
import struct
import sys


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    return {k: v for k, v in h.items() if k != "__metadata__"}

def main():
    model_dir, manifest_path = sys.argv[1], sys.argv[2]
    errors = []

    tensors = {}  # name -> (shape, dtype, shard)
    for shard in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        for name, v in read_header(shard).items():
            if name in tensors:
                errors.append(f"duplicate across shards: {name} "
                              f"({tensors[name][2]} and {shard})")
                continue
            tensors[name] = (v["shape"], v["dtype"], shard)
    print(f"{len(tensors)} tensors across shards")

    m = json.load(open(manifest_path))
    claimed = set()
    embedder_name = m["embedder"]["token_embedding"].get("name")

    def check(spec, what):
        name = spec["name"]
        if not name:
            errors.append(f"{what}: empty tensor name")
            return
        # tied embeddings: the lm_head may share the embedder's tensor —
        # the one legal shared claim (mirrors validate())
        tied = what == "lm_head" and name == embedder_name
        if name in claimed and not tied:
            errors.append(f"{what}: claimed twice: {name}")
        claimed.add(name)
        if name not in tensors:
            errors.append(f"{what}: not found: {name}")
            return
        shape, dtype, _ = tensors[name]
        if list(spec["shape"]) != list(shape):
            errors.append(f"{what}: shape mismatch on {name}: "
                          f"manifest {spec['shape']} vs file {shape}")
        # expected dtype is derived (all current block types are bf16), not
        # declared per weight (2026-09 manifest diet)
        if dtype != "BF16":
            errors.append(f"{what}: dtype mismatch on {name}: "
                          f"expected BF16, file has {dtype}")

    check(m["embedder"]["token_embedding"], "embedder")
    check(m["lm_head"]["weight"], "lm_head")
    for i, n in enumerate(m["trunk"]["pre_norms"]):
        check(n, f"trunk.pre_norms[{i}]")
    check(m["trunk"]["final_norm"], "trunk.final_norm")
    for i, b in enumerate(m["blocks"]):
        for role, spec in b["weights"].items():
            check(spec, f"blocks[{i}].{role}")

    for name in tensors:
        if name not in claimed:
            errors.append(f"unclaimed: {name}")

    if errors:
        print(f"CONTRACT VIOLATED — {len(errors)} problem(s):")
        for e in errors[:40]:
            print("  -", e)
        sys.exit(1)
    print(f"contract holds: {len(claimed)} claims cover {len(tensors)} tensors")


if __name__ == "__main__":
    main()
