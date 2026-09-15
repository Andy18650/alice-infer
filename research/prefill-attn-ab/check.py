#!/usr/bin/env python3
"""Compare scheme outputs against the case's SDPA fp32 reference.

Usage: check.py <case.bin> [case2.bin ...]
Gate (design agreement: fp32 everywhere => fp-ordering noise): max abs
diff < 2e-4 (expected ~1e-5; 20x margin). Prints per-scheme diffs.
"""

import struct
import sys

import numpy as np

GATE = 2e-4


def load(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"ATT1", path
        H, S, T, hd = struct.unpack("<4I", f.read(16))
        n = H * S * hd
        f.seek(20 + (n + 2 * H * T * hd) * 4)  # skip magic+header, q, k, v
        ref = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(H, S, hd)
    return ref


def main() -> int:
    worst_fail = False
    for path in sys.argv[1:]:
        ref = load(path)
        for scheme in ("v1", "A", "A2", "B"):
            out = np.fromfile(path + ".out_" + scheme, dtype="<f4").reshape(ref.shape)
            abs_diff = float(np.max(np.abs(out - ref)))
            rel_diff = float(np.max(np.abs(out - ref) / (1.0 + np.abs(ref))))
            ok = abs_diff < GATE
            worst_fail |= not ok
            print(
                f"{path.split('/')[-1]:28s} {scheme:3s} abs={abs_diff:.3e} "
                f"rel={rel_diff:.3e} {'PASS' if ok else 'FAIL'}"
            )
    return 1 if worst_fail else 0


if __name__ == "__main__":
    sys.exit(main())
