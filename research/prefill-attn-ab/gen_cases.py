#!/usr/bin/env python3
"""Generate attention A/B test cases: Q/K/V fp32 + PyTorch SDPA fp32 reference.

Case semantics (design agreement): query s of S attends keys [0, T-S+s]
(causal continuation prefill; S=T is from-scratch prefill, S=1 is decode).
Single-head correctness cases (H=1) + multi-head timing cases (H=32, to fill
110 CUs the way 32 real heads would).

Binary format (little-endian): magic 'ATT1', u32 H, S, T, hd, then
q [H,S,hd], k [H,T,hd], v [H,T,hd], ref [H,S,hd] as fp32 C-order.

Usage: gen_cases.py [out_dir]   (default: cases/)
"""

import os
import struct
import sys

import torch

CORRECTNESS = [  # (T, S) — includes non-full-chunk and chunk-straddling sizes
    (1, 1),
    (63, 63),
    (64, 64),
    (65, 65),
    (130, 65),
    (1000, 512),
    (1000, 1000),
    (1500, 1),
    (2048, 512),
]
TIMING = [(1000, 512), (2048, 512), (4000, 512), (1000, 1000)]
DECODE_TIMING = [  # S=1 decode across context lengths + S=4 boundary probes
    (1, 1),
    (8, 1),
    (32, 1),
    (64, 1),
    (128, 1),
    (256, 1),
    (512, 1),
    (1000, 1),
    (512, 4),
    (1000, 4),
]
HDS = [64, 128]


def make(out_dir: str, hd: int, T: int, S: int, H: int) -> str:
    g = torch.Generator().manual_seed(hd * 1_000_000 + T * 100 + S)
    q = torch.randn(H, S, hd, generator=g)
    k = torch.randn(H, T, hd, generator=g)
    v = torch.randn(H, T, hd, generator=g)
    # explicit additive mask (not is_causal: its S != T alignment is
    # version-dependent)
    mask = torch.zeros(S, T)
    for s in range(S):
        mask[s, T - S + s + 1 :] = float("-inf")
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=mask
    ).squeeze(1)
    name = f"c_{hd}_{T}_{S}_h{H}.bin"
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(b"ATT1")
        f.write(struct.pack("<4I", H, S, T, hd))
        for t in (q, k, v, ref):
            f.write(t.numpy().astype("<f4").tobytes())
    return name


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "cases"
    os.makedirs(out_dir, exist_ok=True)
    for hd in HDS:
        for T, S in CORRECTNESS:
            print(make(out_dir, hd, T, S, H=1))
        for T, S in TIMING:
            print(make(out_dir, hd, T, S, H=32))
        for T, S in DECODE_TIMING:
            print(make(out_dir, hd, T, S, H=32))


if __name__ == "__main__":
    main()
