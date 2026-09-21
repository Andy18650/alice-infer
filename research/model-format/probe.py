#!/usr/bin/env python3
"""Compression probe for a potential alice model format (EXPLORATORY).

Question: what can compression do to bf16 tensor bytes, at per-tensor or
per-block granularity, and what does it cost?

Families measured on real 8B (deepseek-r1-distill-llama-8b) tensors:
  1. general-purpose lossless: zstd (1/3/19), zlib-6, lzma-6, lz4-1
  2. structural lossless (bf16 byte-split: sign+exp byte vs mantissa byte,
     each zstd'd; plus order-0 entropy = the theoretical ceiling)
  3. cross-tensor redundancy: zstd dictionary trained on other blocks
  4. lossy RTN quantization: int8 (rowwise / group128), int4 (group 32/64/128)
  5. lossless-after-lossy: zstd on the int8 stream

Ratio convention: original bf16 bytes / stored bytes (fp16 scales counted).
Error convention: ||x_hat - x||_2 / ||x||_2, computed in fp32, x = bf16 truth.
"""
import time

import numpy as np
import torch
from safetensors import safe_open

MODEL = "/opt/models/deepseek-ai/deepseek-r1-distill-llama-8b"
SHARDS = [f"{MODEL}/model-0000{i}-of-000002.safetensors" for i in (1, 2)]

import lz4.frame
import zstandard as zstd

MB = 1 << 20


def load_tensors(wanted):
    """wanted: {hf_name: label}. Returns [(label, torch bf16 tensor)]."""
    found = {}
    for shard in SHARDS:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if key in wanted and key not in found:
                    found[key] = f.get_tensor(key)
    return [(wanted[k], t) for k, t in found.items() if k in wanted]


def bf16_bytes(t):
    return t.contiguous().view(torch.uint16).numpy().tobytes()


def ratio(orig_len, comp_len):
    return orig_len / comp_len


def timeit(fn, *a):
    t0 = time.perf_counter()
    out = fn(*a)
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------- 1. lossless
def lossless_suite(label, data, results):
    n = len(data)
    row = {"tensor": label, "MB": n / MB}
    for lvl in (1, 3, 19):
        c = zstd.ZstdCompressor(level=lvl, threads=16)
        (blob, dt) = timeit(c.compress, data)
        row[f"zstd{lvl}"] = ratio(n, len(blob))
        if lvl == 3:
            d = zstd.ZstdDecompressor()
            (_, ddt) = timeit(d.decompress, blob, n)
            row["zstd3_dec_MBps"] = n / ddt / MB
    row["zstd19_MBps"] = n / dt / MB  # dt is from the last loop pass (lvl 19)
    # slow coders on a 64 MB slice (ratio only)
    sl = data[: 64 * MB]
    row["zlib6"] = ratio(len(sl), len(__import__("zlib").compress(sl, 6)))
    import lzma
    row["lzma6"] = ratio(len(sl), len(lzma.compress(sl, preset=6)))
    row["lz4"] = ratio(n, len(lz4.frame.compress(data)))
    results.append(row)
    return row


# --------------------------------------------- 2. byte split + entropy bound
def split_suite(label, t, results):
    data = bf16_bytes(t)
    u16 = np.frombuffer(data, dtype=np.uint16)
    hi = (u16 >> 8).astype(np.uint8).tobytes()   # sign + 7 exp bits
    lo = (u16 & 0xFF).astype(np.uint8).tobytes()  # 1 exp bit + 7 mantissa bits
    c19 = zstd.ZstdCompressor(level=19, threads=16)

    def ent(b):
        h = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
        p = h[h > 0] / len(b)
        return float(-(p * np.log2(p)).sum())

    n = len(data)
    results.append({
        "tensor": label,
        "hi_zstd19": ratio(n, len(c19.compress(hi)) + len(c19.compress(lo))),
        "hi_ent_bits": ent(hi),   # order-0 ceiling for the sign+exp byte
        "lo_ent_bits": ent(lo),   # order-0 ceiling for the mantissa byte
    })


# ------------------------------------------------------ 3. zstd dict transfer
def dict_suite(q0, q15, q31):
    """Train a dict on blocks 15+31 (64 KB samples); compress block 0."""
    c = zstd.ZstdCompressor(level=19, threads=16)
    base = ratio(len(bf16_bytes(q0)), len(c.compress(bf16_bytes(q0))))

    def samples(t):
        b = bf16_bytes(t)
        return [b[i : i + 65536] for i in range(0, len(b), 65536)]

    trainer = zstd.train_dictionary(16 * MB, samples(q15) + samples(q31), level=19)
    dc = zstd.ZstdCompressor(level=19, dict_data=trainer)
    withdict = ratio(len(bf16_bytes(q0)), len(dc.compress(bf16_bytes(q0))))
    return base, withdict


# ---------------------------------------------------------- 4. lossy RTN
def rtn(t, bits, group, results, label):
    x = t.float().flatten()
    g = x.view(-1, group)
    amax = g.abs().amax(dim=1, keepdim=True)
    qmax = float(2 ** (bits - 1) - 1)
    scale = (amax / qmax).clamp_min(1e-12).half().float()  # fp16 scale
    q = torch.round(g / scale).clamp(-qmax - 1, qmax)
    xhat = (q * scale).view_as(x)
    rel = float((xhat - x).norm() / x.norm())
    bits_per = bits + 16.0 / group  # weights + fp16 scale overhead
    results.append({
        "tensor": label, "codec": f"int{bits} g{group}",
        "ratio": 16.0 / bits_per, "rel_err": rel,
    })


# ===========================================================================
def main():
    lossless, splits, lossy = [], [], []
    wanted = {}
    for blk in (0, 15, 31):
        p = f"model.layers.{blk}"
        wanted.update({
            f"{p}.self_attn.q_proj.weight": f"b{blk}.q",
            f"{p}.mlp.down_proj.weight": f"b{blk}.down",
            f"{p}.mlp.gate_proj.weight": f"b{blk}.gate",
        })
    wanted.update({
        "model.embed_tokens.weight": "embed",
        "lm_head.weight": "lm_head",
        "model.norm.weight": "final_norm",
    })
    tensors = load_tensors(wanted)

    for label, t in tensors:
        if label == "final_norm":
            continue  # 8 KB of near-noise; format handles it trivially
        print(f"--- {label}  {tuple(t.shape)}")
        data = bf16_bytes(t)
        lossless_suite(label, data, lossless)
        split_suite(label, t, splits)
        if label in ("b0.q", "embed", "lm_head"):  # representative lossy set
            rtn(t, 8, t.shape[-1], lossy, label)     # rowwise int8
            rtn(t, 8, 128, lossy, label)
            for g in (32, 64, 128):
                rtn(t, 4, g, lossy, label)

    # dict transfer on q_proj
    qs = {lbl: t for lbl, t in tensors}
    base, withdict = dict_suite(qs["b0.q"], qs["b15.q"], qs["b31.q"])

    # lossless-after-lossy: zstd3 on int8 group128 stream of b0.q
    x = qs["b0.q"].float().flatten().view(-1, 128)
    scale = (x.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-12)
    q8 = torch.round(x / scale).clamp(-128, 127).to(torch.int8).numpy().tobytes()
    c3 = zstd.ZstdCompressor(level=3, threads=16)
    q8_z = ratio(len(q8), len(c3.compress(q8)))

    # ---------------------------------------------------------------- report
    print("\n=== LOSSLESS (ratio = bf16 bytes / stored bytes) ===")
    keys = ["tensor", "MB", "zstd1", "zstd3", "zstd19", "zlib6", "lzma6", "lz4",
            "zstd19_MBps", "zstd3_dec_MBps"]
    print("  ".join(f"{k:>14}" for k in keys))
    for r in lossless:
        print("  ".join((f"{r[k]:>14}" if k == "tensor" else f"{r.get(k, float('nan')):>14.2f}")
                      for k in keys))

    print("\n=== BYTE-SPLIT (hi = sign+exp byte, lo = mantissa byte) ===")
    print(f"{'tensor':>14}  {'split+zstd19':>12}  {'hi ent b/B':>10}  {'lo ent b/B':>10}")
    for r in splits:
        print(f"{r['tensor']:>14}  {r['hi_zstd19']:>12.3f}  "
              f"{r['hi_ent_bits']:>10.2f}  {r['lo_ent_bits']:>10.2f}")

    print(f"\n=== DICT TRANSFER (q_proj): plain zstd19 {base:.3f}x, "
          f"dict-trained-on-other-blocks {withdict:.3f}x ===")

    print("\n=== LOSSY RTN (ratio | rel_err) ===")
    print(f"{'tensor':>10}  {'codec':>12}  {'ratio':>6}  {'rel_err':>8}")
    for r in lossy:
        print(f"{r['tensor']:>10}  {r['codec']:>12}  {r['ratio']:>6.2f}  "
              f"{r['rel_err']:>8.4f}")

    print(f"\n=== LOSSLESS-AFTER-LOSSY: int8g128 stream zstd3: {q8_z:.2f}x "
          f"(on top of the 1.97x int8 ratio) ===")


if __name__ == "__main__":
    main()
