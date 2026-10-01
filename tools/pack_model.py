#!/usr/bin/env python3
"""Pack a Hugging Face llama-family model directory into the alice model
format (PROTOTYPE, fused-JSON shape, 2026-09 — NOT frozen). Chicken-and-egg
breaker for the loader: emits the descriptor + packed weights the C++ loader
consumes.

Fused file shape (one JSON object, parse-once; verify-before-parse is
accepted — it only guards against an exploitable nlohmann bug):
  {"Alice": {"format_version": 1},
   "model":  {<descriptor: blocks/members carry content-ID only>},
   "signature": null,                  # placeholder; signing not implemented
   "files":   [{id, file, offset, size, codec}, ...]}   # unsigned location table
then \\n\\0, then (none here — weights.bin is a sibling data file).

Identity is separated from location: a block/member record carries only its
content-ID (the BLAKE3 of its bytes); the `files` array maps IDs to
{file, offset, size, codec}. The ID doubles as the integrity check
(verified on fetch), and identical bytes dedup to one stored copy + one
location entry (content-addressing — the tying/dedup payoff).

Layouts MUST match the C++ registry (src/blocks/*_factory.cpp):
  dense_ffn_bf16 v1:    [w_gate (inter x io)] [w_up (inter x io)] [w_down (io x inter)]
  gqa_attention_bf16 v1: [wq (nh*hd x io)] [wk (nkv*hd x io)] [wv (nkv*hd x io)]
                         [wo (io x nh*hd)]
bf16 row-major, no padding, records 256-byte aligned inside weights.bin.

Output: <out>/model.model (fused JSON + \\n\\0) and <out>/weights.bin.
Members (tokenizer/chat_template/embedder/trunk/language_head) are packed
but NOT interpreted by the loader this round.

Usage: pack_model.py <hf_model_dir> <out_dir> [--producer X] [--packager Y]
                      [--license ID] [--name NAME]
"""
import argparse
import json
import os
import sys

import blake3
import torch
from safetensors import safe_open

ALIGN = 256


class Packer:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.weights_path = os.path.join(out_dir, "weights.bin")
        self.f = open(self.weights_path, "wb")
        self.offset = 0
        self.by_id = {}  # id -> location record (content-addressed dedup)

    def add(self, payload: bytes) -> dict:
        """Append bytes (256-aligned); return {id, file, offset, size, codec}.

        Content-addressed: identical bytes dedup to one stored copy and one
        location entry (the ID == hash, so equal content is equal identity)."""
        h = blake3.blake3(payload).hexdigest()
        if h in self.by_id:
            return self.by_id[h]
        pad = (-self.offset) % ALIGN
        if pad:
            self.f.write(b"\0" * pad)
            self.offset += pad
        rec = {"id": h, "file": "weights.bin", "offset": self.offset,
               "size": len(payload), "codec": "raw"}
        self.f.write(payload)
        self.offset += len(payload)
        self.by_id[h] = rec
        return rec

    def files(self):
        return list(self.by_id.values())

    def close(self):
        self.f.close()


def tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw bf16 bytes, row-major, contiguous."""
    assert t.dtype == torch.bfloat16, f"expected bf16, got {t.dtype}"
    return t.contiguous().view(torch.uint16).numpy().tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--producer", default="unknown")
    ap.add_argument("--packager", default=os.environ.get("USER", "unknown"))
    ap.add_argument("--license", default="LicenseRef-unknown")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = json.load(open(f"{args.model_dir}/config.json"))
    assert cfg["architectures"] == ["LlamaForCausalLM"], cfg["architectures"]

    d = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", d // nh)
    vocab = cfg["vocab_size"]
    name = args.name or os.path.basename(args.model_dir.rstrip("/"))

    shards = sorted(
        f"{args.model_dir}/{f}"
        for f in os.listdir(args.model_dir)
        if f.endswith(".safetensors")
    )
    tensors = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

    pk = Packer(args.out_dir)

    # --- members (packed; the loader parses them structurally this round) ----
    members = {}
    tok_path = f"{args.model_dir}/tokenizer.json"
    if os.path.exists(tok_path):
        members["tokenizer"] = {
            "type": "llama", "version": 1,
            "config": {"vocab_size": vocab, "bos_token": "<|begin_of_text|>",
                       "bos_token_id": 128000, "eos_token_id": [128001, 128008, 128009]},
            "id": pk.add(open(tok_path, "rb").read())["id"],
        }
    tc_path = f"{args.model_dir}/tokenizer_config.json"
    if os.path.exists(tc_path):
        tc = json.load(open(tc_path))
        if isinstance(tc.get("chat_template"), str):
            members["chat_template"] = {
                "type": "jinja", "version": 1, "config": {},
                "id": pk.add(tc["chat_template"].encode())["id"],
            }
    embed_rec = pk.add(tensor_bytes(tensors["model.embed_tokens.weight"]))
    members["embedder"] = {
        "type": "plain_embedder_bf16", "version": 1,
        "config": {"vocab_size": vocab, "embedding_dim": d},
        "id": embed_rec["id"],
    }
    # Trunk: all pre-norms (execution order) then the final norm, each [d].
    norm_bytes = b"".join(
        tensor_bytes(tensors[f"model.layers.{i}.{n}"])
        for i in range(L)
        for n in ("input_layernorm.weight", "post_attention_layernorm.weight")
    ) + tensor_bytes(tensors["model.norm.weight"])
    members["trunk"] = {
        "type": "rmsnorm_add_bf16", "version": 1,
        "config": {"hidden_dim": d, "epsilon": cfg["rms_norm_eps"]},
        "id": pk.add(norm_bytes)["id"],
    }
    if cfg.get("tie_word_embeddings", False):
        lm_rec = embed_rec  # tying: same bytes, same id (one location entry)
        tying = True
    else:
        lm_rec = pk.add(tensor_bytes(tensors["lm_head.weight"]))
        tying = False
    members["language_head"] = {
        "type": "plain_language_head_bf16", "version": 1,
        "config": {"tying": tying, "vocab_size": vocab, "hidden_dim": d},
        "id": lm_rec["id"],
    }

    # --- block specs + blocks --------------------------------------------------
    attn_cfg = {
        "io_dim": d, "qk_dim": hd, "v_dim": hd,
        "num_q_heads": nh, "num_kv_heads": nkv, "rope_theta": cfg["rope_theta"],
    }
    rs = cfg.get("rope_scaling")
    if rs:
        attn_cfg.update({
            "rope_scaling_type": rs.get("rope_type") or rs.get("type"),
            "rope_scaling_factor": rs["factor"],
            "rope_scaling_low_freq_factor": rs["low_freq_factor"],
            "rope_scaling_high_freq_factor": rs["high_freq_factor"],
            "original_max_position_embeddings": rs["original_max_position_embeddings"],
        })

    blocks = []
    for i in range(L):
        p = f"model.layers.{i}"
        blocks.append({"spec": "attention", "id": pk.add(b"".join(
            tensor_bytes(tensors[f"{p}.self_attn.{w}.weight"])
            for w in ("q_proj", "k_proj", "v_proj", "o_proj")))["id"]})
        blocks.append({"spec": "ffn", "id": pk.add(b"".join(
            tensor_bytes(tensors[f"{p}.mlp.{w}.weight"])
            for w in ("gate_proj", "up_proj", "down_proj")))["id"]})
    pk.close()

    descriptor = {
        "name": name,
        "version": 1,
        "producer": args.producer,
        "packager": args.packager,
        "license": args.license,
        "max_context": cfg["max_position_embeddings"],
        **members,
        "block_specs": {
            "ffn": {"type": "dense_ffn_bf16", "version": 1,
                    "config": {"io_dim": d, "intermediate_dim": inter}},
            "attention": {"type": "gqa_attention_bf16", "version": 1,
                          "config": attn_cfg},
        },
        "blocks": blocks,
    }

    # Fused envelope: one JSON object. `model` is what will be signed (its
    # canonical bytes); `files` is the unsigned location table; `signature`
    # is a placeholder. Compact single-line (sorted keys, no whitespace) so
    # head -n 1 | jq works; canonicalization proper belongs to the signing round.
    envelope = {"Alice": {"format_version": 1}, "model": descriptor,
                "signature": None, "files": pk.files()}
    text = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    with open(f"{args.out_dir}/model.model", "wb") as f:
        f.write(text.encode())
        f.write(b"\n\0")

    total = os.path.getsize(pk.weights_path)
    print(f"packed {name}: {L} layers, {len(blocks)} blocks, "
          f"{len(pk.files())} unique data records, "
          f"lm_head {'tied' if tying else 'untied'}, "
          f"weights.bin {total / 1e9:.2f} GB, "
          f"max_context {descriptor['max_context']}")


if __name__ == "__main__":
    sys.exit(main())
