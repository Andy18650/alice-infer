#!/usr/bin/env python3
"""Generate an alice manifest for a llama-family HF model directory.

The manifest stays the TOTAL CONTRACT (design §3): explicit, one entry per
tensor, no expansion magic at load time. This script is just authoring
tooling — like tests/python/dump_canonical.py — its OUTPUT is what the
engine consumes. Sharding is physical packaging and deliberately NOT
represented: the loader resolves tensors across shards by probing.

Usage: gen_manifest.py <model_dir> <out.manifest.json>
"""
import json
import sys


def spec(name, shape):
    # no per-weight dtype (2026-09 manifest diet): derived from block type
    # (`*_bf16`) and enforced at the load boundary; no num_layers in config —
    # the block list is the layer composition
    return {"name": name, "shape": shape}


def main():
    model_dir, out_path = sys.argv[1], sys.argv[2]
    with open(f"{model_dir}/config.json") as f:
        c = json.load(f)

    d = c["hidden_size"]
    inter = c["intermediate_size"]
    L = c["num_hidden_layers"]
    nh, nkv = c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim", d // nh)
    vocab = c["vocab_size"]

    cfg = {
        "hidden_size": d,
        "intermediate_size": inter,
        "num_heads": nh,
        "num_kv_heads": nkv,
        "head_dim": hd,
        "vocab_size": vocab,
        "rms_norm_eps": c["rms_norm_eps"],
        "rope_theta": c["rope_theta"],
    }
    rs = c.get("rope_scaling")
    if rs:
        cfg["rope_scaling"] = {
            "type": "llama3",
            "factor": rs["factor"],
            "low_freq_factor": rs["low_freq_factor"],
            "high_freq_factor": rs["high_freq_factor"],
            "original_max_position_embeddings": rs["original_max_position_embeddings"],
        }
    cfg["bos_token_id"] = c.get("bos_token_id", 128000)
    eos = c.get("eos_token_id", 128001)
    cfg["eos_token_ids"] = eos if isinstance(eos, list) else [eos]

    blocks = []
    pre_norms = []
    for i in range(L):
        p = f"model.layers.{i}"
        blocks.append({
            "type": "gqa_attention_bf16",
            "weights": {
                "wq": spec(f"{p}.self_attn.q_proj.weight", [nh * hd, d]),
                "wk": spec(f"{p}.self_attn.k_proj.weight", [nkv * hd, d]),
                "wv": spec(f"{p}.self_attn.v_proj.weight", [nkv * hd, d]),
                "wo": spec(f"{p}.self_attn.o_proj.weight", [d, nh * hd]),
            },
        })
        blocks.append({
            "type": "dense_ffn_bf16",
            "weights": {
                "w_gate": spec(f"{p}.mlp.gate_proj.weight", [inter, d]),
                "w_up": spec(f"{p}.mlp.up_proj.weight", [inter, d]),
                "w_down": spec(f"{p}.mlp.down_proj.weight", [d, inter]),
            },
        })
        pre_norms.append(spec(f"{p}.input_layernorm.weight", [d]))
        pre_norms.append(spec(f"{p}.post_attention_layernorm.weight", [d]))

    if c.get("tie_word_embeddings", False):
        # tied embeddings: name the EMBEDDER's tensor in the lm_head role —
        # the one legal shared claim (validate() exempts that pair)
        lm_head = spec("model.embed_tokens.weight", [vocab, d])
    else:
        lm_head = spec("lm_head.weight", [vocab, d])

    manifest = {
        "model": c.get("_name_or_path", model_dir).rstrip("/").split("/")[-1],
        "config": cfg,
        "embedder": {"token_embedding": spec("model.embed_tokens.weight", [vocab, d])},
        "trunk": {
            "type": "rmsnorm_add",
            "pre_norms": pre_norms,
            "final_norm": spec("model.norm.weight", [d]),
        },
        "blocks": blocks,
        "lm_head": {"weight": lm_head},
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"wrote {out_path}: {L} layers, {len(blocks)} blocks, "
          f"lm_head {'tied' if c.get('tie_word_embeddings', False) else 'untied'}")


if __name__ == "__main__":
    main()
