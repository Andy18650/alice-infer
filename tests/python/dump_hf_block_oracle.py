#!/usr/bin/env python3
"""Generate single-layer oracle fixtures from HuggingFace for alice block tests.

Extracts a single transformer layer from a real HF model, runs its attention
and MLP submodules in eager mode on synthetic data, and exports weights +
reference outputs as raw binary. The C++ tests load these and compare against
alice::gpu_rocm blocks.

Usage:
    .venv/bin/python tests/python/dump_hf_block_oracle.py \
        --model /opt/models/Llama-3.2-1B-Instruct \
        --layer 5 --S 8 --out tests/hf_oracle

Weights are saved as raw uint16 little-endian (bfloat16 bit patterns).
Tensors are saved as raw float32 little-endian.
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def pattern(n, seed):
    """Deterministic exact-representable pattern, matching C++ test style."""
    return np.array(
        [float(int((i * 7 + seed * 13) % 17) - 8) / 8.0 for i in range(n)],
        dtype=np.float32,
    )


def save_f32(arr, path):
    """Save a numpy float32 array as raw little-endian bytes."""
    arr.astype("<f4").tofile(path)


def save_bf16_bits(tensor, path):
    """Save a torch bfloat16 tensor as raw uint16 little-endian bytes."""
    arr = tensor.detach().cpu().contiguous().view(torch.uint16).numpy()
    arr.astype("<u2").tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/opt/models/Llama-3.2-1B-Instruct")
    ap.add_argument("--layer", type=int, default=5,
                    help="which transformer layer to extract")
    ap.add_argument("--S", type=int, default=8,
                    help="sequence length for synthetic input")
    ap.add_argument("--out", default="tests/hf_oracle")
    args = ap.parse_args()

    # One subdirectory per sequence length so multiple S values can coexist.
    out_dir = os.path.join(args.out, f"S{args.S}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()

    cfg = model.config
    layer = model.model.layers[args.layer]
    d = cfg.hidden_size
    hd = d // cfg.num_attention_heads

    # --- synthetic input: one sequence, fp32 --------------------------------
    x_np = pattern(args.S * d, seed=42).reshape(args.S, d)
    # HF expects [batch, seq, hidden]
    x = torch.from_numpy(x_np.copy()).unsqueeze(0)  # [1, S, d]

    save_f32(x_np, os.path.join(out_dir, "input.f32"))

    meta = {
        "model": args.model,
        "layer": args.layer,
        "S": args.S,
        "hidden_size": d,
        "intermediate_size": cfg.intermediate_size,
        "num_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": hd,
        "num_layers": cfg.num_hidden_layers,
        "rope_theta": getattr(cfg, "rope_theta", 10000.0),
        "rms_norm_eps": getattr(cfg, "rms_norm_eps", 1e-5),
    }

    rope_scaling = getattr(cfg, "rope_scaling", None)
    if rope_scaling is not None:
        meta["has_rope_scaling"] = True
        # Normalise spelling: HF uses "rope_type" or "type"
        rtype = rope_scaling.get("rope_type", rope_scaling.get("type", ""))
        meta["rope_scaling"] = {
            "type": rtype,
            "factor": rope_scaling.get("factor", 1.0),
            "low_freq_factor": rope_scaling.get("low_freq_factor", 1.0),
            "high_freq_factor": rope_scaling.get("high_freq_factor", 1.0),
            "original_max_position_embeddings": rope_scaling.get(
                "original_max_position_embeddings", 8192
            ),
        }
    else:
        meta["has_rope_scaling"] = False

    # --- Attention ----------------------------------------------------------
    attn = layer.self_attn
    q_w = attn.q_proj.weight.data
    k_w = attn.k_proj.weight.data
    v_w = attn.v_proj.weight.data
    o_w = attn.o_proj.weight.data

    save_bf16_bits(q_w, os.path.join(out_dir, "attn_wq.bf16"))
    save_bf16_bits(k_w, os.path.join(out_dir, "attn_wk.bf16"))
    save_bf16_bits(v_w, os.path.join(out_dir, "attn_wv.bf16"))
    save_bf16_bits(o_w, os.path.join(out_dir, "attn_wo.bf16"))

    with torch.no_grad():
        x_bf16 = x.to(torch.bfloat16)
        position_ids = torch.arange(args.S, device=x_bf16.device).unsqueeze(0)
        cos_sin = model.model.rotary_emb(x_bf16, position_ids)

        # Causal mask: upper triangle (above diagonal) filled with bf16 min.
        # HF LlamaAttention does NOT apply causal masking automatically when
        # called standalone; the mask is supplied by LlamaDecoderLayer.
        mask_val = torch.finfo(torch.bfloat16).min
        causal_mask = torch.triu(
            torch.full((args.S, args.S), mask_val, dtype=torch.bfloat16, device=x_bf16.device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        attn_out, _ = attn(x_bf16, attention_mask=causal_mask,
                           position_embeddings=cos_sin)
        attn_out = attn_out.squeeze(0).float().numpy()  # [S, d]

    save_f32(attn_out, os.path.join(out_dir, "attn_out.f32"))

    meta["attn_wq_shape"] = list(q_w.shape)
    meta["attn_wk_shape"] = list(k_w.shape)
    meta["attn_wv_shape"] = list(v_w.shape)
    meta["attn_wo_shape"] = list(o_w.shape)

    # --- FFN ----------------------------------------------------------------
    mlp = layer.mlp
    gate_w = mlp.gate_proj.weight.data
    up_w = mlp.up_proj.weight.data
    down_w = mlp.down_proj.weight.data

    save_bf16_bits(gate_w, os.path.join(out_dir, "ffn_gate.bf16"))
    save_bf16_bits(up_w, os.path.join(out_dir, "ffn_up.bf16"))
    save_bf16_bits(down_w, os.path.join(out_dir, "ffn_down.bf16"))

    with torch.no_grad():
        ffn_out = mlp(x_bf16).squeeze(0).float().numpy()  # [S, d]

    save_f32(ffn_out, os.path.join(out_dir, "ffn_out.f32"))

    meta["ffn_gate_shape"] = list(gate_w.shape)
    meta["ffn_up_shape"] = list(up_w.shape)
    meta["ffn_down_shape"] = list(down_w.shape)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Layer {args.layer} oracle written to {out_dir}")
    print(f"  S={args.S}, hidden={d}, inter={cfg.intermediate_size}, "
          f"heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}")
    print(f"  attn out shape: {attn_out.shape}")
    print(f"  ffn  out shape: {ffn_out.shape}")


if __name__ == "__main__":
    main()
