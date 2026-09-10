#!/usr/bin/env python3
"""Generate canonical reference fixtures for the alice engine tests.

Run from the repo root with the project venv:
    .venv/bin/python tests/python/dump_canonical.py
    .venv/bin/python tests/python/dump_canonical.py \
        --model /opt/models/deepseek-ai/deepseek-r1-distill-llama-8b \
        --out tests/canonical_8b --skip-battery

Outputs (committed to the repo):
  <out>/tokenizer_battery.json  — [{text, ids}] for tokenizer tests
  <out>/case_<i>.json           — case meta (prompt, ids, shape)
  <out>/case_<i>.teacher.f32    — float32 logits [seq_len, vocab]

The generation cases are TEACHER-FORCED: the reference token sequence
(prompt + greedy continuation) is replayed through the model in a single
forward pass and the logits at EVERY position are recorded. The C++ test
replays the same sequence and compares logit vectors only — no argmax
comparison, so reference near-ties are irrelevant by construction.

Determinism notes: greedy decoding, eager attention, CPU, fixed transformers
version. The logits are a bf16-compute reference; exact values are only
guaranteed with the same torch/transformers versions, so C++ tests must treat
them as within-tolerance references, not bit-exact ones.
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BOS = 128000
GEN_STEPS = 8

# --- tokenizer battery: edge cases for the pre-tokenizer + BPE --------------
BATTERY = [
    "Hello world",
    "Hello, world!",
    "don't",
    "I'M SURE WE'LL WIN",
    "'tis 'twas 'RE",
    "1234567890",                      # numbers split into <=3 digit chunks
    "3.14159",
    "  double  spaces",
    "trailing space ",
    " leading space",
    "hello\nworld",
    "a\n\nb",
    "line1\r\nline2\r",
    "tab\there",
    "newlines\n\n\nend",
    "The quick brown fox jumps over 13 lazy dogs.",
    "héllo wörld café",
    "你好世界",
    "hello 👋🌍",
    "def f(x):\n    return x * 2",
    " ",
    "  ",
    "\n",
    "...?!",
    "a\u00a0non-breaking",             # NBSP is whitespace
    "mixed    \n\t whitespace \n\n",
    "supercalifragilisticexpialidocious",
    "https://example.com/path?q=1&r=2",
    "1999-12-31T23:59:59",
    "x" * 100,
]

# --- generation cases: canonical greedy continuations ------------------------
PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a little robot who",
    "def fibonacci(n):",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/opt/models/Llama-3.2-1B-Instruct")
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(__file__), "..", "canonical"))
    ap.add_argument("--skip-battery", action="store_true",
                    help="skip the tokenizer battery (same-family tokenizer "
                         "already covered by the 1B fixtures)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    if not args.skip_battery:
        battery = [{"text": t, "ids": tok.encode(t, add_special_tokens=False)}
                   for t in BATTERY]
        with open(os.path.join(args.out, "tokenizer_battery.json"), "w") as f:
            json.dump(battery, f, ensure_ascii=False, indent=1)
        print(f"battery: {len(battery)} cases")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()

    for i, prompt in enumerate(PROMPTS):
        ids = [BOS] + tok.encode(prompt, add_special_tokens=False)
        with torch.no_grad():
            out = model.generate(
                torch.tensor([ids]), max_new_tokens=GEN_STEPS, do_sample=False,
                num_beams=1, pad_token_id=128004)
        generated = out[0].tolist()[len(ids):]

        # teacher-forced replay of the full reference sequence
        full = ids + generated
        with torch.no_grad():
            logits = model(torch.tensor([full])).logits[0].float().numpy()  # [seq, V]

        meta = {
            "prompt": prompt,
            "input_ids": ids,
            "generated_ids": generated,
            "generated_text": tok.decode(generated),
            "seq_len": int(logits.shape[0]),
            "vocab": int(logits.shape[1]),
            "dtype": "float32",
        }
        with open(os.path.join(args.out, f"case_{i}.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        logits.astype("<f4").tofile(os.path.join(args.out, f"case_{i}.teacher.f32"))
        print(f"case_{i}: {prompt!r} -> {meta['generated_text']!r}")


if __name__ == "__main__":
    main()
