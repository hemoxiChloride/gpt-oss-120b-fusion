#!/usr/bin/env python3
"""Post-transform spot verification: original vs fused checkpoint.

Direct safetensors reads of layer 0 + final norm + lm_head from BOTH
checkpoints; never loads the model. Prints PASS/FAIL per check, exits
nonzero on any FAIL.

Handles two checkpoint types automatically:
  MXFP4 original (openai/gpt-oss-120b)   — checks blocks/scales bit-identical
  BF16 upcast    (unsloth/gpt-oss-120b-BF16) — checks BF16 expert weight bit-identical

Usage:
    python src/verify_fused.py --orig /path/orig --fused /path/fused
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def load_index(ckpt: Path) -> dict:
    return json.loads((ckpt / "model.safetensors.index.json").read_text())


def read_tensor(ckpt: Path, name: str, index: dict) -> torch.Tensor:
    shard = index["weight_map"][name]
    with safe_open(ckpt / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--fused", required=True)
    args = ap.parse_args()
    orig, fused = Path(args.orig), Path(args.fused)

    orig_idx  = load_index(orig)
    fused_idx = load_index(fused)
    wm = orig_idx["weight_map"]

    L0 = "model.layers.0"
    is_mxfp4 = f"{L0}.mlp.experts.gate_up_proj_blocks" in wm

    print(f"Checkpoint type: {'MXFP4 original' if is_mxfp4 else 'BF16 upcast'}\n")

    checks = [
        ("input_layernorm == ones in fused",
         f"{L0}.input_layernorm.weight",
         lambda o, f: torch.all(f == 1.0)),
        ("q_proj.weight differs from original",
         f"{L0}.self_attn.q_proj.weight",
         lambda o, f: not torch.equal(o, f)),
        ("q_proj.bias bit-identical",
         f"{L0}.self_attn.q_proj.bias",
         lambda o, f: torch.equal(o, f)),
        ("post_attention_layernorm bit-identical",
         f"{L0}.post_attention_layernorm.weight",
         lambda o, f: torch.equal(o, f)),
        ("sinks bit-identical",
         f"{L0}.self_attn.sinks",
         lambda o, f: torch.equal(o, f)),
    ]

    if is_mxfp4:
        # MXFP4 original: blocks/scales must survive unchanged
        checks += [
            ("expert gate_up_proj_blocks bit-identical",
             f"{L0}.mlp.experts.gate_up_proj_blocks",
             lambda o, f: torch.equal(o, f)),
            ("expert gate_up_proj_scales bit-identical",
             f"{L0}.mlp.experts.gate_up_proj_scales",
             lambda o, f: torch.equal(o, f)),
        ]
    else:
        # BF16 upcast: expert BF16 weight must be untouched (post_attn_layernorm excluded)
        checks += [
            ("expert gate_up_proj.weight bit-identical (BF16 upcast, post_attn excluded)",
             f"{L0}.mlp.experts.gate_up_proj",
             lambda o, f: torch.equal(o, f)),
        ]

    checks += [
        ("lm_head.weight differs from original",
         "lm_head.weight",
         lambda o, f: not torch.equal(o, f)),
        ("model.norm.weight == ones in fused",
         "model.norm.weight",
         lambda o, f: torch.all(f == 1.0)),
    ]

    failures = 0
    for label, name, pred in checks:
        try:
            o = read_tensor(orig,  name, orig_idx)
            f = read_tensor(fused, name, fused_idx)
            ok = bool(pred(o, f))
        except KeyError as e:
            print(f"FAIL  {label} — tensor {e} missing from index")
            failures += 1
            continue
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
        failures += 0 if ok else 1

    total = len(checks)
    print(f"\n{total - failures}/{total} checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
