#!/usr/bin/env python3
"""Phase 3 correctness: layer-level check on real weights.

Loads layer 0 norm + q_proj from both checkpoints via direct safetensors read.
No AutoModelForCausalLM. 218 GB BF16 model does not fit 80 GB VRAM — this is
the correct approach: validate the transform on actual tensors.

Usage:
    python src/phase3_correctness.py \
        --orig  /workspace/gpt-oss-120b-BF16 \
        --fused /workspace/gpt-oss-120b-BF16-fused
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

N_PROMPTS = 50
HIDDEN    = 2880


def load_index(ckpt: Path) -> dict:
    return json.loads((ckpt / "model.safetensors.index.json").read_text())


def read_tensor(ckpt: Path, name: str, idx: dict) -> torch.Tensor:
    with safe_open(ckpt / idx["weight_map"][name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    return (x32 / (x32.pow(2).mean(-1, keepdim=True) + eps).sqrt() * w.float())


def kl_div(ref: torch.Tensor, got: torch.Tensor) -> float:
    p = F.softmax(ref.float(), dim=-1)
    q = F.softmax(got.float(), dim=-1)
    return F.kl_div(q.log(), p, reduction="sum").item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig",  default="/workspace/gpt-oss-120b-BF16")
    ap.add_argument("--fused", default="/workspace/gpt-oss-120b-BF16-fused")
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()

    orig, fused = Path(args.orig), Path(args.fused)
    oi, fi = load_index(orig), load_index(fused)

    eps = json.loads((orig / "config.json").read_text()).get("rms_norm_eps", 1e-5)
    print(f"rms_norm_eps={eps}  seed={args.seed}  n_prompts={N_PROMPTS}\n")

    L0, L17 = "model.layers.0", "model.layers.17"
    gamma      = read_tensor(orig,  f"{L0}.input_layernorm.weight",       oi)
    ones_l0    = read_tensor(fused, f"{L0}.input_layernorm.weight",       fi)
    ones_l17   = read_tensor(fused, f"{L17}.input_layernorm.weight",      fi)
    W_orig     = read_tensor(orig,  f"{L0}.self_attn.q_proj.weight",      oi)
    W_fused    = read_tensor(fused, f"{L0}.self_attn.q_proj.weight",      fi)
    bias       = read_tensor(orig,  f"{L0}.self_attn.q_proj.bias",        oi)
    post_orig  = read_tensor(orig,  f"{L0}.post_attention_layernorm.weight", oi)
    post_fused = read_tensor(fused, f"{L0}.post_attention_layernorm.weight", fi)

    # --- spot checks ---
    spot = [
        ("layer 0  input_layernorm == ones",       torch.all(ones_l0  == 1.0).item()),
        ("layer 17 input_layernorm == ones",       torch.all(ones_l17 == 1.0).item()),
        ("layer 0  post_attn_layernorm unchanged", torch.equal(post_orig, post_fused)),
    ]
    for label, ok in spot:
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
    print()

    # --- 50 random hidden states ---
    torch.manual_seed(args.seed)
    xs = torch.randn(N_PROMPTS, HIDDEN, dtype=torch.bfloat16)

    cos_sims, max_diffs, kls = [], [], []
    for x in xs:
        x = x.unsqueeze(0)
        ref = (rmsnorm(x, gamma,   eps) @ W_orig.float().T  + bias.float()).squeeze()
        got = (rmsnorm(x, ones_l0, eps) @ W_fused.float().T + bias.float()).squeeze()
        cos_sims.append(F.cosine_similarity(ref.unsqueeze(0), got.unsqueeze(0)).item())
        max_diffs.append((ref - got).abs().max().item())
        kls.append(kl_div(ref, got))

    mean_cs   = sum(cos_sims) / N_PROMPTS
    min_cs    = min(cos_sims)
    mean_diff = sum(max_diffs) / N_PROMPTS
    mean_kl   = sum(kls) / N_PROMPTS

    print(f"cos_sim   mean={mean_cs:.8f}  min={min_cs:.8f}")
    print(f"max|diff| mean={mean_diff:.6f}")
    print(f"KL div    mean={mean_kl:.2e}\n")

    gates = [
        ("mean cos_sim >= 0.999", mean_cs  >= 0.999),
        ("min  cos_sim >= 0.998", min_cs   >= 0.998),
        ("mean KL      <  1e-4",  mean_kl  <  1e-4),
    ]
    all_pass = all(ok for _, ok in gates) and all(ok for _, ok in spot)
    for label, ok in gates:
        print(f"{'PASS' if ok else 'FAIL'}  {label}")

    print(f"\n{'All Phase 3 gates PASS.' if all_pass else 'Phase 3 FAILED — do not proceed to Phase 4.'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
