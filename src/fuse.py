#!/usr/bin/env python3
"""
Offline gamma-absorption transform for openai/gpt-oss-120b.

Streams one safetensors shard at a time; peak RAM ≈ one shard + 37 norm vectors.
Writes a new sharded checkpoint + index; copies config/tokenizer files unchanged.

Usage:
    python src/fuse.py --src /path/to/gpt-oss-120b --dst /path/to/gpt-oss-120b-fused
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------- name patterns

_INPUT_NORM = re.compile(r"model\.layers\.(\d+)\.input_layernorm\.weight")
_POST_NORM  = re.compile(r"model\.layers\.(\d+)\.post_attention_layernorm\.weight")
_QKV_WEIGHT = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.weight")
_FINAL_NORM = "model.norm.weight"
_LM_HEAD    = "lm_head.weight"

EXPECTED = dict(
    bf16_transformed=109,
    norms_reset=37,
    post_attn_norms_untouched=36,
    mxfp4_transformed=0,
)

# ---------------------------------------------------------------- core helpers


def absorb(W: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """W' = W * gamma, upcast to fp32, result cast back to BF16.

    W:     [out_features, in_features]
    gamma: [in_features]  — broadcasts along out_features automatically.
    Bias is unaffected (post-matmul); caller copies it unchanged.
    """
    return (W.to(torch.float32) * gamma).to(torch.bfloat16)


def load_norms(src: Path, weight_map: dict) -> dict:
    """Load all input_layernorm + final norm weights into memory (~200 KB).

    Norm tensors may live in different shards from the q/k/v weights they feed,
    so we pre-load them all before the main streaming pass.
    """
    norm_names = [n for n in weight_map
                  if _INPUT_NORM.match(n) or n == _FINAL_NORM]
    by_shard: dict = {}
    for name in norm_names:
        by_shard.setdefault(weight_map[name], []).append(name)

    norms: dict = {}
    for shard_file, names in by_shard.items():
        with safe_open(src / shard_file, framework="pt", device="cpu") as f:
            for n in names:
                norms[n] = f.get_tensor(n).to(torch.float32)
    return norms


def transform_shard(
    src_path: Path,
    norms: dict,
    counters: dict,
) -> dict:
    """Apply fusion transforms to one shard; return transformed tensor dict."""
    out: dict = {}
    with safe_open(src_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)

            if _INPUT_NORM.match(name):
                out[name] = torch.ones_like(t)
                counters["norms_reset"] += 1

            elif name == _FINAL_NORM:
                out[name] = torch.ones_like(t)
                counters["norms_reset"] += 1

            elif _POST_NORM.match(name):
                out[name] = t
                counters["post_attn_norms_untouched"] += 1

            elif m := _QKV_WEIGHT.match(name):
                L = m.group(1)
                gamma = norms[f"model.layers.{L}.input_layernorm.weight"]
                out[name] = absorb(t, gamma)
                counters["bf16_transformed"] += 1

            elif name == _LM_HEAD:
                out[name] = absorb(t, norms[_FINAL_NORM])
                counters["bf16_transformed"] += 1

            else:
                # Copy unchanged: biases, sinks, o_proj, post_attn norms,
                # MXFP4 blocks/scales, embeddings, router weights, etc.
                out[name] = t

    return out


def assert_counters(counters: dict) -> None:
    ok = True
    rows = []
    for key, exp in EXPECTED.items():
        got = counters.get(key, 0)
        mark = "OK  " if got == exp else "FAIL"
        if got != exp:
            ok = False
        rows.append(f"  {mark}  {key}: {got}  (expected {exp})")
    print("\n--- Gate counters ---")
    print("\n".join(rows))
    if not ok:
        print("\nFAIL: gate counter mismatch — aborting.", file=sys.stderr)
        sys.exit(1)
    print("\nAll gate counters match.\n")


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source checkpoint dir")
    ap.add_argument("--dst", required=True, help="Output checkpoint dir")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Hard assert: tie_word_embeddings must be false before touching lm_head
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("tie_word_embeddings", False):
        print(
            "ABORT: tie_word_embeddings=true — lm_head shares embedding weights; "
            "absorbing final norm would corrupt the embedding table.",
            file=sys.stderr,
        )
        return 1

    # Determine sharding layout
    index_path = src / "model.safetensors.index.json"
    multi_shard = index_path.exists()
    if multi_shard:
        index = json.loads(index_path.read_text())
        weight_map: dict = index["weight_map"]
        shards = sorted(set(weight_map.values()))
        norms = load_norms(src, weight_map)
    else:
        shards = ["model.safetensors"]
        norms = {}
        with safe_open(src / "model.safetensors", framework="pt", device="cpu") as f:
            for name in f.keys():
                if _INPUT_NORM.match(name) or name == _FINAL_NORM:
                    norms[name] = f.get_tensor(name).to(torch.float32)

    counters = {k: 0 for k in EXPECTED}
    new_weight_map: dict = {}

    # Stream and transform shards
    for shard_file in shards:
        print(f"  transforming {shard_file} ...", flush=True)
        result = transform_shard(src / shard_file, norms, counters)
        with safe_open(src / shard_file, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        save_file(result, dst / shard_file, metadata=meta)
        for name in result:
            new_weight_map[name] = shard_file

    # Write updated index
    if multi_shard:
        new_index = {"metadata": index.get("metadata", {}),
                     "weight_map": new_weight_map}
        (dst / "model.safetensors.index.json").write_text(
            json.dumps(new_index, indent=2))

    # Copy config, tokenizer, and all non-weight files unchanged
    skip_ext  = {".safetensors"}
    skip_name = {"model.safetensors.index.json"}
    for p in src.iterdir():
        if p.suffix in skip_ext or p.name in skip_name:
            continue
        if p.is_file():
            shutil.copy2(p, dst / p.name)
    print(f"Config/tokenizer files copied to {dst}")

    assert_counters(counters)
    print(f"Fused checkpoint written to: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
