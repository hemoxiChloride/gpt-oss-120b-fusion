#!/usr/bin/env python3
"""Phase 1 recon: inspect openai/gpt-oss-120b WITHOUT downloading full weights.

Verifies every architectural assumption in plan.md §3/§4 before any fusion
code is written (house rule from the V2-Lite -> V4-Flash port: never trust
expected attribute names).

Uses only config.json + safetensors headers via huggingface_hub, so it runs
in seconds on a laptop — no GPU, no 63GB download.

Usage:
    pip install huggingface_hub
    python inspect_model.py [--repo openai/gpt-oss-120b] [--out inspect_report.md]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict

from huggingface_hub import get_safetensors_metadata, hf_hub_download

# ---------------------------------------------------------------- expectations
# From plan.md — every one of these must be confirmed or corrected here.
EXPECTED = {
    "num_hidden_layers": 36,
    "num_local_experts": 128,
    "num_experts_per_tok": 4,
    "hidden_size": 2880,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
}

NORM_PAT = re.compile(r"(layernorm|norm)", re.IGNORECASE)
LAYER_PAT = re.compile(r"layers\.(\d+)\.")


def load_config(repo: str) -> dict:
    path = hf_hub_download(repo_id=repo, filename="config.json")
    with open(path) as f:
        return json.load(f)


def check(name: str, actual, expected) -> str:
    mark = "OK " if actual == expected else ">>> MISMATCH"
    return f"| {name} | {expected} | {actual} | {mark} |"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="openai/gpt-oss-120b")
    ap.add_argument("--out", default="inspect_report.md")
    args = ap.parse_args()

    lines: list[str] = [f"# Model inspection report — `{args.repo}`\n"]

    # ---------------------------------------------------------- 1. config.json
    cfg = load_config(args.repo)
    lines.append("## 1. Config vs plan.md expectations\n")
    lines.append("| field | expected | actual | status |")
    lines.append("|---|---|---|---|")
    for key, exp in EXPECTED.items():
        lines.append(check(key, cfg.get(key), exp))

    layer_types = cfg.get("layer_types")
    lines.append("\n### Layer attention pattern")
    if layer_types:
        counts = Counter(layer_types)
        lines.append(f"- counts: `{dict(counts)}`")
        lines.append(f"- pattern (first 8): `{layer_types[:8]}`")
        sw = cfg.get("sliding_window")
        lines.append(f"- sliding_window: `{sw}` (expected 128)")
    else:
        lines.append(">>> `layer_types` not in config — inspect modeling code"
                     " before assuming alternating sliding/full attention.")

    # ------------------------------------------------- 2. safetensors headers
    print("Fetching safetensors metadata (headers only)...", file=sys.stderr)
    meta = get_safetensors_metadata(args.repo)
    # Build name -> (shape, dtype) from weight_map + files_metadata
    tensors: dict = {}
    for tname, fname in meta.weight_map.items():
        info = meta.files_metadata[fname].tensors[tname]
        tensors[tname] = (info.shape, info.dtype)

    lines.append(f"\n## 2. Tensor inventory — {len(tensors)} tensors\n")
    dtype_counts = Counter(dt for _, dt in tensors.values())
    lines.append(f"- dtype histogram: `{dict(dtype_counts)}`")

    # Group tensor name "roles" per layer (strip layer index for the pattern).
    roles = Counter(LAYER_PAT.sub("layers.{L}.", n) for n in tensors)
    lines.append("\n### Per-layer tensor roles (deduped across layers)\n")
    lines.append("| role | count | example shape | dtype |")
    lines.append("|---|---|---|---|")
    for role, cnt in sorted(roles.items()):
        example = next(n for n in tensors if LAYER_PAT.sub("layers.{L}.", n) == role)
        shape, dtype = tensors[example]
        lines.append(f"| `{role}` | {cnt} | {list(shape)} | {dtype} |")

    # ------------------------------------------------------ 3. norms and MXFP4
    norms = sorted({LAYER_PAT.sub("layers.{L}.", n)
                    for n in tensors if NORM_PAT.search(n)})
    lines.append("\n## 3. Norm tensors (fusion sources)\n")
    for n in norms:
        lines.append(f"- `{n}`")

    mxfp4 = sorted({LAYER_PAT.sub("layers.{L}.", n) for n in tensors
                    if n.endswith("_blocks") or n.endswith(".blocks")
                    or n.endswith("_scales") or n.endswith(".scales")})
    lines.append("\n## 4. MXFP4-stored tensors (blocks/scales pairs — DO NOT fuse in v1)\n")
    if mxfp4:
        for n in mxfp4:
            lines.append(f"- `{n}`")
    else:
        lines.append(">>> No blocks/scales pairs found — checkpoint layout differs"
                     " from plan.md §3. STOP and reconcile before fusing.")

    # --------------------------------------- 4. predicted gate counters (v1)
    n_layers = cfg.get("num_hidden_layers", 0)
    qkv_roles = [r for r in roles
                 if re.search(r"(q_proj|k_proj|v_proj)\.weight$", r)]
    lm_head_present = any(n == "lm_head.weight" for n in tensors)
    bf16_transformed = len(qkv_roles) * n_layers // max(len(qkv_roles), 1) * len(qkv_roles) \
        if qkv_roles else 0
    # simpler + safer: count actual matching tensors
    bf16_transformed = sum(1 for n in tensors
                           if re.search(r"(q_proj|k_proj|v_proj)\.weight$", n))
    bf16_transformed += 1 if lm_head_present else 0

    input_norms = sum(1 for n in tensors if "input_layernorm" in n)
    final_norm = sum(1 for n in tensors
                     if re.fullmatch(r"model\.norm\.weight", n) or n == "norm.weight")

    lines.append("\n## 5. Predicted v1 gate counters (recompute fuse.py against these)\n")
    lines.append("| counter | predicted | plan.md expected |")
    lines.append("|---|---|---|")
    lines.append(f"| bf16_transformed | {bf16_transformed} | 109 |")
    lines.append(f"| norms_reset | {input_norms + final_norm} | 37 |")
    lines.append(f"| norms_untouched (post_attn) | "
                 f"{sum(1 for n in tensors if 'post_attention_layernorm' in n)} | 36 |")
    lines.append(f"| mxfp4_transformed | 0 | 0 |")

    if not qkv_roles:
        lines.append("\n>>> No q_proj/k_proj/v_proj names found — attention may use a"
                     " fused qkv tensor or different naming. Fusion map in plan.md"
                     " §4.1 MUST be rewritten before coding fuse.py.")

    report = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"\nWritten to {args.out} — paste into DECISIONS.md and reconcile"
          f" any MISMATCH/>>> lines before Phase 2.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
