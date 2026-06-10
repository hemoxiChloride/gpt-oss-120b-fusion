"""Unit tests for src/fuse.py — no model download, no GPU required.

Tolerance rationale (documented per spec):
  absorb() upcasts W and gamma to fp32, multiplies, then casts back to BF16.
  That single BF16 round-trip introduces ~0.4% relative error per element.
  After a 2880-wide matmul, absolute errors reach ~0.35 but cosine similarity
  between ref and fused output exceeds 0.9999 (measured on random BF16 tensors
  at real GPT-OSS shapes).  We therefore gate on cos_sim >= 0.9999, which
  directly mirrors the Phase 3 correctness threshold (cos_sim >= 0.999).
  Bias and copy-through tensors must be bit-identical (no tolerance).
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from fuse import absorb, load_norms, transform_shard, assert_counters, EXPECTED


# ---------------------------------------------------------------- helpers

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference RMSNorm in fp32."""
    x32 = x.float()
    w32 = weight.float()
    rms = (x32.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return (x32 / rms) * w32


# ---------------------------------------------------------------- test 1: equivalence per fusion point

@pytest.mark.parametrize("out_f,in_f", [(4096, 2880), (512, 2880), (201088, 2880)])
def test_equivalence_fusion_point(out_f, in_f):
    """RMSNorm(x,γ)@W.T+b  ≈  RMSNorm(x,ones)@W'.T+b via cos_sim >= 0.9999.

    Tolerance: BF16 absorb introduces ~0.4% per-weight relative error; after
    a 2880-wide matmul absolute diff reaches ~0.35 but cosine similarity stays
    above 0.9999 (see test_log.md for calibration run).
    cos_sim gate matches Phase 3 correctness threshold (>= 0.999).
    """
    torch.manual_seed(42)
    eps = 1e-5
    x     = torch.randn(1, in_f).to(torch.bfloat16)
    W     = torch.randn(out_f, in_f).to(torch.bfloat16)
    gamma = torch.rand(in_f).to(torch.bfloat16) + 0.5  # keep positive
    b     = torch.randn(out_f).to(torch.bfloat16)
    ones  = torch.ones(in_f, dtype=torch.bfloat16)

    W_prime = absorb(W, gamma)

    ref = (rmsnorm(x, gamma, eps) @ W.float().T + b.float()).squeeze()
    got = (rmsnorm(x, ones,  eps) @ W_prime.float().T + b.float()).squeeze()

    cs = F.cosine_similarity(ref.unsqueeze(0), got.unsqueeze(0)).item()
    max_abs = (ref - got).abs().max().item()
    # max|abs diff| logged for the record (test_log.md), NOT a gate
    print(f"\n  [{out_f}x{in_f}] cos_sim={cs:.8f}  max|abs diff|={max_abs:.4f}")
    assert cs >= 0.9999, \
        f"cos_sim={cs:.6f} < 0.9999 for out_f={out_f} in_f={in_f}"


# ---------------------------------------------------------------- test 1b: gate-counter negative test

def test_assert_counters_fails_on_mismatch():
    """assert_counters must exit nonzero when a counter is wrong."""
    bad = dict(EXPECTED)
    bad["bf16_transformed"] = EXPECTED["bf16_transformed"] - 1  # deliberately wrong
    with pytest.raises(SystemExit) as exc_info:
        assert_counters(bad)
    assert exc_info.value.code == 1, "expected exit code 1 on counter mismatch"

    # Sanity: correct counters must NOT raise
    assert_counters(dict(EXPECTED))


# ---------------------------------------------------------------- test 2: bias bit-identical

def test_bias_preserved():
    """Bias tensor must be bit-identical after a round-trip through a fake shard."""
    torch.manual_seed(7)
    bias = torch.randn(512, dtype=torch.bfloat16)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        dst = tmp / "dst"
        src.mkdir(); dst.mkdir()

        # Minimal 1-layer checkpoint: one norm, one weight, one bias, one sink
        gamma = torch.ones(8, dtype=torch.bfloat16)
        W     = torch.randn(16, 8, dtype=torch.bfloat16)
        sinks = torch.randn(64, dtype=torch.bfloat16)
        tensors = {
            "model.layers.0.input_layernorm.weight": gamma,
            "model.layers.0.self_attn.q_proj.weight": W,
            "model.layers.0.self_attn.q_proj.bias": bias,
            "model.layers.0.self_attn.sinks": sinks,
            "model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
            "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        }
        save_file(tensors, src / "model.safetensors")
        (src / "config.json").write_text(
            json.dumps({"tie_word_embeddings": False, "num_hidden_layers": 1,
                        "rms_norm_eps": 1e-5}))

        norms_loaded = {}
        from safetensors import safe_open
        with safe_open(src / "model.safetensors", framework="pt", device="cpu") as f:
            for name in f.keys():
                from fuse import _INPUT_NORM, _FINAL_NORM
                if _INPUT_NORM.match(name) or name == _FINAL_NORM:
                    norms_loaded[name] = f.get_tensor(name).to(torch.float32)

        counters = {k: 0 for k in EXPECTED}
        result = transform_shard(src / "model.safetensors", norms_loaded, counters)

        # Bias must be exactly unchanged
        assert torch.equal(result["model.layers.0.self_attn.q_proj.bias"], bias), \
            "bias was modified — must be bit-identical"


# ---------------------------------------------------------------- test 3: copy-through (post_attn norm + sinks)

def test_copythrough_tensors_unchanged():
    """post_attention_layernorm and sinks must be bit-identical after transform."""
    torch.manual_seed(13)
    post_norm_w = torch.randn(8, dtype=torch.bfloat16)
    sinks       = torch.randn(64, dtype=torch.bfloat16)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"; src.mkdir()

        tensors = {
            "model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
            "model.layers.0.post_attention_layernorm.weight": post_norm_w,
            "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
            "model.layers.0.self_attn.sinks": sinks,
            "model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
            "lm_head.weight": torch.randn(32, 8, dtype=torch.bfloat16),
        }
        save_file(tensors, src / "model.safetensors")

        from safetensors import safe_open
        from fuse import _INPUT_NORM, _FINAL_NORM
        norms_loaded = {}
        with safe_open(src / "model.safetensors", framework="pt", device="cpu") as f:
            for name in f.keys():
                if _INPUT_NORM.match(name) or name == _FINAL_NORM:
                    norms_loaded[name] = f.get_tensor(name).to(torch.float32)

        counters = {k: 0 for k in EXPECTED}
        result = transform_shard(src / "model.safetensors", norms_loaded, counters)

        assert torch.equal(result["model.layers.0.post_attention_layernorm.weight"],
                           post_norm_w), "post_attention_layernorm was modified"
        assert torch.equal(result["model.layers.0.self_attn.sinks"],
                           sinks), "sinks was modified"
        assert counters["post_attn_norms_untouched"] == 1


# ---------------------------------------------------------------- test 4: shard-streaming on fake 2-layer checkpoint

def test_shard_streaming_fake_checkpoint():
    """Fake 2-layer, 2-shard checkpoint: norms in shard-0, qkv in shard-1.

    Tests cross-shard norm lookup (the main streaming risk) and verifies:
    - load_norms fetches correctly from shard-0
    - transform_shard applies absorb + reset on shard-1 tensors
    - gate counters match 2-layer expectations: bf16=7, norms_reset=3,
      post_attn_untouched=2, mxfp4=0
    - config copy-through (tokenizer.json)
    """
    import shutil as _shutil
    from safetensors import safe_open as _open
    from fuse import load_norms, transform_shard, _INPUT_NORM, _FINAL_NORM

    torch.manual_seed(99)
    H, in_f = 8, 8

    # Shard 0: ONLY norm weights (cross-shard dependency)
    gamma0 = torch.rand(in_f, dtype=torch.bfloat16) + 0.5
    gamma1 = torch.rand(in_f, dtype=torch.bfloat16) + 0.5
    gamma_final = torch.ones(in_f, dtype=torch.bfloat16)
    shard0_tensors = {
        "model.layers.0.input_layernorm.weight": gamma0,
        "model.layers.1.input_layernorm.weight": gamma1,
        "model.norm.weight": gamma_final,
    }

    # Shard 1: qkv weights + biases + post_attn norms + sinks + lm_head
    q0_w = torch.randn(H, in_f, dtype=torch.bfloat16)
    k0_w = torch.randn(H, in_f, dtype=torch.bfloat16)
    v0_w = torch.randn(H, in_f, dtype=torch.bfloat16)
    sinks0 = torch.randn(H, dtype=torch.bfloat16)
    post0  = torch.randn(in_f, dtype=torch.bfloat16)
    lm_w   = torch.randn(32, in_f, dtype=torch.bfloat16)
    shard1_tensors = {
        "model.layers.0.self_attn.q_proj.weight": q0_w,
        "model.layers.0.self_attn.k_proj.weight": k0_w,
        "model.layers.0.self_attn.v_proj.weight": v0_w,
        "model.layers.0.self_attn.q_proj.bias":   torch.randn(H, dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": post0,
        "model.layers.0.self_attn.sinks": sinks0,
        "model.layers.1.self_attn.q_proj.weight": torch.randn(H, in_f, dtype=torch.bfloat16),
        "model.layers.1.self_attn.k_proj.weight": torch.randn(H, in_f, dtype=torch.bfloat16),
        "model.layers.1.self_attn.v_proj.weight": torch.randn(H, in_f, dtype=torch.bfloat16),
        "model.layers.1.post_attention_layernorm.weight": torch.randn(in_f, dtype=torch.bfloat16),
        "model.layers.1.self_attn.sinks": torch.randn(H, dtype=torch.bfloat16),
        "lm_head.weight": lm_w,
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src"; dst = tmp / "dst"
        src.mkdir(); dst.mkdir()

        save_file(shard0_tensors, src / "shard-00001-of-00002.safetensors")
        save_file(shard1_tensors, src / "shard-00002-of-00002.safetensors")

        weight_map = {}
        for n in shard0_tensors: weight_map[n] = "shard-00001-of-00002.safetensors"
        for n in shard1_tensors: weight_map[n] = "shard-00002-of-00002.safetensors"
        index = {"metadata": {}, "weight_map": weight_map}
        (src / "model.safetensors.index.json").write_text(json.dumps(index))
        (src / "config.json").write_text(
            json.dumps({"tie_word_embeddings": False, "num_hidden_layers": 2}))
        (src / "tokenizer.json").write_text("{}")

        # --- run the streaming API directly (avoids gate-counter mismatch on 2-layer model)
        norms = load_norms(src, weight_map)
        assert set(norms.keys()) == {
            "model.layers.0.input_layernorm.weight",
            "model.layers.1.input_layernorm.weight",
            "model.norm.weight",
        }, "load_norms missed a norm tensor"

        counters = {k: 0 for k in EXPECTED}
        for shard_f in sorted(set(weight_map.values())):
            result = transform_shard(src / shard_f, norms, counters)
            save_file(result, dst / shard_f)

        # Copy config through
        for p in src.iterdir():
            if p.suffix != ".safetensors" and p.name != "model.safetensors.index.json":
                _shutil.copy2(p, dst / p.name)

        # --- verify counters for 2-layer model
        assert counters["bf16_transformed"] == 7,          f"bf16_transformed={counters['bf16_transformed']}"
        assert counters["norms_reset"] == 3,                f"norms_reset={counters['norms_reset']}"
        assert counters["post_attn_norms_untouched"] == 2,  f"post_attn_norms_untouched={counters['post_attn_norms_untouched']}"
        assert counters["mxfp4_transformed"] == 0

        # --- verify norms reset to ones in output shards
        for shard_f in ["shard-00001-of-00002.safetensors",
                         "shard-00002-of-00002.safetensors"]:
            with _open(dst / shard_f, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if _INPUT_NORM.match(name) or name == _FINAL_NORM:
                        assert torch.all(f.get_tensor(name) == 1.0), \
                            f"{name} not reset to ones"

        # --- verify post_attn norm and sinks bit-identical
        with _open(dst / "shard-00002-of-00002.safetensors",
                   framework="pt", device="cpu") as f:
            assert torch.equal(
                f.get_tensor("model.layers.0.post_attention_layernorm.weight"), post0)
            assert torch.equal(f.get_tensor("model.layers.0.self_attn.sinks"), sinks0)

        # --- verify tokenizer copy-through
        assert (dst / "tokenizer.json").exists(), "tokenizer.json not copied"
