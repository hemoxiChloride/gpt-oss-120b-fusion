# DECISIONS.md — gpt-oss-120b-fusion

Append-only log of architectural discoveries and judgment calls. Never rewrite
history; newer entries supersede older ones explicitly.

---

## 2026-06-10 — Project scoping (pre-kickoff, Ben unavailable)

Judgment calls made in lieu of Phase 0 clarification — full table in plan.md §2:

1. "Fusion" interpreted as Runara's gamma-absorption RMSNorm+Linear fusion,
   measured at the serving level. Control = vLLM serving unfused
   `openai/gpt-oss-120b` (per June 9 standup decision: vLLM = control,
   our optimization = treatment).
2. Hardware: H100 80GB primary, A100 80GB fallback. Never mix results across
   the two (A100/SM80 routes MXFP4 through Marlin dequant — different kernel
   path). Check GCP access first (Ben mentioned GCP in standup), else Vast.ai.
3. v1 fusion scope: BF16 attention path only (input_layernorm → q/k/v) plus
   final norm → lm_head. MoE expert fusion (MXFP4) explicitly out of scope —
   gamma cannot be absorbed into power-of-two block scales, and
   post_attention_layernorm feeds router + all 128 experts so it cannot be
   partially reset.
4. Staged measurement: Stage A = drop-in fused checkpoint (expected ~null
   serving delta — frameworks still run the norm kernel). Stage B = flag-gated
   weight-free-norm patch in the framework's gpt_oss model file (the actual
   treatment).

**To confirm with Ben/Itamar at next sync:** interpretation #1, hardware
choice, whether layer-level microbenchmarks are wanted alongside serving
metrics.

---

## 2026-06-10 — Phase 1 inspection results (inspect_model.py on openai/gpt-oss-120b)

Run: `python3 inspect_model.py --repo openai/gpt-oss-120b --out inspect_report.md`
Full output saved in `inspect_report.md`. **Zero mismatches — no plan rewrites needed.**

### Config confirmed
- num_hidden_layers=36, num_local_experts=128, num_experts_per_tok=4, hidden_size=2880
- num_attention_heads=64, num_key_value_heads=8, head_dim=64
- layer_types: strictly alternating sliding_attention/full_attention (18 each), sliding_window=128

### Tensor naming confirmed (exact names for fuse.py)
- Input norm: `model.layers.{L}.input_layernorm.weight` [2880] BF16
- Post-attn norm: `model.layers.{L}.post_attention_layernorm.weight` [2880] BF16 — UNTOUCHED v1
- Final norm: `model.norm.weight` [2880] BF16
- Attention projections: `self_attn.q_proj.weight` [4096,2880], `self_attn.k_proj.weight` [512,2880], `self_attn.v_proj.weight` [512,2880] — all BF16, all have bias tensors
- lm_head: `lm_head.weight` [201088,2880] BF16

### MXFP4 layout confirmed
- `mlp.experts.gate_up_proj_blocks` [128,5760,90,16] U8
- `mlp.experts.gate_up_proj_scales` [128,5760,90] U8
- `mlp.experts.down_proj_blocks` [128,2880,90,16] U8
- `mlp.experts.down_proj_scales` [128,2880,90] U8
- Also present: `mlp.experts.gate_up_proj_bias` [128,5760] BF16 and `down_proj_bias` [128,2880] BF16 — expert biases exist but are NOT in scope for v1.

### Gate counters (exact match — plan.md §4.2 stands unchanged)
| counter | plan.md | actual |
|---|---|---|
| bf16_transformed | 109 | 109 |
| norms_reset | 37 | 37 |
| norms_untouched | 36 | 36 |
| mxfp4_transformed | 0 | 0 |

### New facts not in plan.md (no rewrites — additive only)
1. **q/k/v and o_proj all have bias tensors.** γ absorption formula: `W' = W × diag(γ)`, bias unaffected (bias is post-matmul, γ is pre-matmul). fuse.py must preserve biases exactly.
2. **Attention sinks:** `self_attn.sinks` [64] BF16, one per query head. fuse.py must not touch these.
3. **Router and expert biases** (`mlp.experts.gate_up_proj_bias`, `down_proj_bias`) exist. Not in scope for v1 — do not touch.
4. **dtype histogram:** 543 BF16, 144 U8 (all U8 are expert blocks/scales — confirms clean precision split).
5. **Total tensors:** 687 across 15 safetensors shards.

---

## 2026-06-10 — Phase 2: real transform BLOCKED on local disk; code+tests complete

State at block:
- `src/fuse.py` + 7/7 unit tests green (incl. gate-counter negative test,
  max|abs diff| logged alongside cos_sim — see test_log.md).
- `src/verify_fused.py` written and validated on a synthetic orig/fused pair
  (9/9 checks) plus a negative run (unfused-as-fused → 4 FAILs, exit 1).

Blocker: MacBook Air has **24 GB free**; the disk gate requires **≥150 GB**
(63 GB input + 63 GB output + headroom). The 63 GB download alone does not
fit. Download NOT started; nothing deleted; no upload performed.

Options (decision needed):
1. **Run the transform on the cloud GPU box** (H100 instance — already the
   plan-of-record dev loop "push → pull on GPU instance → test"; disk is
   cheap there, and Phase 3 logit comparison needs the box anyway). fuse.py
   is CPU-only, so this costs only disk + a few minutes of instance time.
2. External SSD on the Mac (≥150 GB), keep everything local.
3. Free ~130 GB on the Mac (unrealistic).

Recommendation: option 1 — fold the transform into the Phase 3 GPU session:
download original → fuse → verify_fused.py → upload `hchitte/gpt-oss-120b-fused`
(private) → logit correctness, all on the instance.
