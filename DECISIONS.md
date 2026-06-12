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

---

## 2026-06-10 — Phase 2 deviation: real transform moved to the cloud instance (Option 1 APPROVED)

Decision (Hemakshi): the real 120B transform moves from the local box to the
cloud instance because the local disk gate failed (~24 GB free vs ≥150 GB
required; logged as [BLOCKED] in src/test_log.md). Download, `src/fuse.py`,
`src/verify_fused.py`, and the HF upload all run on the instance.

- No GPU needed for the transform itself — CPU + disk only. Benchmarking is
  still a later phase.
- Runbook with exact commands: `scripts/phase2_cloud_runbook.md`.
- Gate unchanged: counters must print exactly 109/37/36/0 — any deviation:
  stop, report, delete nothing. Upload to `hchitte/gpt-oss-120b-fused`
  (PRIVATE) only on full pass of verify_fused.py.
- Precondition: HF token with WRITE access on the box (`hf auth login` or
  `HF_TOKEN`) BEFORE starting the 63 GB download, so the run doesn't stall
  at the upload step.

---

## 2026-06-10 — Plan update: BF16 upcast + both GPUs (plan_vllm.md / plan_sglang.md supersede plan.md)

Two changes introduced. Old entries above remain as historical record.

### Change 1 — Source checkpoint for the transform (supersedes 2026-06-10 blocker entry)

- **Old plan:** `openai/gpt-oss-120b` (MXFP4, ~63 GB) used as fuse.py input.
- **New plan:** `unsloth/gpt-oss-120b-BF16` (~234 GB, fully BF16 upcast) is the
  source for the weight transform and kernel benchmark. The serving benchmark
  uses `openai/gpt-oss-120b` ("MXFP4 + BF16-path fusion applied", ~63 GB,
  fits single H100).
- **Why:** full BF16 upcast gives complete fusion scope for the kernel
  microbenchmark (all expert weights are BF16, no MXFP4 copy-through needed).
  Serving still uses the compact MXFP4 checkpoint for realistic H100 fit.
- **fuse.py impact:** algorithm unchanged. When run against the BF16 upcast
  there are no `_blocks`/`_scales` tensors, so the MXFP4 copy-through branches
  are simply never hit. Gate counters remain 109/37/36/0.
- **New disk gate:** ≥600 GB on the Vast.ai instance (234 GB in + 234 GB out +
  headroom). Old gate was ≥150 GB — obsolete for the BF16 path.
- **Precondition on instance:** `hf_transfer` enabled for the 234 GB download
  (`HF_HUB_ENABLE_HF_TRANSFER=1`); HF token with READ access on `unsloth` and
  WRITE access on `hchitte`.

### Change 2 — Hardware: both H100 and A100, always separate

- **Old plan:** H100 primary, A100 fallback — run one, skip the other.
- **New plan:** run and report on **both** H100 80GB and A100 80GB (Vast.ai
  team subscription). Results in separate tables always — never mixed.
- **Why:** plan_vllm.md and plan_sglang.md both explicitly require both GPUs
  and note that A100/SM80 routes MXFP4 through Marlin dequant (different
  kernel path from H100/SM90 native FP4) — a distinct data point worth
  reporting.

### Change 3 — Experimental protocol: 4-step replaces Stage A/B

- **Old plan:** Stage A (drop-in checkpoint) / Stage B (weight-free norm patch).
- **New plan:** 4-step protocol:
  (1) vanilla baseline — unfused, HF transformers (ground truth, no engine)
  (2) algorithm on vanilla — fused, HF transformers (isolate algorithm delta)
  (3) engine on vanilla — unfused, vLLM or SGLang (engine baseline)
  (4) engine + algorithm — fused, vLLM or SGLang (treatment)
  Steps 1–2 shared across vLLM and SGLang — run once, reused.
- **Why:** cleaner attribution of algorithm vs engine deltas. Stage A ≈ step 2,
  Stage B ≈ step 4 — semantics preserved, framing clarified.

### Change 4 — Kernel benchmark added before serving

- Itamar's `benchmark_rmsnorm_linear_fusion.py` runs on both H100 and A100
  before the serving benchmark. Do not modify the script; report raw CSV.
  This replaces the layer-level TorchAO microbenchmark reference in the
  old plan.

### What did NOT change

Fusion scope, gate counters (109/37/36/0), fuse.py algorithm,
all existing unit tests, correctness gate (cos_sim ≥ 0.999, KL ≈ 0).

Note: verify_fused.py updated to auto-detect checkpoint type (BF16 upcast vs
MXFP4 original) — blocks/scales check replaced with BF16 expert weight
bit-identity check when running against the BF16 upcast. See commit be31886.

---

## 2026-06-12 — Phase 2 complete: real transform verified on Vast.ai H100 instance

### fuse.py run
- Source: `unsloth/gpt-oss-120b-BF16` (73 shards, ~234 GB) at `/workspace/gpt-oss-120b-BF16`
- Output: `/workspace/gpt-oss-120b-BF16-fused`
- Log: `/workspace/fuse_run.log`

Gate counter table (exact — no deviations):

| counter | expected | actual | status |
|---|---|---|---|
| bf16_transformed | 109 | 109 | OK |
| norms_reset | 37 | 37 | OK |
| post_attn_norms_untouched | 36 | 36 | OK |
| mxfp4_transformed | 0 | 0 | OK |

### verify_fused.py run (v2, BF16 upcast path, 8 checks)
```
Checkpoint type: BF16 upcast

PASS  input_layernorm == ones in fused
PASS  q_proj.weight differs from original
PASS  q_proj.bias bit-identical
PASS  post_attention_layernorm bit-identical
PASS  sinks bit-identical
PASS  expert gate_up_proj.weight bit-identical (BF16 upcast, post_attn excluded)
PASS  lm_head.weight differs from original
PASS  model.norm.weight == ones in fused

8/8 checks passed.
```

Next: upload fused checkpoint to `hchitte/gpt-oss-120b-fused` (PRIVATE), then
Phase 3 correctness validation (cos_sim / KL on 50 harmony-format prompts).
