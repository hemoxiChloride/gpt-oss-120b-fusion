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

## (next entry) — Phase 1 inspection results

Paste inspect_report.md summary here after running inspect_model.py.
Reconcile every MISMATCH/>>> line; update plan.md §3/§4 gate counters if the
live model disagrees with predictions.
