# CLAUDE.md — gpt-oss-120b-fusion

Context file for Claude Code sessions. Read fully before editing anything.

## Project

Apply Runara's RMSNorm + Linear fusion (offline gamma absorption, W' = W × γ)
to `openai/gpt-oss-120b`, then benchmark fused vs unfused at the **serving
level** on **vLLM** (Workstream A) and **SGLang** (Workstream B).
**vLLM serving the unfused model is the control. Our fusion is the treatment.**

Full plan: `plan.md` (phases, gate counters, benchmark matrix, risks).
Architecture decisions log: `DECISIONS.md` — append, never rewrite history.

## Critical technical constraints

1. **MoE expert weights are MXFP4** (fp4 blocks + power-of-two scales over
   groups of 32 along the input dim). γ CANNOT be absorbed there. v1 fuses
   ONLY: `input_layernorm → q/k/v_proj` (BF16, per layer) and
   `final norm → lm_head`. `post_attention_layernorm` is NEVER touched — it
   feeds the router AND all 128 experts; partial absorption corrupts outputs.
2. **Gate counters (v1) must match exactly or STOP:**
   bf16_transformed=109, norms_reset=37, post_attn norms untouched=36,
   mxfp4_transformed=0. (Re-derive from `inspect_report.md` if the live model
   disagrees with plan.md.)
3. **A pure weight transform gives ~0 serving speedup** — vLLM/SGLang still run
   the RMSNorm kernel with weight=1. Stage A = drop-in checkpoint (correctness
   + null run). Stage B = patch the framework's gpt_oss model file with a
   weight-free norm path, flag-gated. Don't conflate the two in results.
4. **Never load the full model with AutoModelForCausalLM for weight
   transforms.** Direct safetensors I/O only. Inspection uses headers only
   (`inspect_model.py`).
5. Correctness gates before any benchmark numbers leave this repo:
   cos sim ≥ 0.999, KL ≈ 0 on a fixed 50-prompt harmony-format set.
   GPT-OSS requires the **harmony chat template** — raw prompts give garbage.

## House rules

- Max ~200 lines of new code per turn; write a test after every function;
  log every test run in `src/test_log.md`.
- Inspect live model attributes before patching anything (V2-Lite → V4-Flash
  lesson: names never carry over).
- Push to GitHub at every phase boundary. Remotes: `origin` =
  hemoxiChloride/gpt-oss-120b-fusion, `runaraai` = Runaraai org mirror.
- GPU work runs on the cloud instance (H100 80GB primary, A100 fallback —
  never mix results between the two). Stop instances when idle.
- Yellow/red blockers → engineering Slack channel immediately (team policy,
  June 9 standup).

## Phase status (update as you go)

- [x] Phase 1: provision + `inspect_model.py` run + reconcile vs plan.md
- [x] Phase 2: `src/fuse.py` port + unit tests + fused checkpoint upload
- [ ] Phase 3: correctness (cos sim / max|Δ| / KL + small eval)
- [ ] Phase 4: vLLM — control / Stage A / Stage B benchmark matrix
- [ ] Phase 5: SGLang — same matrix
- [ ] Phase 6: report (fairness section first), PDF + Slack to Ben/Itamar
