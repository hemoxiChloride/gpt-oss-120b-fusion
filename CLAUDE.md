# CLAUDE.md — gpt-oss-120b-fusion

Context file for Claude Code sessions. Read fully before editing anything.

## Project

Apply Runara's RMSNorm + Linear fusion (offline gamma absorption, W' = W × γ)
to GPT-OSS 120B, benchmark fused vs unfused at the **serving level** on
**vLLM** (Workstream A) and **SGLang** (Workstream B).

**Checkpoints:**
- Transform + kernel benchmark: `unsloth/gpt-oss-120b-BF16` (~234 GB, fully BF16)
- Serving benchmark: `openai/gpt-oss-120b` (~63 GB, MXFP4 + BF16-path fused)

Detailed plans: `plan_vllm.md`, `plan_sglang.md` (supersede `plan.md` for
hardware, protocol, and milestone ordering).
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
3. **4-step experimental protocol (per plan_vllm.md / plan_sglang.md):**
   (1) vanilla baseline via HF transformers, (2) fused via HF transformers —
   steps 1–2 shared across both workstreams, run once. (3) engine on vanilla,
   (4) engine + algorithm (treatment). Do not conflate kernel-level and
   serving-level results.
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
- GPU work runs on Vast.ai cloud instances. **Run and report on both H100 80GB
  and A100 80GB.** Results are ALWAYS in separate tables — never mixed.
  Stop instances when idle.
- Yellow/red blockers → engineering Slack channel immediately (team policy,
  June 9 standup).

## Phase status (update as you go)

- [x] Phase 1: provision + `inspect_model.py` run + reconcile vs plan.md
- [x] Phase 2: `src/fuse.py` + unit tests + real transform + verify + upload
      Gate counters 109/37/36/0 exact; verify_fused.py 8/8 PASS (BF16 upcast).
      Uploaded → https://huggingface.co/hchitte/gpt-oss-120b-fused (public)
- [x] Phase 3: correctness — cos_sim mean=0.99999866 min=0.99999833, KL=3.36e-05. All gates PASS.
- [ ] Phase 4: kernel benchmark — Itamar script, raw CSV, H100 + A100
- [ ] Phase 5: vLLM 4-step benchmark matrix — H100 + A100 (separate tables)
- [ ] Phase 6: SGLang 4-step benchmark matrix — H100 + A100 (separate tables)
- [ ] Phase 7: report + Confluence + Slack to Ben/Itamar
