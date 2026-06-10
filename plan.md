# GPT-OSS 120B Fusion — vLLM & SGLang Benchmarking on A100/H100

**Runara AI | Team 2 | Hemakshi Chitte | Drafted June 10, 2026**
Assignment (Ben, June 9): *"Fusion GPT-OSS 120B vLLM A100/H100"* and *"Fusion GPT-OSS 120B vLLM A100/H100 SG-LANG"* — two active workstreams, satisfying the two-assignment policy.

---

## 1. Objective

Apply Runara's RMSNorm + Linear fusion (offline gamma absorption, `W' = W × γ`) to GPT-OSS 120B, integrate it into two serving platforms, and empirically measure end-to-end serving value:

- **Workstream A:** Fused GPT-OSS 120B on **vLLM** vs. unfused GPT-OSS 120B on vLLM (control)
- **Workstream B:** Fused GPT-OSS 120B on **SGLang** vs. the same vLLM-native control

This directly implements the June 9 standup strategy: our optimization is the **treatment**, vLLM's native implementation is the **control**, and value must be proven against the industry-standard serving stack — not against a vanilla/unoptimized model.

---

## 2. Judgment Calls Made (in lieu of Phase 0 with Ben)

These interpretations are documented so they can be confirmed or corrected later. Each has a confidence level and a fallback.

| # | Question | Call made | Rationale | Confidence |
|---|----------|-----------|-----------|------------|
| 1 | What does "fusion" mean? | Our gamma-absorption RMSNorm+Linear fusion, ported to GPT-OSS, measured at the **serving level** | Standup: "inject our optimizations on top of existing serving platforms (vLLM) to demonstrate superior TCO." Matches the team's established fusion technique. | High |
| 2 | What is the baseline? | vLLM serving unfused `openai/gpt-oss-120b` | Explicit standup decision: vLLM = control benchmark | High |
| 3 | A100 or H100? | **H100 80GB primary**, A100 80GB fallback | 120B with MXFP4 MoE fits a single H100 natively; A100 (SM80) has no native FP4 and routes through Marlin dequant kernels — a different code path that muddies fused-vs-unfused attribution. Run A100 only if H100 unavailable, and never mix results across the two. | Medium-High |
| 4 | Which compute provider? | Check GCP access first (Ben mentioned "fusion work on GCP using GPUs" in standup), else Vast.ai H100 | Standup 00:50:51 | Medium |
| 5 | Fusion scope v1 | **BF16 attention path only** (norm → q/k/v) + final norm → lm_head. MoE expert fusion deferred. | MoE weights are MXFP4 — see §4.3, gamma cannot be absorbed into power-of-two block scales | High |
| 6 | Metrics | Standard team benchmarking-doc set: TTFT, TPS mean/P99, max concurrency, VRAM idle/load, KV-cache growth, runtime stability; contexts 1k / 8k / 32k / 128k | Aligns with the multi-team "Inference Frameworks Benchmarking" doc | High |

---

## 3. Model Recon — GPT-OSS 120B (expected facts, ALL to be verified in Phase 1)

Rule #1 from the V2-Lite→V4-Flash port: **never trust expected attribute names — inspect the live model first.**

Expected architecture (verify via `inspect_model.py` before writing any fusion code):

- MoE transformer, ~117B total / ~5.1B active params; **128 experts, 4 active per token, no shared expert**
- **36 layers**, hidden size 2880
- Attention: GQA — 64 query heads, 8 KV heads, head_dim 64; **alternating sliding-window (128) and full-attention layers**; learned **attention sinks** per head
- Norms: RMSNorm (`input_layernorm`, `post_attention_layernorm`, final `norm`) — confirm exact names
- **Precision split (critical):** MoE expert weights (`gate_up_proj`, `down_proj`) are **MXFP4** — fp4 e2m1 values packed 2-per-uint8 in `tensor.blocks`, with **power-of-two block scales over groups of 32 along the last dimension** in `tensor.scales`. Attention, embeddings, router, lm_head are **BF16**.
- Context: 131k; tokenizer o200k_harmony; outputs require the **harmony chat format** (correctness tests must use the chat template)
- Checkpoint: `openai/gpt-oss-120b` (~63GB) — fits one 80GB GPU

### inspect_model.py must dump:
1. Full named-parameter list with shapes and dtypes (safetensors header read — NO `AutoModelForCausalLM` full load for inspection)
2. Norm tensor names and which consumers each norm feeds
3. MXFP4 storage layout (blocks/scales shapes) for one expert tensor
4. Layer-type pattern (sliding vs full) across all 36 layers
5. config.json snapshot into `DECISIONS.md`

---

## 4. Fusion Design

### 4.1 Fusion points per layer (expected — verify)

| Norm | Consumers | Dtype | Status |
|------|-----------|-------|--------|
| `layers.{L}.input_layernorm` | `q_proj`, `k_proj`, `v_proj` | BF16 | **Fuse** — fan-out 1→3, same pattern as V4 Flash `attn_norm → wq_a/wkv` |
| `layers.{L}.post_attention_layernorm` | `mlp.router` + 128 experts' `gate_up_proj` | BF16 + **MXFP4** | **Blocked in v1** — see 4.3 |
| final `norm` | `lm_head` | BF16 | **Fuse** |

### 4.2 Gate counters (must match exactly, à la V4 Flash; stop and reconcile on any deviation)

| Counter | Expected |
|---------|----------|
| bf16_transformed | **109** (36 × 3 qkv + 1 lm_head) |
| norms_reset | **37** (36 input_layernorm + 1 final norm) |
| norms_untouched | **36** (all post_attention_layernorm) |
| mxfp4_transformed | **0** (v1) |

### 4.3 Why the MoE path is excluded in v1

γ multiplies along the **input dimension** of `W`. MXFP4 block scales are (a) shared across 32 consecutive input-dim elements and (b) restricted to powers of two — so a per-element γ cannot be absorbed into scales (worse than the V4 Flash FP8 case, where scales were at least full-precision). Absorbing into the fp4 values themselves changes their distribution and re-clips at the E2M1 boundary.

Additionally, the `post_attention_layernorm` feeds **both** the router (BF16) and all experts (MXFP4). A norm can only be reset to ones if **every** consumer absorbs γ — partial absorption silently corrupts outputs. So unless the experts are handled, this norm cannot be touched at all.

**v2 option (stretch goal, only if v1 lands early):** dequant → fuse → requant the expert tensors to MXFP4, with per-tensor requantization-error analysis (cos sim per expert before/after) before any benchmark. Treat as a separate decision gate with Itamar.

### 4.4 The serving-level reality check (most important technical risk)

A pure weight transform sets norm weights to ones — but vLLM/SGLang **still execute the RMSNorm kernel** (typically `fused_add_rms_norm`), including the now-redundant γ multiply. Unlike our layer-level TorchAO benchmark, end-to-end serving may show **~0 speedup from the weight transform alone.**

So the work is staged:

- **Stage A — Drop-in fused checkpoint.** Pure weight transform, zero serving-code changes. Measures: does the checkpoint behave identically (correctness) and is there any measurable delta (probably noise-level)? This is the honest null-hypothesis run and de-risks correctness early.
- **Stage B — Kernel injection ("treatment").** Patch the GPT-OSS model definition in vLLM (`model_executor/models/gpt_oss.py`) and SGLang equivalently: replace the input-norm with a **weight-free RMSNorm** (skips γ load + multiply) for fused checkpoints, gated behind a flag/env var. This is the "inject low-level optimizations on top of serving platforms" strategy from the standup, in its minimal form.
- **Expectation management:** our 1.03–1.12x V4 Flash numbers were *layer-level*. End-to-end serving dilutes per-layer norm savings heavily; a realistic outcome for Stage B is **0–3% TPS / TTFT improvement**, possibly within noise. A null result against the vLLM control is a *valid empirical finding* under the team's stated methodology — report it straight.

---

## 5. Phases

### Phase 1 — Provision + Recon (Day 1)
1. Confirm GCP access; else provision Vast.ai **1× H100 80GB** (have A100 80GB fallback config saved). Record $/hr.
2. Create repo `hemoxiChloride/gpt-oss-120b-fusion` (+ `runaraai` remote). Seed `CLAUDE.md`, `DECISIONS.md`, `plan.md` (this file), `src/test_log.md`.
3. Download `openai/gpt-oss-120b`; run `inspect_model.py`; record §3 findings in `DECISIONS.md`.
4. Smoke-test baseline: `vllm serve openai/gpt-oss-120b`, one harmony-format completion.
- **Exit gate:** inspection output reconciled against §3/§4 expectations. Any mismatch → update fusion map *before* coding.

### Phase 2 — Fusion transform (Day 2–3)
1. Port `fuse.py` from `deepseekv4flash-fusion-transform`: gamma absorption on BF16 q/k/v + lm_head, norm reset, gate counters, safetensors-direct I/O (no full model load).
2. Unit tests per fusion point: `norm(x) @ W.T == norm_unit(x) @ W'.T` on random inputs, per-layer, tolerance documented in `test_log.md`. Max ~200 lines per Claude Code turn, test after every function.
3. Produce fused checkpoint; upload `hchitte/gpt-oss-120b-fused` (private until validated).
- **Exit gate:** all gate counters exact; unit tests green.

### Phase 3 — Correctness validation (Day 3)
1. Fixed prompt set (~50 prompts, harmony format) → logits fused vs unfused via transformers/safetensors path: **cos sim, max|Δlogit|, KL divergence** (120B is tractable — do KL from day one this time, don't let it become a blocker like on V4 Flash).
2. Quick task sanity: small GSM8K/MMLU subset, fused vs unfused, scores within noise.
- **Exit gate:** cos sim ≥ 0.999, KL ≈ 0 (BF16-only fusion should be near-exact — much tighter than the quantized V4 Flash case). Miss → red status, debug before any benchmarking.

### Phase 4 — Workstream A: vLLM (Day 4–6)
1. **Control:** unfused on vLLM, full benchmark matrix (§6).
2. **Stage A:** fused checkpoint on stock vLLM, same matrix.
3. **Stage B:** vLLM patch (weight-free norm path), fused checkpoint, same matrix. Keep the patch as a small reviewable diff in the repo.
4. KV-cache growth via Prometheus `/metrics` (`vllm:kv_cache_usage_perc`), VRAM via `nvidia-smi` logging, stability = sustained 30-min run.
- **Exit gate:** 3 result sets (control / Stage A / Stage B), each with correctness spot-check on live server output.

### Phase 5 — Workstream B: SGLang (Day 6–8)
1. Same checkpoints, same hardware, same matrix on SGLang (`python -m sglang.launch_server` + `python -m sglang.bench_serving`; KV metrics via `sglang_token_usage`).
2. Stage B equivalent patch in SGLang's GPT-OSS model file.
3. This doubles as input to the team's "SGLang vs vLLM" evaluation (Spoorthi's task) — coordinate to avoid redundant runs.
- **Exit gate:** same 3 result sets on SGLang.

### Phase 6 — Report (Day 8–9)
1. Single report, V4-Flash format: per-scenario table (framework × fused/unfused × context length × concurrency → TTFT, TPS mean/P99, KV growth, VRAM), treatment-vs-control deltas with run-to-run variance so noise-level results are identifiable as such.
2. Fairness section up front (Anirud will ask): identical hardware, identical checkpoints across frameworks, kernel-path notes, Stage A vs Stage B attribution.
3. PDF + Slack summary to Ben/Itamar; push everything to `Runaraai`.

---

## 6. Benchmark Matrix (per team benchmarking doc)

| Dimension | Values |
|-----------|--------|
| Frameworks | vLLM (control + treatment), SGLang |
| Checkpoints | unfused, fused (Stage A), fused + patch (Stage B) |
| Context | 1k, 8k, 32k, 128k input |
| Concurrency | 1, 8, 32, 64 |
| Metrics | TTFT (ms), TPS mean, TPS P99, max concurrency, VRAM idle/load, KV-cache growth, runtime stability |
| Repeats | ≥3 runs per cell; report mean ± std |

Note: half the layers use a 128-token sliding window — long-context KV growth will look very different from DeepSeek MLA; call this out in the report rather than letting it surprise reviewers.

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Stage B shows ~0 end-to-end gain | **High** | Pre-framed as a valid empirical null result vs control; per-layer microbenchmark included to show where the savings go |
| "Fusion" meant something else (e.g., framework-level kernel fusion comparison) | Medium | Judgment table in §2; phases 1, 4-control, 5-control, and 6 are reusable under any interpretation — only Phases 2–3 and the treatment runs would change |
| H100 unavailable / only A100 | Medium | A100 fallback documented; never mix A100/H100 numbers; state SM80 Marlin-dequant path explicitly in report |
| MXFP4 scope creep (expert fusion) | Medium | Hard-scoped out of v1; v2 only behind a decision gate with Itamar |
| Cost burn | Low | Single-GPU H100; stop instance every session; checkpoints cached on HF |

## 8. Operating Rules (new team policy + house rules)

- Yellow/red status → **engineering Slack channel immediately**, not held for standup
- Send this plan.md to Ben via Slack (satisfies the "Submit Plan Documents" action item)
- Dev loop: Claude Code on Mac → git push → pull on GPU instance → test → log to `test_log.md`
- Direct safetensors I/O for any whole-model weight work; no `AutoModelForCausalLM` full loads for transforms
- Push to GitHub at every phase boundary; stop cloud instances when idle
