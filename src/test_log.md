# test_log.md — gpt-oss-120b-fusion

Format: `[PASS/FAIL] <function_name> — <timestamp> — <notes>`

---

<!-- append entries below this line -->
[PASS] absorb — 2026-06-10T05:08:24Z — smoke test; max|ref-got|=0.006 on random BF16 W[16,8], gamma[8], x[8]; tolerance 0.02 (BF16 round-trip error)
[PASS] test_equivalence_fusion_point[4096-2880]  — 2026-06-10T05:30:00Z — cos_sim=0.99999875; gate cos_sim>=0.9999 (abs max|diff|~0.36 from BF16 absorb, not element-wise gated)
[PASS] test_equivalence_fusion_point[512-2880]   — 2026-06-10T05:30:00Z — cos_sim=0.99999851
[PASS] test_equivalence_fusion_point[201088-2880]— 2026-06-10T05:30:00Z — cos_sim≈1.0
[PASS] test_bias_preserved                       — 2026-06-10T05:30:00Z — bit-identical bias after round-trip through transform_shard
[PASS] test_copythrough_tensors_unchanged        — 2026-06-10T05:30:00Z — post_attention_layernorm and sinks bit-identical; post_attn_norms_untouched counter=1
[PASS] test_shard_streaming_fake_checkpoint      — 2026-06-10T05:30:00Z — 2-layer cross-shard: counters bf16=7/norms_reset=3/post_attn=2/mxfp4=0; norms reset to ones; tokenizer.json copied through
[PASS] test_assert_counters_fails_on_mismatch    — 2026-06-10T05:55:00Z — deliberately wrong bf16_transformed=108 → SystemExit(1); correct counters → no raise
[PASS] test_equivalence_fusion_point (rerun w/ max|abs diff| logging) — 2026-06-10T05:55:00Z — [4096x2880] cos_sim=0.99999875 max|abs diff|=0.3594; [512x2880] cos_sim=0.99999851 max|abs diff|=0.3744; [201088x2880] cos_sim=1.00000405 max|abs diff|=0.4249. max|abs diff| is recorded, NOT a gate — BF16 absorb error after 2880-wide matmul; gate remains cos_sim>=0.9999. Full suite 7/7 green.
[PASS] verify_fused.py (synthetic pair)          — 2026-06-10T06:05:00Z — 9/9 checks pass on synthetic orig/fused pair built via transform_shard
[PASS] verify_fused.py (negative)                — 2026-06-10T06:05:00Z — unfused dir passed as --fused → 4 FAILs, exit 1, as required
[BLOCKED] real 120B transform                    — 2026-06-10T06:06:00Z — disk gate FAILED on MacBook Air: 24 GB free, spec requires >=150 GB (63 in + 63 out + headroom). Download NOT started. Nothing deleted. Transform deferred — see DECISIONS.md.
