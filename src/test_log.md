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
