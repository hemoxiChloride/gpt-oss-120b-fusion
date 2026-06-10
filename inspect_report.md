# Model inspection report — `openai/gpt-oss-120b`

## 1. Config vs plan.md expectations

| field | expected | actual | status |
|---|---|---|---|
| num_hidden_layers | 36 | 36 | OK  |
| num_local_experts | 128 | 128 | OK  |
| num_experts_per_tok | 4 | 4 | OK  |
| hidden_size | 2880 | 2880 | OK  |
| num_attention_heads | 64 | 64 | OK  |
| num_key_value_heads | 8 | 8 | OK  |
| head_dim | 64 | 64 | OK  |

### Layer attention pattern
- counts: `{'sliding_attention': 18, 'full_attention': 18}`
- pattern (first 8): `['sliding_attention', 'full_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'full_attention']`
- sliding_window: `128` (expected 128)

## 2. Tensor inventory — 687 tensors

- dtype histogram: `{'U8': 144, 'BF16': 543}`

### Per-layer tensor roles (deduped across layers)

| role | count | example shape | dtype |
|---|---|---|---|
| `lm_head.weight` | 1 | [201088, 2880] | BF16 |
| `model.embed_tokens.weight` | 1 | [201088, 2880] | BF16 |
| `model.layers.{L}.input_layernorm.weight` | 36 | [2880] | BF16 |
| `model.layers.{L}.mlp.experts.down_proj_bias` | 36 | [128, 2880] | BF16 |
| `model.layers.{L}.mlp.experts.down_proj_blocks` | 36 | [128, 2880, 90, 16] | U8 |
| `model.layers.{L}.mlp.experts.down_proj_scales` | 36 | [128, 2880, 90] | U8 |
| `model.layers.{L}.mlp.experts.gate_up_proj_bias` | 36 | [128, 5760] | BF16 |
| `model.layers.{L}.mlp.experts.gate_up_proj_blocks` | 36 | [128, 5760, 90, 16] | U8 |
| `model.layers.{L}.mlp.experts.gate_up_proj_scales` | 36 | [128, 5760, 90] | U8 |
| `model.layers.{L}.mlp.router.bias` | 36 | [128] | BF16 |
| `model.layers.{L}.mlp.router.weight` | 36 | [128, 2880] | BF16 |
| `model.layers.{L}.post_attention_layernorm.weight` | 36 | [2880] | BF16 |
| `model.layers.{L}.self_attn.k_proj.bias` | 36 | [512] | BF16 |
| `model.layers.{L}.self_attn.k_proj.weight` | 36 | [512, 2880] | BF16 |
| `model.layers.{L}.self_attn.o_proj.bias` | 36 | [2880] | BF16 |
| `model.layers.{L}.self_attn.o_proj.weight` | 36 | [2880, 4096] | BF16 |
| `model.layers.{L}.self_attn.q_proj.bias` | 36 | [4096] | BF16 |
| `model.layers.{L}.self_attn.q_proj.weight` | 36 | [4096, 2880] | BF16 |
| `model.layers.{L}.self_attn.sinks` | 36 | [64] | BF16 |
| `model.layers.{L}.self_attn.v_proj.bias` | 36 | [512] | BF16 |
| `model.layers.{L}.self_attn.v_proj.weight` | 36 | [512, 2880] | BF16 |
| `model.norm.weight` | 1 | [2880] | BF16 |

## 3. Norm tensors (fusion sources)

- `model.layers.{L}.input_layernorm.weight`
- `model.layers.{L}.post_attention_layernorm.weight`
- `model.norm.weight`

## 4. MXFP4-stored tensors (blocks/scales pairs — DO NOT fuse in v1)

- `model.layers.{L}.mlp.experts.down_proj_blocks`
- `model.layers.{L}.mlp.experts.down_proj_scales`
- `model.layers.{L}.mlp.experts.gate_up_proj_blocks`
- `model.layers.{L}.mlp.experts.gate_up_proj_scales`

## 5. Predicted v1 gate counters (recompute fuse.py against these)

| counter | predicted | plan.md expected |
|---|---|---|
| bf16_transformed | 109 | 109 |
| norms_reset | 37 | 37 |
| norms_untouched (post_attn) | 36 | 36 |
| mxfp4_transformed | 0 | 0 |
