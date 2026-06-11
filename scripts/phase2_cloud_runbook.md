# Phase 2 cloud runbook — real transform on the instance

CPU + disk only (no GPU needed for the transform). Run steps in order;
every gate must pass before the next step. Any deviation: **stop, report,
delete nothing.**

## 0. Preconditions — do these BEFORE the 63 GB download

```bash
# 0a. HF token with WRITE access (needed for the final upload — set up first
#     so the run doesn't stall at the last step)
export HF_TOKEN=<token-with-write-access>
# or: hf auth login

# 0b. Pull the repo
git clone git@github.com:hemoxiChloride/gpt-oss-120b-fusion.git
cd gpt-oss-120b-fusion
pip install -r requirements.txt
pip install hf_transfer

# 0c. Unit tests must be green on the box before touching the real model
python -m pytest tests/unit/test_fuse.py -v
# GATE: 7/7 pass

# 0d. Disk gate
df -h .
# GATE: >= 150 GB free (63 GB input + 63 GB output + headroom)
```

## 1. Download original checkpoint (hf_transfer enabled)

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download openai/gpt-oss-120b --local-dir ./gpt-oss-120b
# Do NOT load with transformers at any point — safetensors direct I/O only.
```

## 2. Run the transform — capture the full gate-counter table

```bash
python src/fuse.py --src ./gpt-oss-120b --dst ./gpt-oss-120b-fused \
    2>&1 | tee results/fuse_run.log
```

**GATE: counter table must print exactly**

```
bf16_transformed: 109
norms_reset: 37
post_attn_norms_untouched: 36
mxfp4_transformed: 0
```

Any deviation → fuse.py exits nonzero. Stop, report the table, delete nothing.

## 3. Spot verification against both checkpoints

```bash
python src/verify_fused.py --orig ./gpt-oss-120b --fused ./gpt-oss-120b-fused \
    2>&1 | tee results/verify_fused.log
# GATE: 9/9 checks PASS, exit 0
```

## 4. Upload — ONLY on full pass of steps 2 and 3

```bash
hf repo create hchitte/gpt-oss-120b-fused --private   # idempotent if exists
hf upload hchitte/gpt-oss-120b-fused ./gpt-oss-120b-fused . --private
# Includes config.json, tokenizer files, chat template (fuse.py copies them
# into the output dir automatically).
```

## 5. Record + push

```bash
# Append [PASS] entries for fuse run + verify run to src/test_log.md
# Append gate-counter table + verify output to DECISIONS.md Phase 2 entry
# Tick Phase 2 box in CLAUDE.md
git add -A
git commit -m "Phase 2 complete: real transform 109/37/36/0, verified, uploaded private"
git push origin main
```

## 6. STOP

Phase 3 (logit correctness + KL) is a separate session, pending review of
the counter table and verify output. Stop the instance if no further work
is queued (idle-instance house rule).
