# GRPO Trainer Performance Optimization Plan

## Problem
Generation takes **375.9s per step** (97% of total 387.1s step time). Custom `_sample()` with hierarchical pruning runs autoregressive token-by-token generation via `transformers.generate()`. 78.75% of completions hit the 8192 max length cap without terminating.

## Root Causes
1. **Excessive max_completion_length** — 78.75% of completions exhaust the 8192 budget (mean terminated length is only 2595 tokens)
2. **No truncation masking** — gradients computed even on truncated (useless) completions
3. **No torch.compile** — training forward/backward not compiled

## Constraints
- **vLLM cannot be used** — the model repo has `custom_generate/generate.py` loaded via `trust_remote_code=True`, implementing hierarchical pruning during generation. vLLM would bypass this entirely.
- The custom generation with KV cache pruning is critical for the hierarchical CoT training loop.

---

## Changes

### 1. Reduce max_completion_length to 4096

Already the default in `_grpo_worker.py` (line 29). No code change needed — just don't pass `--max-completion-length 8192` in the CLI.

Mean terminated completion length is only 2595 tokens, so 4096 is still generous. This roughly halves generation time since most completions were running to exhaustion at 8192.

### 2. Enable mask_truncated_completions

**File: `scripts/_grpo_worker.py`**

Add to GRPOConfig:
```python
mask_truncated_completions=True,
```

Completions that hit the max length cap get their loss contribution zeroed out. No wasted backward pass compute on low-quality truncated outputs.

### 3. Enable torch.compile for training

**File: `scripts/_grpo_worker.py`**

Add to GRPOConfig:
```python
torch_compile=True,
```

Compiles the training model's forward/backward passes. ~10-30% speedup on loss computation.

---

## Expected Impact

| Component | Before | After (estimated) |
|-----------|--------|-------------------|
| Generation | 375.9s | ~180-200s (halved max length) |
| Loss computation | 1.09s | ~0.8s (torch.compile) |
| Total step time | 387.1s | ~190-210s |
| Speedup | — | ~1.8-2x |

## Run Command (updated)
```bash
modal run --detach scripts/modal_grpo_train.py --num-gpus 4 --dataset-offset 5000 --dataset-limit 2000 --batch-size 1 --grad-accum-steps 4 --max-completion-length 4096 --use-lora --num-generations=8
```
