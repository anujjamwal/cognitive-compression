# GRPO Trainer Performance Optimization Plan

## Problem
Generation takes **375.9s per step** (97% of total 387.1s step time). Using vanilla `transformers.generate()` with 8192 max tokens, 8 generations per prompt, and 78.75% of completions hitting the max length cap.

## Root Causes
1. **No vLLM** — autoregressive token-by-token generation on the training model
2. **Excessive max_completion_length** — 78.75% of completions exhaust the 8192 budget
3. **No truncation masking** — gradients computed even on truncated (useless) completions

## Key Finding
The custom `_sample()` pruning generation is **NOT being used** during GRPO training — no `modeling_*.py` exists in the repo, and no `custom_generate` is passed via `generation_kwargs`. This is actually correct behavior: GRPO needs full unpruned outputs for reward computation. `return_unpruned_output: True` in generation_kwargs is currently a no-op (standard generate doesn't know about it).

---

## Changes

### 1. Enable vLLM for generation (biggest impact ~10-30x speedup on generation)

**File: `scripts/_grpo_worker.py`**

Add vLLM configuration to GRPOConfig:
```python
training_args = GRPOConfig(
    ...
    # vLLM acceleration
    use_vllm=True,
    vllm_mode="colocated",
    vllm_gpu_memory_utilization=0.7,
    ...
)
```

Remove `return_unpruned_output` from generation_kwargs (not supported by vLLM, and was a no-op anyway):
```python
generation_kwargs={
    # removed: "processing_class" and "return_unpruned_output"
    # vLLM handles its own tokenization
},
```

**File: `scripts/modal_grpo_train.py`**

Add `vllm` to pip_install:
```python
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "trl",
        "peft",
        "accelerate",
        "sentencepiece",
        "math-verify",
        "wandb",
        "huggingface_hub",
        "vllm",           # <-- add
    )
    ...
)
```

### 2. Reduce max_completion_length to 4096

Already the default in `_grpo_worker.py` (line 29), so this is just a CLI change. No code change needed — just don't pass `--max-completion-length 8192` in the run command.

### 3. Enable mask_truncated_completions

**File: `scripts/_grpo_worker.py`**

Add to GRPOConfig:
```python
mask_truncated_completions=True,
```

This skips gradient computation for completions that hit the max length without terminating, avoiding wasted backward pass compute.

### 4. Enable torch.compile for training forward/backward passes

**File: `scripts/_grpo_worker.py`**

Add to GRPOConfig:
```python
torch_compile=True,
```

This compiles the training model's forward pass with `torch.compile`, giving ~10-30% speedup on the loss computation phase.

### 5. Clean up generation_kwargs

Remove the no-op kwargs that don't work with vLLM:
```python
generation_kwargs={},  # vLLM handles generation internally
```

---

## Expected Impact

| Component | Before | After (estimated) |
|-----------|--------|-------------------|
| Generation | 375.9s | ~15-30s (vLLM colocated) |
| Loss computation | 1.09s | ~0.8s (torch.compile) |
| Total step time | 387.1s | ~30-50s |
| Speedup | — | ~8-12x |

## Run Command (updated)
```bash
modal run --detach scripts/modal_grpo_train.py --num-gpus 4 --dataset-offset 5000 --dataset-limit 2000 --batch-size 1 --grad-accum-steps 4 --max-completion-length 4096 --use-lora --num-generations=8
```
