"""Run scripts/verify_gemma_prune.py on Modal with a single GPU.

Two tests:
  1. Token-init sanity: the new `<return|>` row in `embed_tokens` and
     `embed_tokens_per_layer` is the average of the seed-phrase rows.
  2. Prune-correctness parity: manual cache truncation + suffix re-process
     produces top-1-agreeing, high-cosine logits vs. a reference forward
     on the post-prune sequence.

Usage:
    modal run scripts/modal_verify_gemma_prune.py
    modal run scripts/modal_verify_gemma_prune.py --gpu A100-40GB --attn eager

Requires:
    modal secret create huggingface HF_TOKEN=hf_xxx   # for gated Gemma 4
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        # transformers ≥5.0 ships Gemma 4 (Gemma4ForCausalLM, Gemma4TextConfig)
        "transformers>=5.0",
        "sentencepiece",
        "accelerate",
        "huggingface_hub",
    )
    .add_local_dir("lib", remote_path="/root/lib")
    .add_local_file(
        "scripts/verify_gemma_prune.py",
        remote_path="/root/scripts/verify_gemma_prune.py",
    )
)

app = modal.App("gemma4-verify-prune", image=image)

SECRETS = [modal.Secret.from_name("huggingface")]


@app.function(
    gpu="A100-40GB",
    timeout=30 * 60,
    secrets=SECRETS,
)
def verify(
    model: str = "google/gemma-4-E4B-it",
    dtype: str = "bfloat16",
    attn: str = "eager",
):
    import subprocess

    cmd = [
        "python", "/root/scripts/verify_gemma_prune.py",
        "--model", model,
        "--device", "cuda",
        "--dtype", dtype,
        "--attn", attn,
    ]
    print("Launching:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"verify_gemma_prune.py failed (exit {result.returncode})")


@app.local_entrypoint()
def main(
    model: str = "google/gemma-4-E4B-it",
    gpu: str = "A100-40GB",
    dtype: str = "bfloat16",
    attn: str = "eager",
):
    # The `gpu` argument is hardcoded in the decorator above; accepted here
    # only for documentation parity with modal_grpo_train.py.
    del gpu
    verify.remote(model=model, dtype=dtype, attn=attn)
