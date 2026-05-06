"""Modal wrapper for the JAX Phase-1 verify harness.

Mirrors `modal_grpo_train.py`'s structure but builds a JAX/CUDA + DeepMind
`gemma` image instead of torch+TRL. Runs `scripts/gemma_jax_verify.py` on
a single A100 and prints the per-test diagnostics straight to Modal logs.

Usage:
    # All three tests on Gemma 4 E4B-IT (default):
    modal run scripts/modal_verify_gemma_jax.py

    # Single test:
    modal run scripts/modal_verify_gemma_jax.py --test sampler_noop

    # Faster iteration on the smaller 2B model:
    modal run scripts/modal_verify_gemma_jax.py --model E2B

GPU: A100 (40 GB) — comfortably fits Gemma 4 E4B (~20 GB params + cache).
If a different GPU is needed, edit the `gpu="A100"` literal below — the
factory pattern was removed because Modal 1.2.6 rejects closure-defined
@app.function targets unless the local and remote Python versions match,
which this repo's setup doesn't guarantee.

Prereqs:
    pip install modal  &&  modal token new

GCS access for `gs://gemma-data/...` works anonymously inside Modal containers
— no gcloud auth or HF mirror is needed.
"""
from __future__ import annotations

import modal


# Python 3.12 is required by the DeepMind gemma library.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]",
        "flax",
        "optax",
        "einops",
        "kauldron",
        "sentencepiece",
        "tokenizers",
        "datasets",
        # gemma's `_tokenizer.py` references `dialog.Format`, which was added
        # to google-deepmind/dialog after the v1.0.0 PyPI release. Install
        # from main to pick it up.
        "dialog @ git+https://github.com/google-deepmind/dialog",
        "gemma @ git+https://github.com/google-deepmind/gemma",
    )
    .add_local_dir("lib", remote_path="/root/lib")
    .add_local_file(
        "scripts/gemma_jax_verify.py",
        remote_path="/root/scripts/gemma_jax_verify.py",
    )
    .add_local_file(
        "scripts/_gemma_jax_verify_runner.py",
        remote_path="/root/scripts/_gemma_jax_verify_runner.py",
    )
)

app = modal.App("hcot-gemma-jax-verify", image=image)


@app.function(
    # A100-80GB: E4B (~20 GB params) plus JIT workspace + activations doesn't
    # fit cleanly on a 40 GB card; 80 GB has ample headroom for both E2B and E4B.
    gpu="A100-80GB",
    timeout=30 * 60,  # 30 min — first JIT + weight load dominates
)
def verify(test: str = "all", model: str = "E4B"):
    import os
    import subprocess
    import sys

    os.chdir("/root")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    env.setdefault("JAX_PLATFORMS", "cuda")
    # Suppress GCE metadata-service auth probes — Modal containers don't
    # have access to metadata.google.internal, and each probe blocks for
    # 60s before timing out. Disabling the probe forces immediate fall-
    # through to anonymous GCS reads (the public Gemma bucket allows
    # anonymous gets after EULA acceptance).
    env["GOOGLE_AUTH_DISABLE_GCE_METADATA_LOOKUP"] = "1"
    env["NO_GCE_CHECK"] = "true"
    env["GCE_METADATA_HOST"] = "disabled"
    # Don't pre-allocate the full GPU for JAX — orbax checkpoint restore
    # needs room to materialize the params before the runtime takes over,
    # and the default preallocation can crowd it out.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    cmd = [
        sys.executable,
        "/root/scripts/_gemma_jax_verify_runner.py",
        "--test", test,
        "--model", model,
    ]

    print(f"[modal] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"verify exited {result.returncode} for --test {test} --model {model}"
        )
    print("[modal] verify exited 0", flush=True)


@app.local_entrypoint()
def main(test: str = "all", model: str = "E4B"):
    if model not in ("E2B", "E4B"):
        raise SystemExit(f"unknown --model {model!r}; choose from ('E2B', 'E4B')")
    if test not in ("token_init", "sampler_noop", "prune_parity", "all"):
        raise SystemExit(
            f"unknown --test {test!r}; choose from "
            "('token_init', 'sampler_noop', 'prune_parity', 'all')"
        )
    verify.remote(test=test, model=model)
