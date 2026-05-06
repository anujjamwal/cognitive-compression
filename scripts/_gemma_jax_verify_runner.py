"""Internal wrapper for gemma_jax_verify.py used by modal_verify_gemma_jax.py.

Adds two things on top of the canonical verify script:

1. `--model E2B|E4B` — pick the IT checkpoint size + matching model class.
   For E2B we monkey-patch BOTH `gm.nn.Gemma4_E4B -> gm.nn.Gemma4_E2B`
   AND `gm.ckpts.load_params -> ... -> GEMMA4_E2B_IT`, since the verify
   script hard-codes the E4B model class and checkpoint path.

2. Subprocess isolation per test when `--test all` is requested.
   Each gemma_jax_verify test independently calls `gm.ckpts.load_params`
   for the full checkpoint; running them in one process makes JAX hold
   the prior test's params alive on the GPU and OOMs on E4B's second
   load. Running each test in its own Python process gets fresh JAX
   device state without re-downloading the weights (orbax/tensorstore
   caches the GCS read on disk).

The canonical verify script is unmodified.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys


_TESTS_IN_ORDER = ("token_init", "sampler_noop", "prune_parity")


def _patch_for_e2b() -> None:
    import gemma.gm.ckpts as ckpts
    import gemma.gm.nn as nn
    from gemma.gm.ckpts._paths import CheckpointPath

    nn.Gemma4_E4B = nn.Gemma4_E2B

    original_load = ckpts.load_params
    e2b_path = CheckpointPath.GEMMA4_E2B_IT

    def patched(path, *a, **kw):
        return original_load(e2b_path, *a, **kw)

    ckpts.load_params = patched

    print(
        "[runner] E2B patch active: "
        f"gm.nn.Gemma4_E4B -> Gemma4_E2B, load_params -> {e2b_path!r}",
        flush=True,
    )


def _run_single_test(test: str) -> None:
    """Run one test in this process. Caller is responsible for ensuring
    we have fresh JAX device state (i.e. fresh subprocess for E4B)."""
    spec = importlib.util.spec_from_file_location(
        "gemma_jax_verify", "/root/scripts/gemma_jax_verify.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = ["gemma_jax_verify.py", "--test", test]
    mod.main()


def _run_all_via_subprocess(model: str) -> None:
    """Dispatch each test as its own subprocess so JAX device memory is
    released between tests. Returns nonzero exit on first failure."""
    failed = []
    for test in _TESTS_IN_ORDER:
        print(f"\n[runner] === subprocess: --test {test} --model {model} ===",
              flush=True)
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--test", test,
                "--model", model,
            ],
        )
        if result.returncode != 0:
            failed.append(test)
            break  # short-circuit — same behaviour as the canonical main()

    if failed:
        print(f"\n[runner] FAILED: {failed}", flush=True)
        sys.exit(1)
    print("\n[runner] ALL PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        choices=["token_init", "sampler_noop", "prune_parity", "all"],
        default="all",
    )
    parser.add_argument("--model", choices=["E2B", "E4B"], default="E4B")
    args = parser.parse_args()

    if args.test == "all":
        _run_all_via_subprocess(args.model)
        return

    # Single test — apply the E2B patch if needed, then run in-process.
    # token_init never loads weights or instantiates the model, so the
    # patch is unnecessary for it.
    if args.test != "token_init" and args.model == "E2B":
        _patch_for_e2b()

    _run_single_test(args.test)


if __name__ == "__main__":
    main()
