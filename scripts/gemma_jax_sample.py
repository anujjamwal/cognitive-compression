"""Hand-run a prompt through the Gemma 4 JAX hierarchical-CoT sampler with
pruning on vs. off.

Not asserted — this is a human-readable smoke test.  Both modes must
produce coherent text.  In pruning-on mode, any `<return|>` events
should leave the visible output collapsed (thought + markers replaced
by the summary).

Note: pruning will only fire on models that actually emit
`<|channel> ... <channel|> summary <return|>` sequences.  The base
Gemma 4 IT checkpoint won't do this organically; this script's pruning
path becomes interesting after Phase 2 SFT.

Usage:
    python scripts/gemma_jax_sample.py \\
        --prompt "Solve: if x + 3 = 10, what is x?" \\
        --max-new-tokens 128

    python scripts/gemma_jax_sample.py --mode on  --prompt "..."
    python scripts/gemma_jax_sample.py --mode off --prompt "..."
"""
from __future__ import annotations

import argparse

import jax


DEFAULT_PROMPT = "Solve: if x + 3 = 10, what is x?"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="user message (default: a simple algebra question)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["both", "on", "off"],
        default="both",
        help="which sampler(s) to run (default: both)",
    )
    args = parser.parse_args()

    from gemma import gm
    from lib.gemma_jax import (
        PruningChatSampler,
        make_gemma4_tokenizer,
        resolve_marker_ids,
    )

    print("=== user prompt ===")
    print(args.prompt)

    print("\n=== loading model & weights ===")
    model = gm.nn.Gemma4_E4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E4B_IT)
    tok = make_gemma4_tokenizer()
    markers = resolve_marker_ids(tok)

    rng = jax.random.key(args.seed)

    if args.mode in ("both", "off"):
        print("\n=== mode: pruning OFF (stock equivalent) ===")
        off = PruningChatSampler(
            model=model,
            params=params,
            tokenizer=tok,
            markers=markers,
            sampling=gm.text.Greedy(),
            pruning_enabled=False,
        )
        text_off = off.chat(
            args.prompt, max_new_tokens=args.max_new_tokens, rng=rng,
        )
        print(text_off)

    if args.mode in ("both", "on"):
        print("\n=== mode: pruning ON ===")
        on = PruningChatSampler(
            model=model,
            params=params,
            tokenizer=tok,
            markers=markers,
            sampling=gm.text.Greedy(),
            pruning_enabled=True,
        )
        text_on = on.chat(
            args.prompt, max_new_tokens=args.max_new_tokens, rng=rng,
        )
        print(text_on)


if __name__ == "__main__":
    main()
