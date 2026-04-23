"""Hand-run a prompt through the Gemma 4 JAX hierarchical-CoT sampler with
pruning on vs. off.

Not asserted — this is a human-readable smoke test.  Both modes must produce
coherent text; in pruning-on mode, any `<return|>` events should leave the
visible output collapsed (thought + markers replaced by the summary).

Usage:
    python scripts/gemma_jax_sample.py \\
        --prompt "Solve: 2 + 2 = ?" \\
        --max-new-tokens 128

    python scripts/gemma_jax_sample.py --preamble thinking --prompt "..."

With `--preamble thinking`, the prompt is wrapped in a template that
instructs the model to emit its reasoning between `<|channel>` and
`<channel|>` before giving the final answer.  This is what lets the
hierarchical sampler exercise its prune path on an untrained base model.
"""
from __future__ import annotations

import argparse
import textwrap

import jax


DEFAULT_PROMPT = "Solve: if x + 3 = 10, what is x?"

THINKING_PREAMBLE = textwrap.dedent(
    """\
    <start_of_turn>user
    {user}
    <end_of_turn>
    <start_of_turn>model
    <|channel>I will think about this step by step.<channel|>
    """
)

PLAIN_PREAMBLE = textwrap.dedent(
    """\
    <start_of_turn>user
    {user}
    <end_of_turn>
    <start_of_turn>model
    """
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="user message (default: a simple algebra question)",
    )
    parser.add_argument(
        "--preamble",
        choices=["plain", "thinking"],
        default="plain",
        help="prompt template: 'plain' = standard chat; 'thinking' = seed "
        "a <|channel>...<channel|> reasoning block (default: plain)",
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
        HierarchicalGemma4Sampler,
        make_gemma4_tokenizer,
        resolve_marker_ids,
    )

    template = THINKING_PREAMBLE if args.preamble == "thinking" else PLAIN_PREAMBLE
    prompt = template.format(user=args.prompt)

    print("=== prompt ===")
    print(prompt)

    print("\n=== loading model & weights ===")
    model = gm.nn.Gemma4_E4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E4B_IT)
    tok = make_gemma4_tokenizer()
    markers = resolve_marker_ids(tok)

    rng = jax.random.key(args.seed)

    if args.mode in ("both", "off"):
        print("\n=== mode: pruning OFF (stock equivalent) ===")
        off = HierarchicalGemma4Sampler(
            model=model,
            params=params,
            tokenizer=tok,
            markers=markers,
            sampling=gm.text.Greedy(),
            enabled=False,
        )
        out_off = off.sample(prompt, max_new_tokens=args.max_new_tokens, rng=rng)
        print(out_off)

    if args.mode in ("both", "on"):
        print("\n=== mode: pruning ON ===")
        on = HierarchicalGemma4Sampler(
            model=model,
            params=params,
            tokenizer=tok,
            markers=markers,
            sampling=gm.text.Greedy(),
            enabled=True,
        )
        out_on = on.sample(prompt, max_new_tokens=args.max_new_tokens, rng=rng)
        print(out_on)


if __name__ == "__main__":
    main()
