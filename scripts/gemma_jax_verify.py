"""Phase-1 verification harness for the Gemma 4 JAX hierarchical-CoT sampler.

Three gated tests:

  1. `token_init` — the Gemma 4 tokenizer registers `<return|>` into an
     unused SentencePiece slot and round-trips it to a single id.  The
     native `<|channel>` and `<channel|>` markers tokenize to single
     pieces too.

  2. `sampler_noop` — on a prompt that never emits hierarchical markers,
     `PruningChatSampler(pruning_enabled=True)` produces the same tokens
     as the stock `gm.text.ChatSampler`, given the same seed and prompt.
     This is the "does not break ordinary generation" gate.

  3. `prune_parity` — hand-construct a
     `[prompt, <|channel>, thought, <channel|>, summary, <return|>]`
     sequence; run the same cache-rewind + summary-re-forward that
     `PruningChatSampler._apply_prune` performs; assert that the
     post-prune next-token logits match a reference forward on
     `[prompt, summary]` within cosine > 0.99 and top-1 agreement.

All three must pass before the plan's Phase 1 is complete.

Usage:
    python scripts/gemma_jax_verify.py --test token_init
    python scripts/gemma_jax_verify.py --test sampler_noop
    python scripts/gemma_jax_verify.py --test prune_parity
    python scripts/gemma_jax_verify.py --test all

Run on a host with JAX + `gemma` package + Gemma 4 weights accessible
(by default the GCS paths from `gm.ckpts.CheckpointPath`).
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def _import_gemma():
    """Delayed imports so `--test token_init` fails fast with a clear error
    if the gemma package isn't installed, rather than at module load."""
    from gemma import gm  # noqa: F401  (imported for side effects)
    return gm


# ---------------------------------------------------------------------------
# Test 1: token_init
# ---------------------------------------------------------------------------


def test_token_init(*, verbose: bool = True) -> bool:
    from lib.gemma_jax import (
        CHANNEL_CLOSE_TOKEN,
        CHANNEL_OPEN_TOKEN,
        RETURN_TOKEN,
        make_gemma4_tokenizer,
        resolve_marker_ids,
    )

    tok = make_gemma4_tokenizer(return_token_slot=0)
    if verbose:
        print(f"  tokenizer vocab_size = {tok.vocab_size}")

    markers = resolve_marker_ids(tok)
    if verbose:
        print(f"  channel_open  id = {markers.channel_open:6d}  "
              f"({CHANNEL_OPEN_TOKEN!r})")
        print(f"  channel_close id = {markers.channel_close:6d}  "
              f"({CHANNEL_CLOSE_TOKEN!r})")
        print(f"  return        id = {markers.return_:6d}  "
              f"({RETURN_TOKEN!r})")

    for name, tok_id, expected in [
        ("channel_open", markers.channel_open, CHANNEL_OPEN_TOKEN),
        ("channel_close", markers.channel_close, CHANNEL_CLOSE_TOKEN),
        ("return", markers.return_, RETURN_TOKEN),
    ]:
        decoded = tok.decode([tok_id])
        if decoded != expected:
            print(f"  FAIL: decode([{tok_id}]) = {decoded!r}, "
                  f"expected {expected!r} for {name}")
            return False
    if verbose:
        print("  OK: all three markers round-trip to single ids.")
    return True


# ---------------------------------------------------------------------------
# Test 2: sampler_noop
# ---------------------------------------------------------------------------


DEFAULT_NOOP_PROMPT = (
    "What is the capital of France? Answer in one short sentence."
)


def test_sampler_noop(
    *,
    prompt: str = DEFAULT_NOOP_PROMPT,
    max_new_tokens: int = 40,
    seed: int = 0,
    verbose: bool = True,
) -> bool:
    """Stock `gm.text.ChatSampler` vs `PruningChatSampler(pruning_enabled=True)`
    on a prompt that emits no hierarchical markers.  Token streams must be
    identical (same seed, same sampling method, same model)."""
    gm = _import_gemma()
    from lib.gemma_jax import (
        PruningChatSampler,
        make_gemma4_tokenizer,
        resolve_marker_ids,
    )

    if verbose:
        print("  loading Gemma4_E4B and params (this may take a minute)...")
    model = gm.nn.Gemma4_E4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E4B_IT)
    tok = make_gemma4_tokenizer()
    markers = resolve_marker_ids(tok)

    stock = gm.text.ChatSampler(
        model=model,
        params=params,
        tokenizer=tok,
        sampling=gm.text.Greedy(),
    )
    ours = PruningChatSampler(
        model=model,
        params=params,
        tokenizer=tok,
        markers=markers,
        sampling=gm.text.Greedy(),
        pruning_enabled=True,
    )

    rng = jax.random.key(seed)
    if verbose:
        print("  chatting with stock gm.text.ChatSampler...")
    stock_text = stock.chat(prompt, max_new_tokens=max_new_tokens, rng=rng)
    if verbose:
        print("  chatting with PruningChatSampler(pruning_enabled=True)...")
    ours_text = ours.chat(prompt, max_new_tokens=max_new_tokens, rng=rng)

    stock_tokens = np.asarray(stock.last_state.predicted_tokens[0])
    ours_tokens = np.asarray(ours.last_state.predicted_tokens[0])
    n = min(max_new_tokens, stock_tokens.shape[0], ours_tokens.shape[0])
    stock_tokens = stock_tokens[:n]
    ours_tokens = ours_tokens[:n]

    matches = int((stock_tokens == ours_tokens).sum())
    if verbose:
        print(f"  stock : {stock_tokens.tolist()}")
        print(f"  ours  : {ours_tokens.tolist()}")
        print(f"  match : {matches}/{n}")
        print(f"  stock text: {stock_text!r}")
        print(f"  ours  text: {ours_text!r}")

    if matches != n:
        diff_positions = np.where(stock_tokens != ours_tokens)[0]
        print(f"  FAIL: token streams diverge at positions "
              f"{diff_positions[:10].tolist()}")
        return False
    print("  OK: token streams are identical (pruner is a no-op when no "
          "<return|> fires).")
    return True


# ---------------------------------------------------------------------------
# Test 3: prune_parity
# ---------------------------------------------------------------------------


DEFAULT_PARITY_PROMPT = (
    "<start_of_turn>user\nSolve 2+2.<end_of_turn>\n<start_of_turn>model\n"
)


def test_prune_parity(
    *,
    prompt: str = DEFAULT_PARITY_PROMPT,
    thought_text: str = "I need to think carefully about arithmetic.",
    summary_text: str = "4",
    cosine_threshold: float = 0.99,
    verbose: bool = True,
) -> bool:
    """Reference forward on `[prompt, summary]` vs. the same logits produced
    by rewinding the cache after a full `[prompt, <|channel>, thought,
    <channel|>, summary, <return|>]` pass and re-forwarding the summary
    into the rewound cache.

    Mirrors the cache-surgery logic inside `PruningChatSampler._apply_prune`
    without going through the sampling state machine, so the test is
    independent of the segmented decode loop.
    """
    gm = _import_gemma()
    from lib.gemma_jax import (
        make_gemma4_tokenizer,
        resolve_marker_ids,
    )

    if verbose:
        print("  loading Gemma4_E4B and params...")
    model = gm.nn.Gemma4_E4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA4_E4B_IT)
    tok = make_gemma4_tokenizer()
    markers = resolve_marker_ids(tok)

    prompt_ids = tok.encode(prompt, add_bos=True)
    thought_ids = tok.encode(thought_text)
    summary_ids = tok.encode(summary_text)

    ref_logits = _forward_logits(
        model, params, prompt_ids + summary_ids, cache_length=256
    )
    pruned_logits = _teacher_forced_prune_logits(
        model=model,
        params=params,
        markers=markers,
        prompt_ids=prompt_ids,
        thought_ids=thought_ids,
        summary_ids=summary_ids,
        cache_length=256,
    )

    cos = _cosine(ref_logits, pruned_logits)
    top1_ref = int(jnp.argmax(ref_logits))
    top1_pruned = int(jnp.argmax(pruned_logits))
    if verbose:
        print(f"  cosine(ref, pruned)  = {cos:.6f}")
        print(f"  top1 ref             = {top1_ref}  "
              f"({tok.decode([top1_ref])!r})")
        print(f"  top1 pruned          = {top1_pruned}  "
              f"({tok.decode([top1_pruned])!r})")

    if cos < cosine_threshold:
        print(f"  FAIL: cosine {cos:.6f} below threshold {cosine_threshold}")
        return False
    if top1_ref != top1_pruned:
        print(f"  FAIL: top-1 disagree: {top1_ref} vs {top1_pruned}")
        return False
    print("  OK: post-prune logits match reference within tolerance.")
    return True


def _forward_logits(model, params, token_ids: list[int], *, cache_length: int):
    """Run a fresh forward over `token_ids` and return last-token logits [V]."""
    cache = model.init_cache(
        batch_size=1,
        dtype=jax.tree.leaves(params)[0].dtype,
        cache_length=cache_length,
        sharding=None,
    )
    tokens = jnp.asarray([token_ids], dtype=jnp.int32)
    L = tokens.shape[1]
    positions = jnp.arange(L, dtype=jnp.int32)[None, :]
    attn = jnp.concatenate(
        [jnp.ones((1, L), dtype=jnp.bool_),
         jnp.zeros((1, cache_length - L), dtype=jnp.bool_)],
        axis=-1,
    )
    out = model.apply(
        {"params": params},
        tokens=tokens,
        cache=cache,
        positions=positions,
        attention_mask=attn[:, None, :],
    )
    return out.logits[0, -1]


def _teacher_forced_prune_logits(
    *,
    model,
    params,
    markers,
    prompt_ids: list[int],
    thought_ids: list[int],
    summary_ids: list[int],
    cache_length: int,
):
    """Run the full hand-crafted sequence to populate the cache, then
    rewind to the `<|channel>` position and re-forward the summary one
    token at a time, matching `PruningChatSampler._apply_prune`.  Return
    the last-summary-token logits."""
    from gemma.gm.utils import _cache_helper

    full_ids = (
        prompt_ids
        + [markers.channel_open]
        + thought_ids
        + [markers.channel_close]
        + summary_ids
        + [markers.return_]
    )
    L = len(full_ids)
    cache = model.init_cache(
        batch_size=1,
        dtype=jax.tree.leaves(params)[0].dtype,
        cache_length=cache_length,
        sharding=None,
    )
    tokens = jnp.asarray([full_ids], dtype=jnp.int32)
    positions = jnp.arange(L, dtype=jnp.int32)[None, :]
    attn_full = jnp.concatenate(
        [jnp.ones((1, L), dtype=jnp.bool_),
         jnp.zeros((1, cache_length - L), dtype=jnp.bool_)],
        axis=-1,
    )
    out = model.apply(
        {"params": params},
        tokens=tokens,
        cache=cache,
        positions=positions,
        attention_mask=attn_full[:, None, :],
    )
    cache_full = out.cache

    open_pos = len(prompt_ids)  # absolute position of <|channel>
    pruned_cache = _cache_helper.Cache(cache_full).set_end_index(
        jnp.asarray(open_pos, dtype=jnp.int32)
    ).cache

    last_logits = None
    cache_cur = pruned_cache
    for i, tok in enumerate(summary_ids):
        pos = open_pos + i
        step_mask = jnp.arange(cache_length) < (pos + 1)
        attn_mask = (attn_full * step_mask)[:, None, :]
        out = model.apply(
            {"params": params},
            tokens=jnp.asarray([[tok]], dtype=jnp.int32),
            cache=cache_cur,
            positions=jnp.asarray([[pos]], dtype=jnp.int32),
            attention_mask=attn_mask,
        )
        cache_cur = out.cache
        last_logits = out.logits[0, 0]

    return last_logits


def _cosine(a, b) -> float:
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    num = float(jnp.sum(a * b))
    denom = float(jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12)
    return num / denom


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


TESTS: dict[str, Callable[..., bool]] = {
    "token_init": test_token_init,
    "sampler_noop": test_sampler_noop,
    "prune_parity": test_prune_parity,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        choices=list(TESTS.keys()) + ["all"],
        default="all",
        help="which test to run (default: all, in order)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="max new tokens for sampler_noop (default: 40)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for sampler_noop (default: 0)",
    )
    args = parser.parse_args()

    tests_to_run = (
        list(TESTS.keys()) if args.test == "all" else [args.test]
    )

    failed = []
    for name in tests_to_run:
        print(f"\n=== {name} ===")
        kwargs = {}
        if name == "sampler_noop":
            kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
            }
        ok = TESTS[name](**kwargs)
        status = "PASS" if ok else "FAIL"
        print(f"=== {name}: {status} ===")
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nFAILED: {failed}")
        sys.exit(1)
    print("\nALL PASSED")


if __name__ == "__main__":
    main()
