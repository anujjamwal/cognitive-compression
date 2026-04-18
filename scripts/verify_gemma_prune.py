"""Verification harness for Gemma 4 prune-aware inference.

Run on a host with a GPU and the Gemma 4 weights available.  Validates two
properties before we commit to training:

  1. **Token init sanity.** The new `<return|>` row in `embed_tokens` and
     `embed_tokens_per_layer` lands close to the seed-phrase average, with a
     norm in the same order of magnitude as existing rows.

  2. **Prune-correctness parity.** A forward pass on the full sequence, then
     in-place cache truncation simulating a `<return|>` event, then a forward
     pass that re-processes the surviving suffix, produces logits within
     numerical tolerance of a reference run on the post-prune sequence
     directly.  This is the gating test for strategy 1A: if it fails on the
     E4B sliding+global+KV-shared layer mix, we need to revisit.

Usage:
    python scripts/verify_gemma_prune.py \\
        --model google/gemma-4-E4B-it \\
        --device cuda --dtype bfloat16
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Make `lib/` importable when running as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from gemma import (  # noqa: E402
    CHANNEL_CLOSE_TOKEN,
    CHANNEL_OPEN_TOKEN,
    RETURN_TOKEN,
    RETURN_TOKEN_SEED,
    prepare_gemma_model,
)
from gemma.generate import _truncate_kv_cache_layer_aware  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("verify_gemma_prune")


# ---------------------------------------------------------------------------
# Test 1: token-init sanity
# ---------------------------------------------------------------------------

def test_token_init(model, tokenizer) -> None:
    """Confirm the new `<return|>` row is the average of the seed phrase rows
    in BOTH the main embedding table and the per-layer (PLE) table.
    """
    logger.info("=== Test 1: token-init sanity ===")

    return_id = tokenizer.convert_tokens_to_ids(RETURN_TOKEN)
    seed_ids = tokenizer.encode(RETURN_TOKEN_SEED, add_special_tokens=False)
    logger.info("Seed %r -> sub-token ids %s", RETURN_TOKEN_SEED, seed_ids)

    # Resolve text model (matches lib/gemma/setup.py path resolution).
    text_model = (
        getattr(model, "language_model", None) and model.language_model.model
    ) or model.model

    main = text_model.embed_tokens
    expected_main = main.weight[seed_ids].mean(0)
    actual_main = main.weight[return_id]
    diff_main = (expected_main - actual_main).abs().max().item()
    logger.info(
        "embed_tokens row[%d]  L2=%.4f  expected_L2=%.4f  max|diff|=%.6e",
        return_id, actual_main.norm().item(), expected_main.norm().item(), diff_main,
    )
    assert diff_main < 1e-4, f"embed_tokens seeding diverged: max|diff|={diff_main}"

    ple = getattr(text_model, "embed_tokens_per_layer", None)
    if ple is not None:
        expected_ple = ple.weight[seed_ids].mean(0)
        actual_ple = ple.weight[return_id]
        diff_ple = (expected_ple - actual_ple).abs().max().item()
        logger.info(
            "embed_tokens_per_layer row[%d]  L2=%.4f  expected_L2=%.4f  max|diff|=%.6e",
            return_id, actual_ple.norm().item(), expected_ple.norm().item(), diff_ple,
        )
        assert diff_ple < 1e-4, f"PLE seeding diverged: max|diff|={diff_ple}"
    else:
        logger.warning("No PLE table on this checkpoint — skipping PLE assertion.")

    logger.info("Test 1 PASSED")


# ---------------------------------------------------------------------------
# Test 2: prune parity
# ---------------------------------------------------------------------------

def _build_test_sequence(tokenizer, device) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Construct (full_ids, pruned_ref_ids, open_pos, close_pos).

    full sequence:    "<prompt><|channel>thought\\n<thought><channel|><summary><return|>"
    pruned reference: "<prompt><|channel><channel|><summary><return|>"  (thought removed)
    """
    open_id = tokenizer.convert_tokens_to_ids(CHANNEL_OPEN_TOKEN)
    close_id = tokenizer.convert_tokens_to_ids(CHANNEL_CLOSE_TOKEN)
    return_id = tokenizer.convert_tokens_to_ids(RETURN_TOKEN)

    prompt_ids = tokenizer.encode(
        "Question: What is 2 plus 3? Answer: ", add_special_tokens=False,
    )
    thought_text_ids = tokenizer.encode(
        "thought\nLet me think step by step. Two plus three equals five.",
        add_special_tokens=False,
    )
    summary_ids = tokenizer.encode(
        "The answer is 5.", add_special_tokens=False,
    )

    full = (
        prompt_ids
        + [open_id]
        + thought_text_ids
        + [close_id]
        + summary_ids
        + [return_id]
    )
    open_pos = len(prompt_ids)
    close_pos = open_pos + 1 + len(thought_text_ids)

    pruned_ref = (
        prompt_ids
        + [open_id]
        + [close_id]
        + summary_ids
        + [return_id]
    )

    full_t = torch.tensor([full], dtype=torch.long, device=device)
    pruned_ref_t = torch.tensor([pruned_ref], dtype=torch.long, device=device)
    return full_t, pruned_ref_t, open_pos, close_pos


@torch.no_grad()
def test_prune_parity(model, tokenizer, atol: float = 5e-2, rtol: float = 5e-2) -> None:
    """Compare logits from prune-then-continue vs reference forward pass.

    Tolerance is loose because bf16 forward + cache truncation introduces
    non-trivial numerical drift; we want to confirm the *qualitative*
    behaviour (top-k token agreement, low KL), not bit-equivalence.
    """
    logger.info("=== Test 2: prune-correctness parity ===")
    device = next(model.parameters()).device

    full_ids, ref_ids, open_pos, close_pos = _build_test_sequence(tokenizer, device)
    logger.info(
        "full_len=%d  ref_len=%d  open_pos=%d  close_pos=%d",
        full_ids.shape[1], ref_ids.shape[1], open_pos, close_pos,
    )

    # ---- Reference: single forward on the post-prune sequence ----
    ref_out = model(input_ids=ref_ids, use_cache=False)
    ref_logits_last = ref_out.logits[:, -1, :].float()

    # ---- Prune path: forward full -> truncate cache -> forward suffix ----
    cache = DynamicCache(config=model.config)
    pre_out = model(input_ids=full_ids, past_key_values=cache, use_cache=True)
    # Sanity: cache should hold every token we just processed.
    cache_len_before = full_ids.shape[1]
    logger.info("cache populated to length %d", cache_len_before)

    prune_map = {0: (open_pos, close_pos)}
    new_cache_len = _truncate_kv_cache_layer_aware(
        cache=cache,
        prune_map=prune_map,
        batch_size=1,
        old_seq_len=cache_len_before,
    )
    logger.info("after truncate: cache length = %d (expected %d)",
                new_cache_len, open_pos + 1)
    assert new_cache_len == open_pos + 1, (
        f"Truncated cache length {new_cache_len} != expected {open_pos + 1}"
    )

    # The post-prune visible sequence equals ref_ids; the cache holds
    # [0..open_pos], so the next forward should re-process [<channel|>, summary..., <return|>]
    # at positions [open_pos+1 ..].
    suffix = ref_ids[:, open_pos + 1 :]
    cache_position = torch.arange(
        open_pos + 1, ref_ids.shape[1], dtype=torch.long, device=device,
    )
    post_out = model(
        input_ids=suffix,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
    )
    prune_logits_last = post_out.logits[:, -1, :].float()

    # ---- Compare ----
    max_abs = (ref_logits_last - prune_logits_last).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        ref_logits_last, prune_logits_last, dim=-1,
    ).item()
    ref_top1 = ref_logits_last.argmax(-1).item()
    prune_top1 = prune_logits_last.argmax(-1).item()
    logger.info(
        "max|ref - prune| = %.4f   cosine = %.6f   top1 ref=%d prune=%d",
        max_abs, cos, ref_top1, prune_top1,
    )

    # Loose pass: top-1 agreement + high cosine.  Tighter checks (KL, top-k
    # overlap) can be added once we know the noise floor.
    assert cos > 0.99, f"Cosine similarity too low: {cos}"
    assert ref_top1 == prune_top1, (
        f"Top-1 disagreement: ref={ref_top1} ({tokenizer.decode([ref_top1])!r}) "
        f"prune={prune_top1} ({tokenizer.decode([prune_top1])!r})"
    )

    logger.info("Test 2 PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn", default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    logger.info("Loading %s (dtype=%s, attn=%s)", args.model, args.dtype, args.attn)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn,
        device_map=args.device,
    )
    model.eval()

    model, tokenizer = prepare_gemma_model(model, tokenizer)

    test_token_init(model, tokenizer)
    test_prune_parity(model, tokenizer)

    logger.info("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
