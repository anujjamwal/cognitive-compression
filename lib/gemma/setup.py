"""Gemma 4 model + tokenizer setup for hierarchical-CoT training and inference.

Adds a single project-specific special token (`<return|>`) and seeds its
embedding rows in both the main embedding table and Gemma 4's per-layer
embedding (PLE) table.

The PLE table (`embed_tokens_per_layer`) is unique to the smaller Gemma 4
variants (E2B, E4B); it has shape `[vocab_size, num_hidden_layers * hidden_size_per_layer_input]`
(for E4B that is `[262144, 42 * 256 = 10752]`) and contributes a token-keyed,
position-independent residual at every decoder layer.  Any new token must be
seeded in this table too or the per-layer modulation it adds will be junk.

Because `tie_word_embeddings=True`, `lm_head.weight` aliases
`embed_tokens.weight`; we assert this rather than copying.
"""
from __future__ import annotations

import logging
from typing import Mapping, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .markers import RETURN_TOKEN, RETURN_TOKEN_SEED

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_SEED: Mapping[str, str] = {
    RETURN_TOKEN: RETURN_TOKEN_SEED,
}


def _resolve_text_model(model: PreTrainedModel):
    """Return the `Gemma4TextModel` instance regardless of wrapper class.

    Supports both `Gemma4ForCausalLM` (text-only) and
    `Gemma4ForConditionalGeneration` (multimodal) load paths.
    """
    # Gemma4ForCausalLM:                  model.model is Gemma4TextModel
    # Gemma4ForConditionalGeneration:     model.language_model.model is Gemma4TextModel
    candidates = []
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model"):
            candidates.append(lm.model)
        candidates.append(lm)
    if hasattr(model, "model"):
        candidates.append(model.model)
    candidates.append(model)

    for c in candidates:
        if hasattr(c, "embed_tokens_per_layer") or hasattr(c, "embed_tokens"):
            return c
    raise AttributeError(
        "Could not locate the Gemma4 text model on the given model instance; "
        "expected `embed_tokens` (and optionally `embed_tokens_per_layer`)."
    )


def _seed_row(
    table: torch.nn.Embedding,
    target_id: int,
    seed_ids: list[int],
    label: str,
) -> None:
    """Average the embedding rows of `seed_ids` into row `target_id`."""
    rows = table.weight[seed_ids]
    avg = rows.mean(dim=0)
    table.weight[target_id] = avg
    logger.info(
        "  %-12s row[%d] <- mean of %s rows %s (dim=%d)",
        label, target_id, table.weight.shape[1], seed_ids, table.weight.shape[1],
    )


def prepare_gemma_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    token_seed: Optional[Mapping[str, str]] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Register `<return|>` (and any extra tokens in `token_seed`) and seed
    their embedding rows in `embed_tokens` and `embed_tokens_per_layer`.

    Returns the (mutated) model and tokenizer.
    """
    if token_seed is None:
        token_seed = DEFAULT_TOKEN_SEED

    new_tokens = [t for t in token_seed.keys()
                  if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id
                  or t not in tokenizer.get_vocab()]

    if new_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(tokenizer.additional_special_tokens) + new_tokens}
        )
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Added %d special tokens to Gemma 4 tokenizer: %s",
                    len(new_tokens), new_tokens)
    else:
        logger.info("All requested special tokens already present; skipping resize.")

    text_model = _resolve_text_model(model)
    main_embed: torch.nn.Embedding = text_model.embed_tokens  # type: ignore[assignment]
    ple_embed: Optional[torch.nn.Embedding] = getattr(text_model, "embed_tokens_per_layer", None)

    if ple_embed is None:
        logger.info("No per-layer embedding table found on this model "
                    "(likely a 31B / 26B-A4B variant); skipping PLE seeding.")

    with torch.no_grad():
        for tok, seed in token_seed.items():
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            if tok_id is None or tok_id == tokenizer.unk_token_id:
                raise ValueError(f"Token {tok!r} is not registered in the tokenizer.")

            seed_ids = tokenizer.encode(seed, add_special_tokens=False)
            if not seed_ids:
                raise ValueError(f"Seed phrase {seed!r} for {tok!r} tokenized to nothing.")

            _seed_row(main_embed, tok_id, seed_ids, f"{tok}/embed")
            if ple_embed is not None:
                _seed_row(ple_embed, tok_id, seed_ids, f"{tok}/PLE")

        # Verify lm_head is tied to embed_tokens; otherwise we'd need to copy
        # the new row over too.  Gemma 4 ships with tie_word_embeddings=True,
        # so this should always hold.
        lm_head = model.get_output_embeddings()
        if lm_head is not None and lm_head.weight.data_ptr() != main_embed.weight.data_ptr():
            logger.warning(
                "lm_head is NOT tied to embed_tokens on this checkpoint; "
                "copying new-token rows into lm_head explicitly."
            )
            for tok in token_seed.keys():
                tok_id = tokenizer.convert_tokens_to_ids(tok)
                lm_head.weight[tok_id] = main_embed.weight[tok_id]

    return model, tokenizer
