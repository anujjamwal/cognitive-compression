"""Gemma 4 hierarchical-CoT pipeline (isolated from the Qwen2.5 codepath).

Modules:
    markers : project special-token name constants
    setup   : model / tokenizer load + <return|> seeding (incl. PLE)

Coming next:
    generate : prune-aware generation with layer-aware KV pruning (1A)
    dataset  : SFT data prep with Gemma chat template + nested thought blocks
    trainer  : SFT trainer subclass (prune-aware staged loss only)
    rewards  : general-purpose GRPO rewards
    dataprep : Gemini hierarchization + open-reasoning sampling
"""
from .markers import (
    THINK_TOKEN,
    CHANNEL_OPEN_TOKEN,
    CHANNEL_CLOSE_TOKEN,
    RETURN_TOKEN,
    RETURN_TOKEN_SEED,
)
from .setup import prepare_gemma_model, DEFAULT_TOKEN_SEED

__all__ = [
    "THINK_TOKEN",
    "CHANNEL_OPEN_TOKEN",
    "CHANNEL_CLOSE_TOKEN",
    "RETURN_TOKEN",
    "RETURN_TOKEN_SEED",
    "prepare_gemma_model",
    "DEFAULT_TOKEN_SEED",
]
