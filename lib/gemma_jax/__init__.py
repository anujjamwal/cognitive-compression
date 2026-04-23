"""JAX / Flax implementation of context-compression hierarchical-CoT sampling
for Gemma 4 (Phase 1 of the plan).

Public entry points:

    from lib.gemma_jax import (
        make_gemma4_tokenizer,
        resolve_marker_ids,
        HierarchicalGemma4Sampler,
        MarkerIds,
    )

Phase 1 scope: the pruning mechanism only.  SFT, RL, and benchmark modules
live in later phases and import from the same package.
"""
from .markers import (
    CHANNEL_CLOSE_TOKEN,
    CHANNEL_OPEN_TOKEN,
    DEFAULT_RETURN_TOKEN_SLOT,
    RETURN_TOKEN,
    THINK_TOKEN,
)
from .prune_sampler import HierarchicalGemma4Sampler
from .setup import MarkerIds, make_gemma4_tokenizer, resolve_marker_ids

__all__ = [
    "CHANNEL_CLOSE_TOKEN",
    "CHANNEL_OPEN_TOKEN",
    "DEFAULT_RETURN_TOKEN_SLOT",
    "HierarchicalGemma4Sampler",
    "MarkerIds",
    "RETURN_TOKEN",
    "THINK_TOKEN",
    "make_gemma4_tokenizer",
    "resolve_marker_ids",
]
