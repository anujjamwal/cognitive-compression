"""JAX / Flax implementation of context-compression hierarchical-CoT sampling
for Gemma 4 (Phase 1 of the plan).

Public entry points:

    from lib.gemma_jax import (
        make_gemma4_tokenizer,
        resolve_marker_ids,
        PruningChatSampler,
        MarkerIds,
    )

Phase 1 scope: the pruning mechanism only.  SFT, RL, and benchmark modules
live in later phases and import from the same package.

Note: PruningChatSampler is imported lazily to avoid pulling in the full
kauldron/tensorflow import chain at module load time, which would conflict
with sentencepiece's C++ mutex on macOS arm64 when custom token protos are
modified via LoadFromSerializedProto.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .markers import (
    CHANNEL_CLOSE_TOKEN,
    CHANNEL_OPEN_TOKEN,
    DEFAULT_RETURN_TOKEN_SLOT,
    RETURN_TOKEN,
    THINK_TOKEN,
)
from .setup import MarkerIds, make_gemma4_tokenizer, resolve_marker_ids

if TYPE_CHECKING:
    from .prune_sampler import PruningChatSampler


def __getattr__(name: str):
    if name == "PruningChatSampler":
        from .prune_sampler import PruningChatSampler  # noqa: PLC0415
        return PruningChatSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CHANNEL_CLOSE_TOKEN",
    "CHANNEL_OPEN_TOKEN",
    "DEFAULT_RETURN_TOKEN_SLOT",
    "MarkerIds",
    "PruningChatSampler",
    "RETURN_TOKEN",
    "THINK_TOKEN",
    "make_gemma4_tokenizer",
    "resolve_marker_ids",
]
