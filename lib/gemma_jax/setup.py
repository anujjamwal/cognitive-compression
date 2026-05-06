"""Tokenizer setup and marker-ID resolution for the Gemma 4 JAX path.

Two things happen here:

1. Build a `gm.text.Gemma4Tokenizer` with `<return|>` mapped into an unused
   `<unusedN>` slot.  The base `Gemma4Tokenizer.custom_tokens` machinery
   expects `special_tokens.CUSTOM` to exist; the shipped `_Gemma4SpecialTokens`
   enum does not define it (upstream oversight).  We patch that attribute on
   the tokenizer instance before SentencePiece loads the proto.

2. Resolve integer IDs for the hierarchical-CoT markers.  `<|channel>` and
   `<channel|>` are native Gemma 4 vocab pieces; `<return|>` is the custom
   slot we just registered.  All three must resolve to single token IDs — a
   multi-piece token would break the sampler's "one event per predicted
   token" assumption.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

from gemma import gm

from .markers import (
    CHANNEL_CLOSE_TOKEN,
    CHANNEL_OPEN_TOKEN,
    DEFAULT_RETURN_TOKEN_SLOT,
    RETURN_TOKEN,
)


# Gemma 3 and Gemma 4 share the same SentencePiece layout for the
# `<unused0>..<unused98>` range, starting at id 6.  We hardcode this here
# because `_Gemma4SpecialTokens` doesn't (yet) expose a `CUSTOM` member.
_GEMMA4_CUSTOM_BASE = 6


class MarkerIds(NamedTuple):
    """Resolved integer token IDs for the hierarchical-CoT markers."""

    channel_open: int
    channel_close: int
    return_: int


def make_gemma4_tokenizer(
    *,
    return_token_slot: int = DEFAULT_RETURN_TOKEN_SLOT,
) -> gm.text.Gemma4Tokenizer:
    """Return a Gemma 4 tokenizer with `<return|>` registered at
    `<unused{return_token_slot}>`.

    The returned tokenizer encodes the string `"<return|>"` to a single ID
    equal to `_GEMMA4_CUSTOM_BASE + return_token_slot`.
    """
    if not 0 <= return_token_slot <= 98:
        raise ValueError(
            f"return_token_slot {return_token_slot} out of [0, 98]"
        )

    # The base class's `_add_custom_tokens` reads `self.special_tokens.CUSTOM`
    # to locate the slot.  Gemma 4's enum doesn't define CUSTOM, so we inject
    # it directly onto the class.  special_tokens is a _DisplayEnumType
    # (enum class), and Python allows adding regular class attributes to it.
    _probe = gm.text.Gemma4Tokenizer()
    if not hasattr(_probe.special_tokens, "CUSTOM"):
        type(_probe.special_tokens).CUSTOM = _GEMMA4_CUSTOM_BASE

    # Warm-up: load the tokenizer once without custom tokens to initialise the
    # sentencepiece C++ runtime on macOS/arm64.  A cold call to
    # spm.LoadFromSerializedProto() with a modified proto crashes with
    # "mutex lock failed: Invalid argument" until the C++ layer has been
    # initialised via at least one plain Load / LoadFromSerializedProto call.
    _ = _probe._sp  # noqa: SLF001

    # Now create the tokenizer with the custom token registered.
    tok = gm.text.Gemma4Tokenizer(
        custom_tokens={return_token_slot: RETURN_TOKEN},
    )
    # Force SP init (which triggers proto patching via upstream _add_custom_tokens).
    _ = tok._sp  # noqa: SLF001

    # Sanity check: round-trip the new token.
    encoded = tok.encode(RETURN_TOKEN)
    expected = _GEMMA4_CUSTOM_BASE + return_token_slot
    if encoded != [expected]:
        raise RuntimeError(
            f"Custom token registration failed: encode({RETURN_TOKEN!r}) "
            f"= {encoded}, expected [{expected}]"
        )
    return tok


def resolve_marker_ids(
    tokenizer: gm.text.Gemma4Tokenizer,
) -> MarkerIds:
    """Resolve `<|channel>`, `<channel|>`, `<return|>` to single integer IDs.

    Raises if any token tokenizes to more than one piece — the sampler's
    marker-detection stack requires one token per event.
    """

    def _single(tok: str) -> int:
        ids = tokenizer.encode(tok)
        if len(ids) != 1:
            raise ValueError(
                f"Expected {tok!r} to encode to 1 token, got {len(ids)}: {ids}"
            )
        return ids[0]

    return MarkerIds(
        channel_open=_single(CHANNEL_OPEN_TOKEN),
        channel_close=_single(CHANNEL_CLOSE_TOKEN),
        return_=_single(RETURN_TOKEN),
    )


def _patch_gemma4_custom_tokens(tok: gm.text.Gemma4Tokenizer) -> None:
    """Replace `tok._add_custom_tokens` with a version that reads
    `tok._custom_base` instead of `tok.special_tokens.CUSTOM`.

    This is the minimum surgery needed to make `custom_tokens={N: '<return|>'}`
    work on Gemma 4 until upstream adds `_Gemma4SpecialTokens.CUSTOM`.
    """
    from sentencepiece import sentencepiece_model_pb2  # type: ignore

    def _add_custom_tokens(serialized_proto: bytes) -> bytes:
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(serialized_proto)
        base = tok._custom_base  # noqa: SLF001
        for i, token in tok.custom_tokens.items():
            if not 0 <= i <= 98:
                raise ValueError(
                    f"Custom token id {i} for {token!r} not in [0, 98]"
                )
            piece = proto.pieces[base + i]
            if piece.piece != f"<unused{i}>":
                raise AssertionError(
                    f"Expected piece at id {base + i} to be '<unused{i}>', "
                    f"got {piece.piece!r}. Vocab layout differs from expected."
                )
            piece.piece = token
            if proto.trainer_spec.user_defined_symbols:
                for idx, sym in enumerate(
                    proto.trainer_spec.user_defined_symbols
                ):
                    if sym == f"<unused{i}>":
                        proto.trainer_spec.user_defined_symbols[idx] = token
                        break
        return proto.SerializeToString()

    object.__setattr__(tok, "_add_custom_tokens", _add_custom_tokens)
