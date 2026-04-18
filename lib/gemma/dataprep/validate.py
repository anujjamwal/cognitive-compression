"""Well-formedness checks for hierarchical-CoT traces.

A valid Gemma 4 hierarchical trace obeys:

  - exactly one outer channel pair: opens with `<|channel>thought` and
    closes with `<channel|>`; text after the outer close is the final
    answer.  The outer block does NOT end with `<return|>`.
  - every NESTED sub-chain-of-thought is of the form
    `<|channel>thought ... <channel|> <summary> <return|>` — i.e. every
    nested `<|channel>` eventually reaches a `<return|>`, and every
    `<return|>` has a matching nested `<|channel>` open.
  - nesting is balanced (no crossing, no dangling markers).
  - there is at least one character of text between each `<channel|>`
    and its following `<return|>` (the summary paragraph).

These are cheap token-level checks — they don't verify semantic quality
of the segmentation, just structural validity.  The caller can apply
stricter quality filters (e.g. minimum summary length, minimum depth) on
top.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"<\|channel>|<channel\|>|<return\|>")


def _tokenize_markers(text: str) -> list[tuple[str, int]]:
    """Yield (marker, position) for every marker token in `text`."""
    return [(m.group(0), m.start()) for m in _TOKEN_RE.finditer(text)]


def is_well_formed(hcot: str, min_summary_len: int = 1) -> bool:
    """Return True iff `hcot` is a well-formed hierarchical trace.

    See module docstring for the structural rules.
    """
    if not hcot or len(hcot) < 20:
        return False
    markers = _tokenize_markers(hcot)
    if not markers:
        return False

    # Root shape: first marker must be an open, and there must be at least
    # one close somewhere.  Final answer text lives after the LAST top-level
    # `<channel|>` — we do not enforce anything about it beyond that the
    # root block exists.
    first_marker, _ = markers[0]
    if first_marker != "<|channel>":
        return False

    # Walk the marker list with a stack.  Each stack frame records the
    # position of the open and the position of its close (if seen).
    stack: list[dict] = []
    depth = 0
    root_closed = False

    for marker, pos in markers:
        if marker == "<|channel>":
            stack.append({"open_pos": pos, "close_pos": None})
            depth += 1
        elif marker == "<channel|>":
            if not stack:
                return False  # close without open
            if stack[-1]["close_pos"] is not None:
                # Two consecutive closes without a return — only legal for
                # the root, which never nests a close inside another close.
                return False
            stack[-1]["close_pos"] = pos
            if depth == 1:
                # Root close — mark it, but do NOT expect a <return|> for it.
                root_closed = True
        elif marker == "<return|>":
            if not stack:
                return False  # return without open
            frame = stack[-1]
            if frame["close_pos"] is None:
                return False  # return without a preceding close
            if depth == 1:
                return False  # root must not have a <return|>
            summary = hcot[frame["close_pos"] + len("<channel|>"): pos]
            if len(summary.strip()) < min_summary_len:
                return False
            stack.pop()
            depth -= 1

    # At end: exactly one frame must remain (the root), and it must be
    # closed.  No nested frames should be open.
    if len(stack) != 1:
        return False
    if not root_closed:
        return False
    return True


def count_depth(hcot: str) -> int:
    """Return the maximum nesting depth of `hcot` (root counts as 1).

    Useful for dataset filtering / logging.  Does NOT validate correctness;
    pair with `is_well_formed` for validated records.
    """
    max_d = 0
    d = 0
    for marker, _ in _tokenize_markers(hcot):
        if marker == "<|channel>":
            d += 1
            max_d = max(max_d, d)
        elif marker == "<return|>":
            d -= 1
    return max_d
