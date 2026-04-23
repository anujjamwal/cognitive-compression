"""Pure-Python collapse logic for hierarchical-CoT traces.

Given a well-formed hierarchical trace of the form

    <|channel>thought ... (nested <|channel>...<channel|> summary <return|>) ...
    <channel|> final answer

`collapse_nested` replaces every *nested* sub-block with its summary
paragraph, iteratively peeling nesting off from the innermost level outward.
The outer root block (which has no `<return|>` terminator) is preserved.

The regex matches an INNERMOST block: a `<|channel>thought` open followed by
content that does NOT contain another open, followed by `<channel|>`, a
summary span, and a `<return|>`.  `re.DOTALL` lets the body span lines.
Iterative substitution to a fixed point handles arbitrary nesting depth.

This module is intentionally dependency-free (stdlib `re` only) so it can
be imported from either the HF or JAX training path without dragging in a
heavy ML stack.
"""
from __future__ import annotations

import re

_INNERMOST_BLOCK_RE = re.compile(
    r"<\|channel>(?:(?!<\|channel>).)*?<channel\|>(.*?)<return\|>",
    re.DOTALL,
)


def collapse_nested(hcot: str) -> str:
    """Replace every nested sub-block with its summary paragraph.

    The outer root block `<|channel>thought ... <channel|>` + final answer is
    preserved intact, because the root has no `<return|>` and therefore does
    not match the regex.
    """
    prev = None
    current = hcot
    # Safety cap: pathological inputs shouldn't loop forever.
    for _ in range(32):
        prev = current
        current = _INNERMOST_BLOCK_RE.sub(
            lambda m: m.group(1).strip(), current,
        )
        if current == prev:
            break
    return current
