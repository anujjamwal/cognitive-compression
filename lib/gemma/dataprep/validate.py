"""Well-formedness and preservation checks for hierarchical-CoT traces.

Two independent checks live here:

`is_well_formed(hcot)` — STRUCTURAL validity:
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

`preserves_raw_trace(hcot, raw_trace)` — SEMANTIC preservation:
  - inverse of segmentation: every innermost nested
    `<|channel>thought\\n{body}\\n<channel|>{summary}<return|>` block is
    replaced by `{body}` (the summary is discarded), iteratively to a
    fixed point.  The result is then whitespace-normalised and compared
    to a whitespace-normalised raw trace.  If they match, the teacher
    inserted markers around verbatim spans without paraphrasing,
    reordering, or otherwise editing the underlying thought text.
  - This is the runtime check that backs the "wrap, don't write"
    invariant enforced by the prompt in `prompts.py`.

Both checks are cheap and local to a single record.  Use them together
in stage-2 dataprep:

    if not is_well_formed(hcot):
        drop
    elif not preserves_raw_trace(hcot, raw_trace):
        drop
    else:
        keep
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"<\|channel>|<channel\|>|<return\|>")

# Matches an INNERMOST nested block: opens with `<|channel>thought\n`,
# body contains no further `<|channel>` open, closes with `\n<channel|>`,
# a non-greedy summary span, and `<return|>`.  Group 1 captures the body
# without surrounding newlines or the marker-block `thought\n` prefix.
_NESTED_UNWRAP_RE = re.compile(
    r"<\|channel>thought\n((?:(?!<\|channel>).)*?)\n<channel\|>.*?<return\|>",
    re.DOTALL,
)


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


def unwrap_nested_blocks(hcot: str) -> str:
    """Inverse of segmentation: replace every innermost
    `<|channel>thought\\n{body}\\n<channel|>{summary}<return|>` with
    `{body}`, iteratively to a fixed point.

    The outer `<|channel>thought ... <channel|>{final_answer}` block has
    no matching `<return|>` and is left intact.  After full unwrap the
    result should be byte-identical (modulo whitespace) to the raw trace
    the teacher was annotating, provided the teacher honoured the
    "wrap, don't write" invariant.
    """
    prev = None
    current = hcot
    for _ in range(32):
        prev = current
        current = _NESTED_UNWRAP_RE.sub(lambda m: m.group(1), current)
        if current == prev:
            break
    return current


def preserves_raw_trace(hcot: str, raw_trace: str) -> bool:
    """Return True iff `unwrap_nested_blocks(hcot)` equals `raw_trace`
    after light whitespace normalisation.

    This is the runtime check that the teacher inserted markers around
    verbatim spans of the raw trace rather than paraphrasing them.
    Whitespace normalisation tolerates blank-line shifts and tab/space
    differences around the inserted markers; semantic edits (added
    words, reordered sentences, fixed typos) still surface as a
    mismatch.
    """
    return _normalize_whitespace(unwrap_nested_blocks(hcot)) == \
        _normalize_whitespace(raw_trace)


def preservation_diff(hcot: str, raw_trace: str) -> str | None:
    """Return `None` if `hcot` preserves `raw_trace`, otherwise a
    short human-readable diff of the first divergent character.

    Useful for debugging stage-2 dataprep failures: it tells you
    WHERE the teacher's output drifted from the raw trace, not just
    that it did.
    """
    a = _normalize_whitespace(unwrap_nested_blocks(hcot))
    b = _normalize_whitespace(raw_trace)
    if a == b:
        return None
    window = 60
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return (
                f"divergence at char {i} (of unwrapped len={len(a)}, "
                f"raw len={len(b)}):\n"
                f"  unwrapped: ...{a[max(0, i - window):i + window]!r}...\n"
                f"  raw      : ...{b[max(0, i - window):i + window]!r}..."
            )
    return (
        f"length-only mismatch: unwrapped={len(a)}, raw={len(b)}; "
        f"shorter is a prefix of longer"
    )


def _normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace into a single space; strip ends.

    Tolerates blank-line / indentation drift introduced by the teacher
    around its inserted markers without masking content edits.
    """
    return re.sub(r"\s+", " ", text).strip()
