"""SFT dataset preparation for Gemma 4 hierarchical-CoT training.

Converts hierarchical records from `lib.gemma.dataprep` into TRL-style
`{prompt, completion}` chat dicts, optionally emitting multiple variants
per source record to bridge the training-inference gap introduced by
Option A pruning (all three marker tokens are deleted at inference).

Variants per source record:

    "full"       - the full hierarchical trace as produced by Gemini:
                   outer <|channel>thought ... nested blocks ...<channel|>
                   followed by Gemma's final answer.  Teaches the model to
                   emit hierarchical structure.

    "collapsed"  - every nested sub-block `<|channel>thought ... <channel|>
                   <summary> <return|>` is replaced by its `<summary>` text.
                   Markers for the NESTED blocks disappear entirely; the
                   outer channel wrapper and the final answer are preserved.
                   Teaches the model to continue from a post-prune context
                   where sub-thinking has already been removed.

Chat-template override: Gemma 4's default template runs `strip_thinking`
on every role=model message.  In SFT that would silently destroy our
training signal (assistant completions are templated with
`add_generation_prompt=False`, which makes them subject to the strip).
`install_sft_chat_template` swaps the macro for an identity pass-through.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nested-block collapse
# ---------------------------------------------------------------------------

# Matches an INNERMOST nested block: an `<|channel>thought` open followed by
# content that does NOT contain another open, followed by `<channel|>`, a
# summary span, and a `<return|>`.  DOTALL so the reasoning body can span
# multiple lines.  Because the regex forbids nested opens inside, iteratively
# substituting until a fixed point peels nesting off one level at a time
# from the inside out.
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


# ---------------------------------------------------------------------------
# TRL record construction
# ---------------------------------------------------------------------------

def convert_to_trl(
    example: dict[str, Any],
    variant: str = "full",
    question_key: str = "question",
    hcot_key: str = "hierarchical_cot",
) -> dict[str, Any]:
    """Convert one hierarchical record into a TRL chat dict.

    The completion content is the full hierarchical trace (variant="full")
    or its nested-collapse (variant="collapsed").  The outer `<|channel>`
    wrapper + final answer is preserved in both variants; in "collapsed"
    the nested markers are all gone.
    """
    if variant not in ("full", "collapsed"):
        raise ValueError(f"unknown variant {variant!r}; use 'full' or 'collapsed'")

    question = example[question_key]
    hcot = example[hcot_key]
    if variant == "collapsed":
        hcot = collapse_nested(hcot)

    return {
        "prompt": [{"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": hcot}],
        "variant": variant,
    }


def expand_to_variants(
    dataset: Dataset,
    variants: Iterable[str] = ("full", "collapsed"),
    question_key: str = "question",
    hcot_key: str = "hierarchical_cot",
) -> Dataset:
    """Emit one TRL record per (source_record, variant) combination.

    For a dataset of N source records and `variants=("full", "collapsed")`,
    the output has 2N records.
    """
    variants = list(variants)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for rec in dataset:
        for v in variants:
            try:
                rows.append(convert_to_trl(
                    rec, variant=v,
                    question_key=question_key, hcot_key=hcot_key,
                ))
            except (KeyError, ValueError) as e:
                skipped += 1
                logger.warning("skipping record (%s): %s", v, e)
    if skipped:
        logger.info("skipped %d records during expand_to_variants", skipped)
    logger.info("expanded %d source -> %d TRL records", len(dataset), len(rows))
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Chat-template patching
# ---------------------------------------------------------------------------

# Match the strip_thinking macro in the Gemma 4 chat template.  The block
# spans from "{%- macro strip_thinking(text) -%}" through the corresponding
# "{%- endmacro -%}".
_STRIP_THINKING_MACRO_RE = re.compile(
    r"\{%-?\s*macro\s+strip_thinking\s*\(\s*text\s*\)\s*-?%\}"
    r".*?"
    r"\{%-?\s*endmacro\s*-?%\}",
    re.DOTALL,
)

_IDENTITY_MACRO = (
    "{%- macro strip_thinking(text) -%}{{- text -}}{%- endmacro -%}"
)


def install_sft_chat_template(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Replace the `strip_thinking` macro with identity so assistant-turn
    thinking channels survive chat-template tokenization during SFT.

    Mutates `tokenizer.chat_template` in place and returns the tokenizer
    for chaining.  If the macro is not present (custom template already
    installed), the template is left alone and a warning is logged.
    """
    tmpl = tokenizer.chat_template
    if tmpl is None:
        logger.warning("tokenizer has no chat_template; nothing to patch")
        return tokenizer

    new_tmpl, n_subs = _STRIP_THINKING_MACRO_RE.subn(_IDENTITY_MACRO, tmpl)
    if n_subs == 0:
        logger.warning(
            "strip_thinking macro not found in chat_template; leaving untouched. "
            "Either the template was already patched, or this is not a Gemma 4 "
            "tokenizer."
        )
        return tokenizer

    tokenizer.chat_template = new_tmpl
    logger.info("installed SFT chat template (strip_thinking -> identity)")
    return tokenizer
