"""Prompt templates for Gemini-based hierarchical-CoT segmentation.

The teacher (Gemini 3.x) takes a raw reasoning trace produced by Gemma 4
(via its native `<|channel>thought ... <channel|>` wrapper) and annotates it
with NESTED sub-chain-of-thought blocks, each terminated by the project's
`<return|>` boundary token.

Output invariants:
  - The outermost channel is preserved intact; it contains all of Gemma's
    original reasoning text, now with nested structure inserted.  The
    outermost `<channel|>` still marks the transition from reasoning to the
    final answer.  The outermost block does NOT end with `<return|>` — it is
    the root, never pruned.
  - Each nested `<|channel>thought ... <channel|>` block must be followed by
    a short summary paragraph and a single `<return|>` token.  The summary
    replaces the nested content after a prune event at inference time; it
    must be self-contained enough that downstream reasoning can continue
    from the summary alone.
  - Original wording is preserved wherever it appears; Gemini's role is to
    INSERT structure, not rewrite.  Summary paragraphs may be newly written
    if the original trace does not already contain a suitable one.
"""

PERSONA = (
    "You are a research assistant helping prepare a dataset for training a "
    "new kind of reasoning LLM that generates hierarchical chains of thought "
    "and can discard sub-reasoning at inference time via KV-cache pruning."
)

PROMPT = """\
## Background

Traditional chain-of-thought (CoT) reasoning is a long flat stream of tokens
that grows as the model works through a problem.  This is wasteful — most of
the intermediate reasoning is only useful locally, but it stays in the
context window forever, inflating the KV cache and slowing generation.

If you read a real CoT trace carefully, you find the model is actually
reasoning HIERARCHICALLY: it breaks the problem into subproblems, solves each
one, and combines the results.  The flat trace is a depth-first linearization
of that tree.

We want to teach the model to emit that tree structure EXPLICITLY, using
Gemma 4's native thinking-channel tokens plus a new boundary token:

    <|channel>thought    opens a (sub-)chain-of-thought
    <channel|>           closes the reasoning part; what follows is a summary
    <return|>            ends a sub-chain-of-thought; the pruner fires here

A sub-chain-of-thought looks like this:

    <|channel>thought
    <reasoning>
    <channel|>
    <summary paragraph>
    <return|>

At inference time, when the model emits `<return|>`, our runtime prunes the
entire `<|channel>...<channel|>` block plus its markers, leaving only the
summary paragraph.  The summary therefore has to be a faithful, self-
contained compression of the reasoning: downstream steps see only the
summary, not the reasoning that produced it.

Sub-chains can be nested to arbitrary depth.  A fully hierarchical trace
looks like:

    <|channel>thought
    <root-level reasoning text>

    <|channel>thought
    <sub-problem 1 reasoning>
    <channel|>
    <sub-problem 1 summary>
    <return|>

    <more root-level reasoning that uses the summary above>

    <|channel>thought
    <sub-problem 2 reasoning>

    <|channel>thought
    <sub-sub-problem 2.1 reasoning>
    <channel|>
    <sub-sub-problem 2.1 summary>
    <return|>

    <more sub-problem 2 reasoning>
    <channel|>
    <sub-problem 2 summary>
    <return|>

    <root-level synthesis>
    <channel|>
    <final answer>

Rules for the ROOT `<|channel>thought ... <channel|>` block:
  - It is never pruned; it wraps the entire reasoning.
  - It does NOT end with `<return|>`.  After the root `<channel|>` comes the
    final answer (plain text, outside any channel).
  - The original Gemma trace already starts with `<|channel>thought` and
    ends with `<channel|>` + final answer — keep those as the outer shell.

Rules for NESTED sub-chain-of-thought blocks:
  - Every nested block is `<|channel>thought ... <channel|> <summary> <return|>`.
  - Nested blocks appear inside the parent's reasoning text, between its
    `<|channel>thought` and `<channel|>`.
  - The summary paragraph between `<channel|>` and `<return|>` replaces the
    nested reasoning after pruning — it must be self-sufficient.

## Task

You will be given:
  - The original question / problem statement.
  - The raw Gemma 4 reasoning trace (already wrapped in the outer
    `<|channel>thought ... <channel|>` + final answer).
  - The expected final answer (ground truth, if available).

Your job is to SEGMENT the raw trace into hierarchical form by inserting
nested `<|channel>thought ... <channel|> <summary> <return|>` blocks.

Strict rules:

1. Preserve the original wording.  Do NOT rewrite, paraphrase, or reorder
   Gemma's reasoning text.  Insert markers around existing text; do not
   replace existing text with your own.
2. Write summary paragraphs ONLY when no suitable summary sentence exists in
   the original trace at that boundary.  Prefer to promote an existing
   summary-like sentence into the `<channel|> ... <return|>` slot.
3. Every nested sub-block must be COMPLETE: it has an open, a close, a
   summary, and a `<return|>`.  Partial/unclosed sub-blocks are invalid.
4. Do not add headings, bullets, or any other structure not present in the
   original trace.  Only the marker tokens and (when necessary) summary
   sentences may be added.
5. Depth: aim for 2-4 levels of nesting total.  Over-nesting is worse than
   under-nesting.  If the trace is short and genuinely flat, return it with
   no nested sub-blocks.
6. The outer channel and final answer are already in place — do not modify
   them except to insert nested sub-blocks inside the outer reasoning.

At the end, write a brief sanity check: count the nested sub-blocks, and
confirm every one has all four parts (open, close, summary, return).

## Output format

Wrap the full segmented trace in `<hierarchical-cot> ... </hierarchical-cot>`
tags.  Nothing outside the tags will be used.
"""

INPUT_TEMPLATE = """\
## Inputs

### Question

{question}

### Raw Gemma trace

{raw_trace}

### Expected answer

{expected_answer}
"""
