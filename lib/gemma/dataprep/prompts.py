"""Prompt templates for Gemini-based hierarchical-CoT segmentation.

The teacher (Gemini 3.x) takes a raw reasoning trace produced by Gemma 4
(via its native `<|channel>thought ... <channel|>` wrapper) and annotates
it with NESTED sub-chain-of-thought blocks, each terminated by the
project's `<return|>` boundary token.

The hard invariant this prompt enforces:

    The teacher INSERTS markers around verbatim spans of the raw trace.
    The teacher does NOT paraphrase, reorder, polish, fix, or otherwise
    modify the thought text.  The only new text the teacher writes is
    the short summary that follows each nested `<channel|>` and precedes
    its `<return|>`.

This is a strict superset invariant of "is_well_formed" in validate.py.
Structural validity is a necessary but insufficient condition;
preservation of the thought text is an additional, semantic condition
the prompt is responsible for.
"""

PERSONA = (
    "You are a research assistant helping prepare a dataset for training a "
    "new kind of reasoning LLM that generates hierarchical chains of "
    "thought and can discard sub-reasoning at inference time via KV-cache "
    "pruning. Your job is to annotate existing reasoning traces with "
    "structure — never to rewrite them."
)

PROMPT = """\
## Background

Traditional chain-of-thought (CoT) reasoning is a long flat stream of
tokens that grows as the model works through a problem.  This is
wasteful: most intermediate reasoning is only useful locally, but it
stays in the context window forever, inflating the KV cache and
slowing generation.

If you read a real CoT trace carefully, the model is reasoning
HIERARCHICALLY: it breaks the problem into subproblems, solves each
one, and combines results.  The flat trace is a depth-first
linearisation of that tree.

We want to teach a model to emit that tree EXPLICITLY using Gemma 4's
native thinking-channel tokens plus a new boundary token:

    <|channel>thought    opens a (sub-)chain-of-thought
    <channel|>           closes the reasoning part; what follows is a summary
    <return|>            ends a sub-chain-of-thought; the pruner fires here

A nested sub-chain-of-thought block has exactly four parts:

    <|channel>thought
    <verbatim reasoning text copied from the raw trace>
    <channel|>
    <short summary paragraph you write>
    <return|>

At inference time, when the model emits `<return|>`, our runtime
prunes the entire `<|channel>...<channel|>` span plus its markers,
leaving only the summary.  The summary therefore must be a faithful,
self-contained compression of the reasoning it replaces: downstream
steps see ONLY the summary, not the reasoning that produced it.

A fully hierarchical trace is the original outer wrapper with
zero or more nested sub-blocks INSERTED inside its reasoning span:

    <|channel>thought                    <-- root, from the raw trace
    <root reasoning, possibly with nested sub-blocks inside it>
    <channel|>                           <-- root close, from the raw trace
    <final answer>                       <-- Gemma's answer, from the raw trace

## The core invariant

You are an annotator, not an author.  Specifically:

1. **Wrap, don't write.**  Each nested sub-block wraps a CONTIGUOUS
   SPAN of the raw reasoning text.  You choose where the span starts
   and ends, then insert exactly four things around it:
   - `<|channel>thought\\n` at the start of the span,
   - `\\n<channel|>` at the end of the span,
   - the summary you wrote,
   - `<return|>` at the very end.
   The characters BETWEEN those four insertions are byte-identical to
   the corresponding span of the raw trace.

2. **No paraphrasing.**  Do not change wording, do not normalise
   spelling, do not fix grammar, do not reflow sentences, do not
   delete filler ("Hmm,", "OK so,", "Wait —"), do not add transition
   phrases between blocks.  If the raw trace contains a typo, the
   typo stays.

3. **No reordering.**  Sub-blocks appear in the same order their
   reasoning appears in the raw trace.  You may nest but you may not
   move text earlier or later.

4. **No structural decoration.**  Do not add headings, bullet lists,
   numbered steps, or whitespace beyond what was already there
   (except the single newline immediately after `<|channel>thought`
   and immediately before `<channel|>`, which is part of the marker
   block).

5. **Summaries are the only new text.**  The text between
   `<channel|>` and `<return|>` is the ONE place you author new
   prose.  Constraints on that text:
   - 1-3 sentences.
   - Self-contained — the rest of the reasoning will be removed and
     only this summary survives.
   - Faithful — it must accurately compress what the just-closed
     block computed/decided.
   - If the raw trace already contains a sentence at the end of the
     wrapped span that reads like a summary ("So x = 408." or "This
     gives us the formula F = ma."), prefer that sentence verbatim.
     Only synthesise a new summary if no suitable sentence exists.

6. **Don't touch the root or the final answer.**  The outer
   `<|channel>thought` open, the outer `<channel|>` close, and
   everything after that close (Gemma's final answer) appear in
   your output exactly as they appear in the raw trace.  The outer
   block has NO `<return|>` of its own — only nested blocks do.

7. **Nesting depth: aim for 2-4 levels.**  Going deeper costs more
   than it saves; staying flat saves nothing.  If the trace is
   genuinely short and flat (one or two logical steps), return it
   with NO nested sub-blocks — just the outer wrapper and the final
   answer.  An honest non-segmentation is better than invented
   structure.

## Worked example

### Input raw trace

    <|channel>thought
    I need to compute 17 * 24.
    Let me split it: 17 * 24 = 17 * 20 + 17 * 4 = 340 + 68 = 408.
    Let me double-check by a different route: 20 * 24 - 3 * 24 = 480 - 72 = 408. Same answer.
    So 17 * 24 = 408.
    <channel|>
    The answer is 408.

### Valid output (one good way to segment it)

    <hierarchical-cot>
    <|channel>thought
    I need to compute 17 * 24.
    <|channel>thought
    Let me split it: 17 * 24 = 17 * 20 + 17 * 4 = 340 + 68 = 408.
    <channel|>17 * 24 = 408 via the split 17*20 + 17*4.<return|>
    <|channel>thought
    Let me double-check by a different route: 20 * 24 - 3 * 24 = 480 - 72 = 408. Same answer.
    <channel|>Cross-check confirms 408 via 20*24 - 3*24.<return|>
    So 17 * 24 = 408.
    <channel|>
    The answer is 408.
    </hierarchical-cot>

Notice:
  - The four sentences of the raw trace appear in the output in the
    same order, with the same words.  Nothing was rewritten.
  - Two nested sub-blocks wrap two CONTIGUOUS spans (each one a
    single sentence here).  No span overlap, no skipping.
  - Summaries ("17 * 24 = 408 via the split ...", "Cross-check
    confirms 408 ...") are short, self-contained, and faithful.
    Neither summary is taken verbatim from the trace because the
    trace doesn't contain a clean compression at that boundary.
  - "So 17 * 24 = 408." is NOT wrapped — it's already the natural
    boundary between the two checks and the final conclusion, and
    wrapping it would just create a trivial sub-block.
  - The outer `<|channel>thought` open, the outer `<channel|>`
    close, and "The answer is 408." are byte-identical to the raw
    trace.

## Forbidden output examples

These would all be REJECTED:

  - Rewriting "Let me split it" as "Decomposing the multiplication".
    Even if "better", this is paraphrase and violates rule 2.
  - Promoting "So 17 * 24 = 408." into a summary and dropping the
    period.  The original punctuation is part of the trace.
  - Reordering the cross-check to appear before the split.  Order is
    fixed by the raw trace.
  - Adding "**Step 1:**" or numbered bullets.  No decoration.
  - A nested block whose `<channel|>` is followed immediately by
    `<return|>` with no summary text.

## Procedure

Work in this order:

1. Read the raw trace once, identifying natural logical
   subproblems and their summary sentences (if any).
2. Pick contiguous spans for the most useful 2-4 sub-blocks.
   Larger spans (covering a whole subproblem) are better than tiny
   ones (covering a single equation).
3. For each chosen span, write a 1-3 sentence summary.  Try to use
   an existing sentence at the end of the span; only synthesise if
   the trace doesn't provide one.
4. Assemble the output: outer wrapper + nested blocks inserted,
   followed by Gemma's verbatim final answer.

## Self-check before you output

Mentally perform this collapse on your output:

    For every innermost nested block
        <|channel>thought\\n{body}\\n<channel|>{summary}<return|>
    replace the entire block (including all four markers and the
    summary) with `{body}`.  Repeat to a fixed point.

The result MUST equal the raw trace exactly, character for
character.  If it doesn't, you paraphrased, reordered, or added
something — fix that before producing the final output.

Then run the structural checklist:
  - Exactly one outer `<|channel>thought` and one outer `<channel|>`.
  - The outer block has NO `<return|>`.
  - Every nested `<|channel>thought` is closed by `<channel|>` and
    terminated by `<return|>`.
  - Every nested `<channel|>` is followed by a non-empty summary
    before its `<return|>`.
  - Nesting is balanced and non-crossing.

## Output format

Wrap the full segmented trace in `<hierarchical-cot>...</hierarchical-cot>`
tags.  Nothing outside the tags is read.  Do not include the
collapse / checklist output — those are mental steps you run before
emitting the final wrapped trace.
"""

INPUT_TEMPLATE = """\
## Inputs

### Question

{question}

### Raw Gemma trace (thinking + final answer, exactly as Gemma produced it)

{raw_trace}

### Gemma's final answer (this is the text after the outer `<channel|>`)

{final_answer}

### Expected answer (ground truth, for your reference only)

{expected_answer}

## Reminder

Return a single `<hierarchical-cot>...</hierarchical-cot>` block.
Inside it: the outer `<|channel>thought ... <channel|>` from the raw
trace with nested sub-blocks INSERTED into the reasoning span,
followed by Gemma's verbatim final answer.

Strict invariant: the only new text you write is the summary that
sits between each nested `<channel|>` and its `<return|>`.  Every
other character must match the raw trace, in order.  If you cannot
honour this, return the raw trace unchanged (no nested sub-blocks).
"""
