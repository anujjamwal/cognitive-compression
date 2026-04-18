"""Sample problems from an open-reasoning dataset, run them through Gemma 4
with thinking enabled, and emit (question, raw_trace, final_answer) records.

The traces this produces are Gemma's NATURAL reasoning outputs — they follow
Gemma's own style and always have the outer `<|channel>thought ... <channel|>`
wrapper.  Gemini's job in the next stage is to INSERT nested structure into
these traces, not rewrite them.

Default source dataset: `open-thoughts/OpenThoughts-114k` (broad general
reasoning, ~114k examples).  Other slices can be passed via --source.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from typing import Any, Iterable

import torch
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "open-thoughts/OpenThoughts-114k"
DEFAULT_MODEL = "google/gemma-4-E4B-it"

# Regex to split a Gemma-4-it response into (thinking, answer) on the
# `<channel|>` close token.  The raw string decoded from the tokenizer
# contains these exact substrings.
_CHANNEL_SPLIT_RE = re.compile(r"<channel\|>", re.DOTALL)
_CHANNEL_OPEN_RE = re.compile(r"<\|channel>thought\s*", re.DOTALL)


def extract_question(record: dict[str, Any], question_key: str) -> str:
    """Pull the question text out of an OpenThoughts-style record.

    Supports both bare string fields and chat-style lists.
    """
    val = record[question_key]
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        # Chat-style: take the first user turn.
        for msg in val:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        # Fallback: concatenate all string contents.
        return "\n".join(
            m.get("content", "") for m in val if isinstance(m, dict)
        )
    raise TypeError(f"Unsupported question field type: {type(val)}")


def split_trace(raw_text: str) -> tuple[str, str]:
    """Split a Gemma 4 completion into (thinking_body, final_answer).

    `raw_text` is the decoded generation (tokens AFTER the assistant prompt,
    not including `<|turn>model\\n`).  Expected shape:

        <|channel>thought\\n<reasoning><channel|><answer>

    Returns (reasoning_with_outer_channel_markers, answer).  If the trace
    does not contain the expected markers, returns ("", raw_text) and logs a
    warning — the caller can then discard or flag the record.
    """
    parts = _CHANNEL_SPLIT_RE.split(raw_text, maxsplit=1)
    if len(parts) != 2:
        logger.warning("Trace missing `<channel|>` close marker; skipping split.")
        return "", raw_text.strip()
    thinking_body, answer = parts[0], parts[1]
    # thinking_body should start with `<|channel>thought`; keep it intact so
    # downstream segmentation sees the full outer wrapper.
    if not _CHANNEL_OPEN_RE.search(thinking_body):
        logger.warning("Trace missing `<|channel>thought` open marker.")
    # Re-attach the close marker we split on, so downstream has the full
    # outer shell to modify.
    return f"{thinking_body.rstrip()}\n<channel|>".lstrip(), answer.strip()


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    questions: list[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> list[str]:
    """Run one batch of questions through Gemma 4 with thinking enabled.

    Returns decoded completion strings (tokens after the assistant prompt).
    """
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=False,
        )
        for q in questions
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(
        model.device,
    )
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # Strip the prompt tokens off each completion before decoding.
    prompt_len = enc["input_ids"].shape[1]
    completions = [
        tokenizer.decode(out[i, prompt_len:], skip_special_tokens=False)
        for i in range(out.shape[0])
    ]
    return completions


def collect_traces(
    source: str,
    model_id: str,
    n_samples: int,
    offset: int,
    batch_size: int,
    max_new_tokens: int,
    question_key: str,
    answer_key: str | None,
    do_sample: bool,
    temperature: float,
    dtype: torch.dtype,
    attn_impl: str,
) -> Dataset:
    """Load `n_samples` from `source`, run them through `model_id`, and
    return a `Dataset` of `{question, raw_trace, final_answer, expected_answer}`.
    """
    # Local import so importing this module does not require torch-heavy deps.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading %s [offset=%d, limit=%d]", source, offset, n_samples)
    stream = load_dataset(source, split="train", streaming=True)
    records = list(stream.skip(offset).take(n_samples))
    logger.info("Loaded %d source records", len(records))

    logger.info("Loading model %s (dtype=%s, attn=%s)", model_id, dtype, attn_impl)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation=attn_impl, device_map="auto",
    )
    model.eval()

    out_rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        questions = [extract_question(r, question_key) for r in chunk]
        completions = generate_batch(
            model, tokenizer, questions, max_new_tokens, do_sample, temperature,
        )
        for rec, q, comp in zip(chunk, questions, completions):
            raw_trace, final_answer = split_trace(comp)
            if not raw_trace:
                continue  # malformed — skip
            expected = rec.get(answer_key, "") if answer_key else ""
            out_rows.append(
                {
                    "question": q,
                    "raw_trace": raw_trace,
                    "final_answer": final_answer,
                    "expected_answer": expected,
                    "source": source,
                    "generator_model": model_id,
                }
            )
        logger.info("Processed %d/%d", min(start + batch_size, len(records)), len(records))

    logger.info("Emitting %d valid (question, trace, answer) records", len(out_rows))
    return Dataset.from_list(out_rows)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help=f"HF dataset repo id (default: {DEFAULT_SOURCE})")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--question-key", default="problem",
                   help="Field on source records that holds the question text")
    p.add_argument("--answer-key", default="expected_answer",
                   help="Optional field with ground-truth answer (empty if absent)")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn", default="sdpa",
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--out-repo", required=True,
                   help="HF repo id to push the (question, raw_trace, ...) dataset to")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    ds = collect_traces(
        source=args.source,
        model_id=args.model,
        n_samples=args.n_samples,
        offset=args.offset,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        question_key=args.question_key,
        answer_key=args.answer_key,
        do_sample=args.do_sample,
        temperature=args.temperature,
        dtype=dtype,
        attn_impl=args.attn,
    )
    logger.info("Pushing %d records to %s", len(ds), args.out_repo)
    ds.push_to_hub(args.out_repo)
    logger.info("Done.")


if __name__ == "__main__":
    main()
