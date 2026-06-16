"""Gemini-based hierarchical-CoT segmentation for Gemma 4 traces.

Takes (question, raw_trace, expected_answer) records produced by `sample.py`
and calls Gemini 3.x with the prompt defined in `prompts.py` to insert
nested `<|channel>thought ... <channel|> <summary> <return|>` blocks into
each trace.

Isolated from the Qwen pipeline in `lib/dataprep/segment.py` — this module
does not import from that one; duplicated retry/parallel logic is kept
local so the two pipelines can evolve independently.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import re
from typing import Any

import tenacity

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from .prompts import INPUT_TEMPLATE, PERSONA, PROMPT

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("GEMMA_SEGMENT_MODEL", "gemini-3.1-pro-preview")
DEFAULT_PARALLELISM = int(os.environ.get("PARALLELISM", "4"))
DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs_gemma_segment")
MAX_TOKENS = 32000


# ---------------------------------------------------------------------------
# Single-record segmentation
# ---------------------------------------------------------------------------

def _is_retryable(e: Exception) -> bool:
    s = str(e)
    return "429" in s or "ResourceExhausted" in s or "503" in s or "UNAVAILABLE" in s


@tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    wait=tenacity.wait_exponential_jitter(initial=30, jitter=30),
    stop=tenacity.stop_after_attempt(8),
)
def _call_gemini(prompt: str, model: str) -> str:
    if genai is None:
        raise ImportError(
            "google-genai is not installed. `pip install google-genai`."
        )
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=PERSONA)]),
            types.Content(role="user", parts=[types.Part.from_text(text=PROMPT)]),
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=MAX_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        ),
    )
    return response.text or ""


_HCOT_RE = re.compile(r"<hierarchical-cot>(.*?)</hierarchical-cot>", re.DOTALL)


def parse_result(text: str) -> tuple[str, str]:
    """Extract the annotated trace from the model's output.

    Returns (hierarchical_cot, full_raw_response).
    """
    m = _HCOT_RE.search(text)
    hcot = m.group(1).strip() if m else text.strip()
    return hcot, text


def segment_trace(
    question: str,
    raw_trace: str,
    final_answer: str,
    expected_answer: str,
    model: str | None = None,
    output_file: str | None = None,
) -> tuple[str, str]:
    """Segment a single (question, raw_trace, final_answer, expected_answer)
    record.

    `raw_trace` is Gemma's thinking body ending with `<channel|>`.
    `final_answer` is Gemma's own answer text that followed the outer close.
    `expected_answer` is ground truth from the source dataset.

    If `output_file` exists and is non-empty, return its cached contents
    without invoking the API.
    """
    if output_file and os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            cached = f.read().strip()
        if cached:
            logger.info("cache hit: %s", output_file)
            return parse_result(cached)

    if model is None:
        model = DEFAULT_MODEL

    prompt = INPUT_TEMPLATE.format(
        question=question,
        raw_trace=raw_trace,
        final_answer=final_answer,
        expected_answer=expected_answer,
    )
    text = _call_gemini(prompt, model=model)

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

    return parse_result(text)


# ---------------------------------------------------------------------------
# Parallel segmentation of a list of records
# ---------------------------------------------------------------------------

def segment_batch(
    records: list[dict[str, Any]],
    model: str | None = None,
    parallelism: int = DEFAULT_PARALLELISM,
    output_dir: str | None = None,
) -> list[tuple[str, str] | Exception]:
    """Segment a list of records in parallel.

    Each record must have `question`, `raw_trace`, and `expected_answer`
    string fields.  Returns a list the same length as `records`, with either
    a `(hcot, raw_response)` tuple or an Exception for each slot.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    results: list[Any] = [None] * len(records)

    def _worker(idx: int, rec: dict[str, Any]):
        output_file = os.path.join(output_dir, f"example_{idx}.txt")
        try:
            hcot, raw = segment_trace(
                question=rec["question"],
                raw_trace=rec["raw_trace"],
                final_answer=rec.get("final_answer", ""),
                expected_answer=rec.get("expected_answer", ""),
                model=model,
                output_file=output_file,
            )
            return idx, (hcot, raw)
        except Exception as e:
            logger.error("segment example %d failed: %s", idx, e)
            return idx, e

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as ex:
        futures = [ex.submit(_worker, i, r) for i, r in enumerate(records)]
        for fut in concurrent.futures.as_completed(futures):
            idx, out = fut.result()
            results[idx] = out

    return results
