"""Answer extraction and verification for math benchmarks.

Uses the official QwenLM/PolyMath evaluation logic (added as a git submodule
under ``third_party/PolyMath``) for answer comparison via ``math_equal``.

Falls back to HuggingFace's ``math-verify`` library when the official
evaluation code is unavailable.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import the official PolyMath evaluation helpers.
# ``scripts.py`` lives at third_party/PolyMath/eval/scripts.py and does
# ``from bundled.… import …``, so we need that directory on sys.path.
# ---------------------------------------------------------------------------
_POLYMATH_EVAL_DIR = str(Path(__file__).resolve().parents[2] / "third_party" / "PolyMath" / "eval")

_has_official = False
try:
    if _POLYMATH_EVAL_DIR not in sys.path:
        sys.path.insert(0, _POLYMATH_EVAL_DIR)
    from scripts import math_equal as _official_math_equal  # type: ignore[import-untyped]
    from scripts import extract_answer as _official_extract_answer  # type: ignore[import-untyped]
    _has_official = True
    logger.info("Using official PolyMath evaluation from %s", _POLYMATH_EVAL_DIR)
except ImportError as exc:
    logger.warning(
        "Could not import official PolyMath evaluation (%s). "
        "Falling back to math-verify.",
        exc,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_answer(model_output: str) -> str | None:
    """Extract the answer from model output.

    Uses the official PolyMath ``extract_boxed_content`` logic when available
    (strips whitespace, takes the **first** boxed match).  Otherwise falls
    back to a local implementation that takes the **last** match.
    """
    if _has_official:
        # Official: strips spaces then finds ALL \boxed{} matches, uses first
        text = model_output.replace(" ", "")
        pattern = re.compile(r"boxed{")
        matches = pattern.finditer(text)
        results: list[str] = []
        for match in matches:
            start_pos = match.end()
            brace_count = 1
            i = start_pos
            while i < len(text) and brace_count > 0:
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                i += 1
            if brace_count == 0:
                results.append(text[start_pos : i - 1])
        return results[0] if results else None

    # Fallback: last \boxed{} match (original behaviour)
    pattern = r"boxed\{"
    matches = list(re.finditer(pattern, model_output))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(model_output) and depth > 0:
        if model_output[i] == "{":
            depth += 1
        elif model_output[i] == "}":
            depth -= 1
        i += 1
    return model_output[start : i - 1].strip() if depth == 0 else None


def check_answer(predicted: str | None, expected: str) -> bool:
    """Compare predicted answer against expected.

    When the official PolyMath evaluation is available, delegates to
    ``math_equal`` which performs extensive normalisation, numeric
    comparison with tolerance, percentage equivalence, unit stripping,
    and symbolic equality via sympy.

    Otherwise falls back to ``math-verify`` with a simple string
    normalisation fallback.
    """
    if predicted is None:
        return False

    if _has_official:
        try:
            return _official_math_equal(predicted, expected)
        except Exception:
            logger.debug("official math_equal raised; falling back", exc_info=True)

    # Fallback: math-verify
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        gold = parse(expected, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        pred = parse(predicted, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        return verify(gold, pred)
    except Exception:
        pass

    # Last resort: normalised string comparison
    return _normalize(predicted) == _normalize(expected)


def _normalize(s: str) -> str:
    """Basic normalization for fallback string comparison."""
    s = s.strip()
    # Remove surrounding $...$
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Remove common LaTeX wrappers
    for cmd in (r"\text", r"\mathrm", r"\displaystyle"):
        s = s.replace(cmd, "")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s
