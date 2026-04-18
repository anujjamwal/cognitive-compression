"""Gemma 4 data-preparation pipeline.

Two-stage flow:
    1. sample.py   - sample prompts from an open-reasoning dataset, run them
                     through Gemma 4 to collect baseline thinking traces.
    2. segment.py  - call a teacher LLM (Gemini 3.x) to annotate each trace
                     with nested <|channel>thought ... <channel|><return|>
                     boundaries; emit hierarchical SFT records.

Supporting modules:
    prompts.py  - Gemini persona + task prompt + input template
    validate.py - structural well-formedness checks
    prepare.py  - orchestrator: stage-1 dataset -> stage-2 hierarchical dataset
"""
from . import prompts, sample, segment, validate

__all__ = ["prompts", "sample", "segment", "validate"]

