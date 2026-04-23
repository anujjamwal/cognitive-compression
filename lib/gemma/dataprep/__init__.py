"""Gemma 4 data-preparation pipeline.

Two-stage flow:
    1. sample   - sample prompts from an open-reasoning dataset and run them
                  through a base Gemma 4 sampler to collect raw reasoning
                  traces.  The HF/torch implementation was removed when the
                  project moved to JAX; the Phase-2 replacement will live
                  in lib/gemma_jax/ and drive `HierarchicalGemma4Sampler`.
    2. segment  - call a teacher LLM (Gemini 3.x) to annotate each trace
                  with nested <|channel>thought ... <channel|><return|>
                  boundaries; emit hierarchical SFT records.

Supporting modules:
    prompts     Gemini persona + task prompt + input template.
    validate    Structural well-formedness checks.
    collapse    Pure-regex `collapse_nested` helper (no ML deps).
    prepare     Orchestrator: stage-1 raw -> stage-2 hierarchical.
"""
from . import collapse, prompts, segment, validate

__all__ = ["collapse", "prompts", "segment", "validate"]
