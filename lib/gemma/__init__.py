"""Hierarchical-CoT data preparation for Gemma 4.

Historically this package also housed a PyTorch/HF model-side implementation
(pruner, SFT dataset wrappers, tokenizer+embedding seeding) — that path was
retired when the project moved to the Google DeepMind gemma JAX repo.  The
model-side code now lives in `lib/gemma_jax/`.

What remains here is the *model-agnostic* data pipeline:

    dataprep.sample     (deleted; will be rewritten against lib/gemma_jax
                         in Phase 2 of the plan)
    dataprep.segment    Gemini 3.x hierarchization of raw reasoning traces.
    dataprep.prepare    Orchestrator: stage-1 samples -> stage-2 hierarchical.
    dataprep.prompts    Prompt templates for the teacher LLM.
    dataprep.validate   Well-formedness checks on hierarchical traces.
    dataprep.collapse   Pure-regex `collapse_nested` (shared by HF and JAX).
"""
