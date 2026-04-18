"""Gemma 4 data-preparation pipeline.

Two-stage flow (to be implemented):
    1. sample.py   - sample prompts from an open-reasoning dataset, run them
                     through Gemma 4 to collect baseline thinking traces.
    2. segment.py  - call a teacher LLM (Gemini 3.x) to annotate each trace
                     with nested <|channel>thought ... <channel|><return|>
                     boundaries; emit hierarchical SFT records.
"""
