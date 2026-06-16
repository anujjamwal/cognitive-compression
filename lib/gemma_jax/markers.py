"""Special-token names used by the Gemma 4 hierarchical-CoT pipeline (JAX path).

Mirrors the string constants from `lib/gemma/markers.py` so dataprep records
produced for the HF branch are byte-compatible with this one.

Native Gemma 4 tokens (already in the base tokenizer):
    THINK_TOKEN          - "<|think|>"
    CHANNEL_OPEN_TOKEN   - "<|channel>"  opens a thought channel
    CHANNEL_CLOSE_TOKEN  - "<channel|>"  closes a thought channel

Project-added token (registered into an unused <unusedN> vocab slot):
    RETURN_TOKEN         - "<return|>"   end of sub-CoT; the pruner fires here
                                         at inference time.
"""

THINK_TOKEN = "<|think|>"
CHANNEL_OPEN_TOKEN = "<|channel>"
CHANNEL_CLOSE_TOKEN = "<channel|>"
RETURN_TOKEN = "<return|>"

# Default SentencePiece <unusedN> slot index to repurpose for `<return|>`.
# Concretely, slot `N` corresponds to piece id `special_tokens.CUSTOM + N` in
# the Gemma tokenizer (Gemma 3: CUSTOM=6 so `<unused0>`=id 6).  Gemma 4 follows
# the same layout in the shipped SentencePiece binary.
DEFAULT_RETURN_TOKEN_SLOT = 0
