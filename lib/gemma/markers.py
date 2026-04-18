"""Special-token names used by the Gemma 4 hierarchical-CoT pipeline.

Native Gemma 4 tokens (already in the base tokenizer):
    THINK_TOKEN          - "<|think|>"   trigger for thinking mode (system block)
    CHANNEL_OPEN_TOKEN   - "<|channel>"  opens a thought channel
    CHANNEL_CLOSE_TOKEN  - "<channel|>"  closes a thought channel

Project-added token (registered as additional special token, embedding seeded
from a semantically related phrase — see `lib/gemma/setup.py`):
    RETURN_TOKEN         - "<return|>"   end of sub-chain-of-thought; pruner
                                         fires here at inference time.

Naming convention follows Gemma 4's own asymmetric delimiters: open delimiter
is "<|...>", close delimiter is "<...|>".  The new RETURN_TOKEN uses the close
form, since semantically it ends a region.
"""

THINK_TOKEN = "<|think|>"
CHANNEL_OPEN_TOKEN = "<|channel>"
CHANNEL_CLOSE_TOKEN = "<channel|>"
RETURN_TOKEN = "<return|>"

# Seed phrase used to initialize the new RETURN_TOKEN's row in both the main
# embedding table and the per-layer (PLE) embedding table.  Tokenized with the
# Gemma 4 SentencePiece vocab; the resulting sub-token rows are averaged.
RETURN_TOKEN_SEED = "end of subproblem return"
