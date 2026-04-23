"""Hierarchical-CoT sampler for Gemma 4 with KV-cache pruning on `<return|>`.

Drop-in replacement for `gm.text.Sampler` (text-only, batch=1) that watches
the generated token stream for hierarchical markers and rewinds the KV cache
when a sub-chain-of-thought completes:

    <|channel> ... thought ... <channel|> summary <return|>
    ^ open_pos                  ^ close_pos        ^ return emitted here

On `<return|>`, we:
  1. Rewind every layer's `end_index` back to `open_pos` (the integer
     position of the matching `<|channel>`).  KV entries at positions
     `[open_pos, end_index_before_prune)` remain physically resident in the
     ring buffer but become unreachable — the next write overwrites them,
     and the attention mask (derived from `positions[:end_index]`) no longer
     includes them.
  2. Re-forward the summary tokens through the pruned cache, at positions
     `[open_pos, open_pos + L_summary)`.  This rebuilds the KV so that the
     summary's attention is conditioned on `[prompt]` alone — not on the
     thought content.  The last step's logits are then used to sample the
     next token, preserving the normal autoregressive invariant.
  3. Collapse `predicted_tokens` so that the visible buffer matches the
     logical, post-prune sequence: `[prompt-era text, summary, next_token, 0,
     0, ...]`.  The pruned thought and markers disappear from the output.

When `enabled=False`, this sampler is a no-op pass-through to the JIT
`SamplerLoop._sample_loop`, producing byte-identical output to the stock
`gm.text.Sampler` for the same `(prompt, seed, sampling)` triple.  The
Phase-1 verification harness relies on this equivalence.

Limitations (Phase 1):
  * batch size must be 1.  Per-element surgery on a batched cache is
    mechanically possible (mask-the-other-elements) but not yet implemented.
  * text-only.  Vision / audio paths are out of scope here; use
    `gm.text.Gemma4Sampler` directly for those until Phase 2+.
"""
from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import random as py_random
from typing import Any

import einops
import jax
import jax.numpy as jnp
import numpy as np

from gemma import gm
from gemma.gm.data import _functional
from gemma.gm.nn import _transformer_like
from gemma.gm.text import _prefill
from gemma.gm.text import _sampler as _gemma_sampler
from gemma.gm.text import _sampler_loop
from gemma.gm.text import _sampling
from gemma.gm.text import _tokenizer as _gemma_tokenizer
from gemma.gm.typing import _common
from gemma.gm.utils import _types

from .setup import MarkerIds


@dataclasses.dataclass(frozen=True, kw_only=True)
class HierarchicalGemma4Sampler:
    """Text-only Gemma 4 sampler with hierarchical-CoT prune hooks.

    Attributes:
      model: the Gemma 4 transformer (e.g. `gm.nn.Gemma4_E4B()`).
      params: model parameters (loaded via `gm.ckpts.load_params`).
      markers: resolved integer IDs for `<|channel>`, `<channel|>`, `<return|>`.
      tokenizer: Gemma 4 tokenizer (must be the same one that produced
        `markers`; if omitted, a default `gm.text.Gemma4Tokenizer()` is used,
        but then `markers.return_` will not round-trip to a real piece).
      sampling: sampling method (default greedy).
      forbidden_tokens: tokens forbidden from generation.
      stop_tokens: tokens that terminate generation.
      cache_length: KV-cache capacity.
      max_out_length: output buffer size.
      pad_length: prompt-padding buckets for JIT cache reuse.
      enabled: when False, pruning is off and this sampler is bit-equivalent
        to the stock `gm.text.Sampler`.
    """

    model: _transformer_like.TransformerLike
    params: _common.Params
    markers: MarkerIds
    tokenizer: _gemma_tokenizer.Tokenizer | None = None
    sampling: _sampling.SamplingMethod = dataclasses.field(
        default_factory=_sampling.Greedy
    )
    forbidden_tokens: Sequence[str | int] | None = None
    stop_tokens: Sequence[str | int] | None = None
    cache_length: int = 4096
    max_out_length: int = 2048
    pad_length: None | int | tuple[int, ...] = (256, 512, 1024)
    enabled: bool = True

    def __post_init__(self):
        if self.tokenizer is None:
            if not self.model.INFO.tokenizer_version:
                raise ValueError(
                    "Model does not specify a tokenizer version; pass "
                    "`tokenizer` explicitly."
                )
            object.__setattr__(
                self,
                "tokenizer",
                _gemma_tokenizer.Tokenizer.from_version(
                    self.model.INFO.tokenizer_version
                ),
            )

    def sample(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        rng: int | jax.Array | None = None,
        sampling: _sampling.SamplingMethod | None = None,
        return_state: bool = False,
    ) -> str | _gemma_sampler.SamplerOutput:
        """Sample text from a single string prompt (batch=1)."""
        if not isinstance(prompt, str):
            raise TypeError(
                f"HierarchicalGemma4Sampler.sample expects a single str "
                f"prompt (Phase 1 is batch=1); got {type(prompt).__name__}"
            )

        sampling = sampling or self.sampling
        rng = _gemma_sampler._normalize_rng(rng)  # noqa: SLF001

        tokens = self.tokenizer.encode(prompt, add_bos=True)
        max_prompt_len = len(tokens)
        padded = _functional.pad([tokens], max_length=max_prompt_len)
        text = jnp.asarray(padded)

        inputs = _types.Input(
            text=text,
            images=None,
            config=self.model.config.input_config,
        )

        init_state = _prefill.prefill(
            model=self.model,
            params=self.params,
            input=inputs,
            last_state=None,
            cache_length=self.cache_length,
            pad_length=self.pad_length,
            rng=rng,
            sharding=None,
            max_out_length=self.max_out_length,
        )

        if max_new_tokens and max_new_tokens > self.max_out_length:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} > max_out_length="
                f"{self.max_out_length}"
            )
        max_new_tokens = max_new_tokens or self.max_out_length
        max_new_tokens_arr = jnp.asarray(max_new_tokens)

        sampler_loop = _sampler_loop.SamplerLoop(
            model=self.model,
            end_tokens=(
                self.tokenizer.special_tokens.EOS,
                self.tokenizer.special_tokens.END_OF_TURN,
                self.tokenizer.special_tokens.BEGIN_OF_TOOL_RESPONSE,
                *self._normalized_stop_tokens,
            ),
            forbidden_tokens=self._normalized_forbidden_tokens,
            sampling=sampling,
            cache_length=self.cache_length,
            special_tokens=self.tokenizer.special_tokens,
        )

        if not self.enabled:
            # Exact passthrough: same JIT path as the stock sampler.
            state = sampler_loop.sample(
                params=self.params,
                init_state=init_state,
                max_new_tokens=max_new_tokens_arr,
                stream=False,
            )
        else:
            state = self._run_with_prune(
                sampler_loop=sampler_loop,
                init_state=init_state,
                max_new_tokens=int(max_new_tokens),
                sampling=sampling,
            )

        tokens_out = state.predicted_tokens[0]
        text_out = self.tokenizer.decode(tokens_out)
        if return_state:
            return _gemma_sampler.SamplerOutput(text=text_out, state=state)
        return text_out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_with_prune(
        self,
        *,
        sampler_loop: _sampler_loop.SamplerLoop,
        init_state: _sampler_loop.SamplingState,
        max_new_tokens: int,
        sampling: _sampling.SamplingMethod,
    ) -> _sampler_loop.SamplingState:
        """Streaming decode with a Python-side channel stack + prune hooks.

        Returns the final `SamplingState` after masking tokens after end
        tokens (mirroring `_sample_loop`'s post-processing).
        """
        state = init_state
        # Per-batch channel stack: list of {'open_step', 'close_step'}
        # tracking the step indices (into predicted_tokens) where events fired.
        stacks: list[list[dict[str, int]]] = [
            [] for _ in range(int(state.last_token.shape[0]))
        ]
        init_cache_length = int(state.init_cache_length)

        steps_done = 0
        while steps_done < max_new_tokens:
            if bool(jnp.all(state.done)) or bool(state.cache_info.is_full):
                break
            state = sampler_loop._sample_step(  # noqa: SLF001
                state=state,
                params=self.params,
            )
            steps_done += 1
            step_idx = int(state.step) - 1  # index of token just written

            # Batch=1 is enforced at sample() entry; index b=0 throughout.
            b = 0
            tok = int(state.last_token[b])
            done_b = bool(state.done[b])
            if done_b:
                continue

            if tok == self.markers.channel_open:
                stacks[b].append({"open_step": step_idx, "close_step": -1})
            elif (
                tok == self.markers.channel_close
                and stacks[b]
                and stacks[b][-1]["close_step"] == -1
            ):
                stacks[b][-1]["close_step"] = step_idx
            elif (
                tok == self.markers.return_
                and stacks[b]
                and stacks[b][-1]["close_step"] >= 0
            ):
                frame = stacks[b].pop()
                state = self._prune_and_continue(
                    state=state,
                    sampler_loop=sampler_loop,
                    sampling=sampling,
                    open_step=frame["open_step"],
                    close_step=frame["close_step"],
                    return_step=step_idx,
                    init_cache_length=init_cache_length,
                )

        # Mirror _sample_loop's post-processing: mask tokens after end tokens.
        predicted_tokens = _mask_tokens_after_end_tokens(
            state.predicted_tokens,
            end_tokens=sampler_loop.end_tokens,
        )
        return dataclasses.replace(state, predicted_tokens=predicted_tokens)

    def _prune_and_continue(
        self,
        *,
        state: _sampler_loop.SamplingState,
        sampler_loop: _sampler_loop.SamplerLoop,
        sampling: _sampling.SamplingMethod,
        open_step: int,
        close_step: int,
        return_step: int,
        init_cache_length: int,
    ) -> _sampler_loop.SamplingState:
        """Rewind cache to `open_pos`, re-forward the summary, sample next token.

        Summary is `predicted_tokens[0, close_step+1 : return_step]`.  After
        this call, matching the stock `_sample_step` post-state invariants:
          * `cache.end_index == open_pos + L_summary`.
          * `last_token == next_token`, `last_token_pos == open_pos + L_summary`
            — i.e. `next_token`'s KV has *not* yet been written; the next
            `_sample_step` will write it at `cache.end_index`.
          * `predicted_tokens[0]` is `[..., summary..., next_token, 0, 0, ...]`.
          * `step == open_step + L_summary + 1`.
        """
        open_pos = init_cache_length + open_step
        summary = np.asarray(
            state.predicted_tokens[0, close_step + 1 : return_step]
        )
        l_sum = int(summary.shape[0])
        if l_sum == 0:
            # Malformed event — empty summary.  Skip the prune; keep going.
            return state

        # 1. Rewind every layer's end_index to open_pos.
        pruned_cache = state.cache_info.set_end_index(
            jnp.asarray(open_pos, dtype=jnp.int32)
        ).cache

        # 2. Re-forward the summary one token at a time into the pruned cache.
        cache = pruned_cache
        full_attn = state.full_attention_mask
        cache_len = full_attn.shape[-1]
        last_logits = None
        for i, tok in enumerate(summary.tolist()):
            pos = open_pos + i
            step_mask = jnp.arange(cache_len) < (pos + 1)
            attn_mask = (full_attn * step_mask)[:, None, :]
            out = self.model.apply(
                {"params": self.params},
                tokens=jnp.asarray([[tok]], dtype=jnp.int32),
                cache=cache,
                positions=jnp.asarray([[pos]], dtype=jnp.int32),
                attention_mask=attn_mask,
            )
            cache = out.cache
            last_logits = out.logits  # [1, 1, V]

        # 3. Sample next_token from the post-summary logits.  Note that
        # `cache` has already been advanced to end_index = open_pos + L_s by
        # the re-forward loop above; `next_token`'s KV is *not* yet written.
        # This mirrors the stock invariant after `_sample_step`:
        # `cache.end_index == last_token_pos`, and the next `_sample_step`
        # is responsible for writing `last_token`'s KV at that index.
        logits = einops.rearrange(last_logits, "B 1 V -> B V")
        if sampler_loop.forbidden_tokens:
            logits = logits.at[:, sampler_loop.forbidden_tokens].set(-jnp.inf)
        next_rng, curr_rng = jax.random.split(state.rng)
        next_token = sampling.get_next_tokens(logits, rng=curr_rng)  # [B]

        # 4. Collapse predicted_tokens: keep [:, :open_step], then summary, then
        # next_token, then zeros.  The thought and markers disappear.
        new_pred = jnp.zeros_like(state.predicted_tokens)
        new_pred = new_pred.at[:, :open_step].set(
            state.predicted_tokens[:, :open_step]
        )
        new_pred = new_pred.at[0, open_step : open_step + l_sum].set(
            jnp.asarray(summary, dtype=new_pred.dtype)
        )
        new_pred = new_pred.at[:, open_step + l_sum].set(next_token)

        new_step = open_step + l_sum + 1
        new_last_token_pos = jnp.asarray(
            [open_pos + l_sum], dtype=jnp.int32
        )
        done = state.done | jnp.isin(
            next_token, jnp.asarray(sampler_loop.end_tokens)
        )

        return dataclasses.replace(
            state,
            step=jnp.asarray(new_step, dtype=state.step.dtype),
            done=done,
            last_token=next_token,
            last_token_pos=new_last_token_pos,
            cache=cache,
            predicted_tokens=new_pred,
            rng=next_rng,
        )

    @functools.cached_property
    def _normalized_forbidden_tokens(self) -> tuple[int, ...] | None:
        forbidden = _normalize_tokens(self.tokenizer, self.forbidden_tokens)
        forbidden += self.tokenizer.FORBIDDEN_TOKENS
        return forbidden

    @functools.cached_property
    def _normalized_stop_tokens(self) -> tuple[int, ...]:
        return _normalize_tokens(self.tokenizer, self.stop_tokens)


def _normalize_tokens(
    tokenizer: _gemma_tokenizer.Tokenizer,
    tokens: Sequence[str | int] | None,
) -> tuple[int, ...]:
    if tokens is None:
        return ()

    def _one(t: str | int) -> int:
        if isinstance(t, int):
            return t
        ids = tokenizer.encode(t)
        if len(ids) != 1:
            raise ValueError(
                f"Token {t!r} must map to a single id; got {ids}"
            )
        return ids[0]

    return tuple(_one(t) for t in tokens)


def _mask_tokens_after_end_tokens(
    tokens: jax.Array,
    *,
    end_tokens: tuple[int, ...],
) -> jax.Array:
    """Re-implementation of `_sampler_loop._mask_tokens_after_end_tokens`.

    The upstream is module-private; we inline it to keep the streaming path's
    post-processing consistent with the JIT path.
    """
    end_tokens_mask = jnp.isin(tokens, jnp.asarray(end_tokens))
    end_tokens_mask = jnp.cumsum(end_tokens_mask, axis=-1) - end_tokens_mask == 0
    return tokens * end_tokens_mask
