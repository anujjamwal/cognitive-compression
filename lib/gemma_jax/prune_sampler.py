"""Hierarchical-CoT chat sampler for Gemma 4 with KV-cache pruning on `<return|>`.

`PruningChatSampler` subclasses `gm.text.ChatSampler` so the user-facing
API is identical: stateful multi-turn chat, transparent prompt formatting,
auto-detection of the underlying model.  The only difference is what
happens *during* a turn: the inner sampling loop watches for the
hierarchical markers

    <|channel> ... thought ... <channel|> summary <return|>
    ^ open_pos                  ^ close_pos        ^ return emitted here

and on every `<return|>` rewinds the KV cache back to the matching
`<|channel>` position, re-forwards the summary, and continues.

Mechanism (per prune event):
  1. Rewind every layer's `end_index` to `open_pos` via
     `state.cache_info.set_end_index`.  KV entries at positions
     `[open_pos, end_index_before_prune)` remain in the ring buffer but
     are no longer addressable; the next write overwrites them.
  2. Re-forward the summary tokens one at a time into the rewound cache,
     at positions `[open_pos, open_pos + L_summary)`.  The summary's
     attention is now conditioned on `[prompt]` only — not on the thought
     content that just got pruned.
  3. Sample `next_token` from the post-summary logits.  Preserve the
     stock invariant `cache.end_index == state.last_token_pos`, i.e.
     `next_token`'s KV is *not* yet written; the next `_sample_step`
     writes it at `cache.end_index = open_pos + L_summary`.
  4. Collapse `predicted_tokens` so the visible buffer is
     `[..., summary, next_token, 0, ...]`.  The pruned thought, both
     channel markers, and the `<return|>` itself disappear from the
     output.

Outer loop: segmented JIT.  We add `<return|>` to the `SamplerLoop`'s
`end_tokens`, run the JIT-compiled `_sample_loop` until any end token
fires, then check `state.last_token`:
  * `<return|>`     → reconstruct the open/close positions by scanning
                      `predicted_tokens[:state.step]`, prune, reset
                      `state.done`, loop.
  * other end token → exit; `ChatSampler` records the turn.

Detecting the channel markers is done lazily at prune time (one Python
pass over the just-emitted tokens) rather than per-step, so the
JIT-fused `while_loop` runs the long stretches between prune events at
full speed.

When `pruning_enabled=False`, `chat()` delegates straight to
`super().chat(...)` and produces output byte-identical to stock
`ChatSampler`.

Limitations (Phase 1):
  * Gemma 4 only.  Other versions raise.
  * Batch size 1.  Per-element cache surgery on a batched cache is
    mechanically possible but not implemented.
  * Text-only.  Image/audio inputs go through `super().chat()` (i.e.
    pruning is silently disabled for multimodal turns).
"""
from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools

import dialog
import einops
import jax
import jax.numpy as jnp
import numpy as np

from gemma import gm
from gemma.gm.data import _functional
from gemma.gm.text import _chat_sampler
from gemma.gm.text import _prefill
from gemma.gm.text import _sampler as _gemma_sampler
from gemma.gm.text import _sampler_loop
from gemma.gm.text import _sampling
from gemma.gm.text import _template
from gemma.gm.text import _tokenizer as _gemma_tokenizer
from gemma.gm.utils import _types

from .setup import MarkerIds


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class PruningChatSampler(_chat_sampler.ChatSampler):
    """`gm.text.ChatSampler` subclass with `<return|>`-triggered KV pruning.

    Additional attributes (on top of `ChatSampler`):
      markers:           resolved integer IDs for `<|channel>`, `<channel|>`,
                         `<return|>` (built via
                         `lib.gemma_jax.setup.resolve_marker_ids`).
      pruning_enabled:   when `False`, `chat()` delegates straight to
                         `super().chat(...)` — byte-identical to stock.

    Phase 1 is text-only Gemma 4 batch=1.
    """

    markers: MarkerIds = None  # type: ignore[assignment]
    pruning_enabled: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.markers is None:
            raise ValueError(
                "PruningChatSampler requires `markers` (use "
                "lib.gemma_jax.setup.resolve_marker_ids)."
            )
        if not self._is_gemma4:
            raise ValueError(
                "PruningChatSampler only supports Gemma 4 models."
            )

    # ------------------------------------------------------------------
    # Public API: ChatSampler.chat(...) override
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt,
        *,
        images=None,
        audio=None,
        audio_lengths=None,
        sampling=None,
        rng=None,
        max_new_tokens=None,
        multi_turn=None,
        print_stream=None,
        is_legacy_tool_answer=False,
        sharding=None,
    ):
        # Multimodal / pruning-off paths: defer to stock ChatSampler.
        if (
            not self.pruning_enabled
            or images is not None
            or audio is not None
        ):
            return super().chat(
                prompt,
                images=images,
                audio=audio,
                audio_lengths=audio_lengths,
                sampling=sampling,
                rng=rng,
                max_new_tokens=max_new_tokens,
                multi_turn=multi_turn,
                print_stream=print_stream,
                is_legacy_tool_answer=is_legacy_tool_answer,
                sharding=sharding,
            )

        if multi_turn is None:
            multi_turn = self.multi_turn
        if not multi_turn:
            object.__setattr__(self, "last_state", None)
            object.__setattr__(self, "turns", [])

        sampling = sampling or self.sampling
        rng_key = _gemma_sampler._normalize_rng(rng)  # noqa: SLF001

        # Mirror ChatSampler's prompt normalization (text-only branch).
        if isinstance(prompt, str):
            prompt = dialog.Conversation(dialog.User(prompt))
        elif not isinstance(prompt, dialog.Conversation):
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        prompt_text = prompt.as_text(format=self.tokenizer.FORMAT)

        # Tokenize + prefill (Gemma 4 text-only path, batch=1).
        last_state = self.last_state
        token_ids = self.tokenizer.encode(prompt_text, add_bos=last_state is None)
        padded = _functional.pad([token_ids], max_length=len(token_ids))
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
            last_state=last_state,
            cache_length=self.cache_length,
            pad_length=self.pad_length,
            rng=rng_key,
            sharding=sharding,
            max_out_length=self.max_out_length,
        )

        if max_new_tokens and max_new_tokens > self.max_out_length:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} > max_out_length="
                f"{self.max_out_length}"
            )
        max_new_tokens_arr = jnp.asarray(
            max_new_tokens or self.max_out_length
        )

        # Build the SamplerLoop with `<return|>` added to end_tokens so the
        # JIT loop terminates at every prune event in addition to the real
        # terminators.  We also keep a `real_end_tokens` tuple (= end_tokens
        # minus `<return|>`) for post-prune "is this token actually
        # terminating?" checks.
        real_end_tokens = (
            self.tokenizer.special_tokens.EOS,
            self.tokenizer.special_tokens.END_OF_TURN,
            self.tokenizer.special_tokens.BEGIN_OF_TOOL_RESPONSE,
            *self._normalized_stop_tokens,
        )
        sampler_loop = _sampler_loop.SamplerLoop(
            model=self.model,
            end_tokens=real_end_tokens + (self.markers.return_,),
            forbidden_tokens=self._normalized_forbidden_tokens,
            sampling=sampling,
            cache_length=self.cache_length,
            special_tokens=self.tokenizer.special_tokens,
        )

        # Segmented loop: JIT _sample_loop until any end_token fires; on
        # `<return|>` apply the prune and resume; otherwise we're done.
        state = self._run_segmented_with_prune(
            sampler_loop=sampler_loop,
            init_state=init_state,
            max_new_tokens_arr=max_new_tokens_arr,
            sampling=sampling,
            real_end_tokens=real_end_tokens,
        )

        # Decode + bookkeeping (mirror ChatSampler.chat tail).
        text_out = self.tokenizer.decode(state.predicted_tokens[0])
        self.turns.append(_template.Prompt(prompt_text))
        self.turns.append(_template.Response(text_out))
        object.__setattr__(self, "last_state", state)
        return text_out

    # ------------------------------------------------------------------
    # Segmented JIT loop + prune bookkeeping
    # ------------------------------------------------------------------

    def _run_segmented_with_prune(
        self,
        *,
        sampler_loop: _sampler_loop.SamplerLoop,
        init_state: _sampler_loop.SamplingState,
        max_new_tokens_arr: jax.Array,
        sampling: _sampling.SamplingMethod,
        real_end_tokens: tuple[int, ...],
    ) -> _sampler_loop.SamplingState:
        """Drive the JIT `_sample_loop` in segments separated by prune events."""
        state = init_state
        init_cache_length = int(state.init_cache_length)

        while True:
            # JIT segment: runs until any token in `end_tokens` fires (or
            # max_new_tokens / cache full).
            state = sampler_loop._sample_loop(  # noqa: SLF001
                params=self.params,
                state=state,
                max_new_tokens=max_new_tokens_arr,
            )

            # Cache-full or budget exhausted with no terminator: just exit.
            if (
                int(state.step) >= int(max_new_tokens_arr)
                or bool(state.cache_info.is_full)
            ):
                break

            # Why did we stop?  Inspect last_token.
            last_tok = int(state.last_token[0])
            if last_tok in real_end_tokens:
                break  # Real terminator (EOS / EOT / ...).
            if last_tok != self.markers.return_:
                # Defensive: shouldn't happen — `_sample_loop` exited but the
                # last token is neither a real end nor `<return|>`.  Stop.
                break

            # Prune event.  Reconstruct open/close positions from history
            # (the just-emitted segment is in predicted_tokens[:state.step]).
            event = self._find_matching_channel_block(
                predicted=state.predicted_tokens[0],
                step=int(state.step),
            )
            if event is None:
                # Malformed `<return|>` (no matching <|channel> .. <channel|>).
                # Treat it as a no-op terminator: reset done so the loop can
                # continue past this token, do not prune.
                state = dataclasses.replace(
                    state, done=jnp.zeros_like(state.done)
                )
                continue

            open_step, close_step = event
            return_step = int(state.step) - 1
            state = self._apply_prune(
                state=state,
                sampler_loop=sampler_loop,
                sampling=sampling,
                real_end_tokens=real_end_tokens,
                open_step=open_step,
                close_step=close_step,
                return_step=return_step,
                init_cache_length=init_cache_length,
            )

            # If the post-prune `next_token` is itself a real terminator,
            # `state.done` is already True and the next `_sample_loop` call
            # will exit immediately (cond_fn short-circuits).  Otherwise the
            # next iteration continues normal generation.

        return state

    def _find_matching_channel_block(
        self,
        *,
        predicted: jax.Array,
        step: int,
    ) -> tuple[int, int] | None:
        """Walk `predicted[:step]` once with a channel-balance stack and
        return the `(open_step, close_step)` for the most recently
        emitted `<return|>` (which is at `predicted[step - 1]`).

        Returns None if the token at `step - 1` is `<return|>` but no
        well-formed matching `<|channel> ... <channel|>` precedes it.
        """
        seq = np.asarray(predicted[:step])
        return_step = step - 1
        if seq[return_step] != self.markers.return_:
            return None

        # Stack of [open_step, close_step] frames.  An entry's close_step
        # stays -1 until the matching <channel|> is seen.  On `<return|>`,
        # the top frame must already have a close_step >= 0 to be valid.
        stack: list[list[int]] = []
        for i in range(step):
            tok = int(seq[i])
            if tok == self.markers.channel_open:
                stack.append([i, -1])
            elif (
                tok == self.markers.channel_close
                and stack
                and stack[-1][1] == -1
            ):
                stack[-1][1] = i
            elif tok == self.markers.return_ and stack:
                if i == return_step:
                    frame = stack[-1]
                    if frame[1] < 0:
                        return None  # No matching <channel|>.
                    return (frame[0], frame[1])
                # Earlier <return|>: pop and continue scanning.
                if stack[-1][1] >= 0:
                    stack.pop()
        return None

    def _apply_prune(
        self,
        *,
        state: _sampler_loop.SamplingState,
        sampler_loop: _sampler_loop.SamplerLoop,
        sampling: _sampling.SamplingMethod,
        real_end_tokens: tuple[int, ...],
        open_step: int,
        close_step: int,
        return_step: int,
        init_cache_length: int,
    ) -> _sampler_loop.SamplingState:
        """Rewind cache to the `<|channel>` position, re-forward the summary,
        sample the next token, and produce a state ready for the next JIT
        sampling segment.

        Post-conditions (matching the stock `_sample_step` invariants):
          * `cache.end_index == open_pos + L_summary`.
          * `last_token == next_token`, `last_token_pos == open_pos + L_summary`
            — `next_token`'s KV is *not* yet written.
          * `predicted_tokens[0]` is `[..., summary..., next_token, 0, ...]`.
          * `step == open_step + L_summary + 1`.
          * `done = (next_token in real_end_tokens)` — `<return|>` is not a
            terminator after a prune (would mean immediate empty channel).
        """
        open_pos = init_cache_length + open_step
        summary = np.asarray(
            state.predicted_tokens[0, close_step + 1 : return_step]
        )
        l_sum = int(summary.shape[0])
        if l_sum == 0:
            # Empty summary — nothing to re-forward.  Treat the `<return|>`
            # as a no-op: drop it from the buffer, reset done, continue.
            new_pred = state.predicted_tokens.at[0, return_step].set(0)
            return dataclasses.replace(
                state,
                done=jnp.zeros_like(state.done),
                predicted_tokens=new_pred,
            )

        # 1. Rewind every layer's end_index to open_pos.
        cache = state.cache_info.set_end_index(
            jnp.asarray(open_pos, dtype=jnp.int32)
        ).cache

        # 2. Re-forward the summary one token at a time.
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

        # 3. Sample next_token from the post-summary logits.
        logits = einops.rearrange(last_logits, "B 1 V -> B V")
        if sampler_loop.forbidden_tokens:
            logits = logits.at[:, sampler_loop.forbidden_tokens].set(-jnp.inf)
        next_rng, curr_rng = jax.random.split(state.rng)
        next_token = sampling.get_next_tokens(logits, rng=curr_rng)  # [B]

        # 4. Collapse predicted_tokens.
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
        # Reset done relative to the freshly-sampled next_token.  Crucially
        # `<return|>` is NOT in real_end_tokens, so a malformed back-to-back
        # `<return|>` after a prune would not terminate generation here.
        done = jnp.isin(next_token, jnp.asarray(real_end_tokens))

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
