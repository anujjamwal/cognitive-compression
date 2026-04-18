"""Prune-aware generation for Gemma 4 with hierarchical CoT.

Marker semantics:
    <|channel>     opens a thought channel        (THOUGHT)
    <channel|>     closes a thought channel       (SOLUTION boundary)
    <return|>      end of sub-chain-of-thought    (prune trigger)

When `<return|>` is emitted, the pruner walks the stack to the matching
`<|channel>` open and rewrites the visible sequence to drop ALL THREE
markers along with the thought content:

    before:  [..prefix..] <|channel> <thought> <channel|> <summary> <return|>
    after:   [..prefix..] <summary>

The KV cache is truncated to the prefix length (excluding `<|channel>`), and
the next forward pass re-RoPEs the summary tokens at renumbered positions
`[open_pos, open_pos+1, ...]`.

Strategy 1A: prune every layer's cache to the prefix length.  Per-layer KV
shape and head dim are read inside the loop because Gemma 4 sliding layers
use head_dim=256 while global layers use head_dim=512.  Layers in the
KV-shared tail (last `config.num_kv_shared_layers`) may have no own cache
entry — we skip them.

Sliding-layer coherence: the prefix's cached K/V keeps its original RoPE
positions, which are unchanged by the renumber (we renumber only positions
at and after the prune point).  The next forward pass appends re-processed
summary tokens with positions `[open_pos, open_pos+1, ...]`, monotonically
continuing the prefix.  The sliding window's eviction logic then trims
naturally.

Training-data invariant for this scheme: post-prune stages must show the
model `[prefix, summary]` without any of the three markers, so it learns to
continue from the summary as if it were plain text — and in particular,
never predicts `<return|>` as the next token from a fresh post-prune
context (which would be ill-formed since there is no matching `<|channel>`
in the visible context).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import Cache, DynamicCache, LogitsProcessorList, StoppingCriteriaList
from transformers import PreTrainedTokenizerBase
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    ALL_CACHE_NAMES,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerationMixin,
)
from transformers.utils.generic import ModelOutput

from .markers import CHANNEL_CLOSE_TOKEN, CHANNEL_OPEN_TOKEN, RETURN_TOKEN


# ---------------------------------------------------------------------------
# Layer-aware KV-cache truncation (1A)
# ---------------------------------------------------------------------------

def _truncate_kv_cache_layer_aware(
    cache: DynamicCache,
    prune_map: dict[int, Tuple[int, int, int]],
    batch_size: int,
    old_seq_len: int,
) -> int:
    """Truncate a hybrid KV cache to per-element prefix lengths.

    For each batch element in `prune_map`, retain only `[0..open_pos)` of the
    cached K/V at every layer (i.e. drop `<|channel>` itself along with
    everything after it).  Non-pruned elements keep the full cache.  Per-
    layer head_dim is read from each layer's tensor (Gemma 4: sliding=256,
    global=512).  Layers without an own cache entry (KV-shared tail) are
    skipped silently.

    `prune_map` values are `(open_pos, close_pos, return_pos)`; only
    `open_pos` is used here.

    Returns the new sequence length (max prefix length across the batch).
    """
    # Resolve the API: transformers ≥5 exposes cache.layers; older uses key_cache.
    use_layers_api = hasattr(cache, "layers")
    if use_layers_api:
        num_layers = len(cache.layers)
    elif hasattr(cache, "key_cache"):
        num_layers = len(cache.key_cache)  # type: ignore[attr-defined]
    else:
        return 0

    if num_layers == 0:
        return 0

    # Per-element prefix lengths.  Pruned -> open_pos (drop the `<|channel>`
    # token itself along with everything after); otherwise full length.
    prefix_lengths: list[int] = []
    for b in range(batch_size):
        if b in prune_map:
            open_pos, _close_pos, _return_pos = prune_map[b]
            prefix_lengths.append(open_pos)
        else:
            prefix_lengths.append(old_seq_len)

    max_new_seq = max(prefix_lengths)
    all_same_prefix = len(set(prefix_lengths)) == 1

    for layer_idx in range(num_layers):
        if use_layers_api:
            layer = cache.layers[layer_idx]
            old_keys = getattr(layer, "keys", None)
            old_vals = getattr(layer, "values", None)
        else:
            kc = cache.key_cache  # type: ignore[attr-defined]
            vc = cache.value_cache  # type: ignore[attr-defined]
            old_keys = kc[layer_idx] if layer_idx < len(kc) else None
            old_vals = vc[layer_idx] if layer_idx < len(vc) else None

        # KV-shared layers in Gemma 4 may carry no own cache state — skip them.
        if old_keys is None or old_vals is None or old_keys.numel() == 0:
            continue

        # Read head_dim *per layer* — Gemma 4 sliding vs global differ (256/512).
        head_dim = old_keys.shape[-1]
        device = old_keys.device

        if all_same_prefix:
            new_keys = old_keys[:, :, :prefix_lengths[0], :].contiguous()
            new_vals = old_vals[:, :, :prefix_lengths[0], :].contiguous()
        else:
            num_heads = old_keys.shape[1]
            idx = torch.zeros(batch_size, max_new_seq, dtype=torch.long, device=device)
            for b in range(batch_size):
                n = prefix_lengths[b]
                idx[b, :n] = torch.arange(n, device=device)
            idx_expanded = idx[:, None, :, None].expand(-1, num_heads, -1, head_dim)
            new_keys = torch.gather(old_keys, 2, idx_expanded)
            new_vals = torch.gather(old_vals, 2, idx_expanded)
            for b in range(batch_size):
                n = prefix_lengths[b]
                if n < max_new_seq:
                    new_keys[b, :, n:, :] = 0
                    new_vals[b, :, n:, :] = 0

        if use_layers_api:
            layer.keys = new_keys
            layer.values = new_vals
            # DynamicSlidingWindowLayer tracks a `cumulative_length` used by
            # its causal-mask helper; keep it in sync with the truncated cache.
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = max_new_seq
        else:
            cache.key_cache[layer_idx] = new_keys      # type: ignore[attr-defined]
            cache.value_cache[layer_idx] = new_vals    # type: ignore[attr-defined]

    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = max_new_seq
    return max_new_seq


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _prepare_inputs_for_generation(
    model,
    input_ids: torch.LongTensor,
    past_key_values: Cache | None = None,
    attention_mask: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    cache_position: torch.LongTensor | None = None,
    is_first_iteration: bool | None = False,
    **kwargs,
):
    """Slice input_ids to the tokens at `cache_position`.

    After a prune, `cache_position` covers only the post-`<channel|>` tokens
    that need re-processing (fewer than input_ids).  HF's standard helper
    expects them to match, so pre-slice here.
    """
    if (
        cache_position is not None
        and past_key_values is not None
        and input_ids.shape[1] != cache_position.shape[0]
    ):
        input_ids = input_ids[:, cache_position]

    return GenerationMixin.prepare_inputs_for_generation(
        model,
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        is_first_iteration=is_first_iteration,
        **kwargs,
    )


def _update_model_kwargs_for_generation(
    model,
    outputs: ModelOutput,
    model_kwargs: dict[str, Any],
    is_encoder_decoder: bool = False,
    num_new_tokens: int = 1,
) -> dict[str, Any]:
    model_kwargs = GenerationMixin._update_model_kwargs_for_generation(
        model,
        outputs,
        model_kwargs,
        is_encoder_decoder=is_encoder_decoder,
        num_new_tokens=num_new_tokens,
    )
    # After a prune the multi-element `cache_position` from `_prune_model_inputs`
    # would otherwise get carried forward and cause re-selection of cached tokens
    # on every subsequent step; collapse to last element (the next decode pos).
    if model_kwargs.get("cache_position") is not None:
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:]
    return model_kwargs


def _prune_model_inputs(
    model,
    prune_input_candidates: Sequence[int],
    prune_input_locations: Sequence[Sequence[Tuple[int, int, int]]],
    input_ids: torch.LongTensor,
    model_kwargs: dict[str, Any],
) -> Tuple[torch.LongTensor, dict[str, Any]]:
    """Prune-aware: drop `<|channel>`, the thought content, `<channel|>`, and
    `<return|>` from input_ids; truncate KV cache to the prefix that
    immediately precedes `<|channel>`; set `cache_position` so the next
    forward re-processes the summary tokens against the cached prefix.
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device

    # batch index -> (open_pos, close_pos, return_pos)
    # Only entries with a matched close are eligible.
    prune_map: dict[int, Tuple[int, int, int]] = {}
    for cand_idx, batch_idx in enumerate(prune_input_candidates):
        for open_pos, close_pos, return_pos in prune_input_locations[cand_idx]:
            if close_pos is not None and return_pos is not None:
                prune_map[batch_idx] = (open_pos, close_pos, return_pos)

    if not prune_map:
        return input_ids, model_kwargs

    cache = model_kwargs.get("past_key_values", None)
    use_layers_api = hasattr(cache, "layers")
    if cache is None:
        cache_populated = False
    elif use_layers_api:
        cache_populated = len(cache.layers) > 0
    elif hasattr(cache, "key_cache"):
        cache_populated = len(cache.key_cache) > 0  # type: ignore[union-attr]
    else:
        cache_populated = False

    can_retain_cache = (
        cache is not None
        and isinstance(cache, DynamicCache)
        and cache_populated
    )

    # Build pruned rows: keep [0..open_pos) + [close_pos+1..return_pos)
    # for pruned elements.  All three markers are dropped along with the
    # thought content; only the summary survives.
    new_rows: list[torch.Tensor] = []
    for b in range(batch_size):
        if b in prune_map:
            open_pos, close_pos, return_pos = prune_map[b]
            new_rows.append(
                torch.cat(
                    (input_ids[b, :open_pos], input_ids[b, close_pos + 1 : return_pos])
                )
            )
        else:
            new_rows.append(input_ids[b])

    if batch_size == 1:
        new_input_ids = new_rows[0].unsqueeze(0)
        max_len = new_input_ids.shape[1]
        model_kwargs["attention_mask"] = torch.ones(1, max_len, dtype=torch.long, device=device)
    else:
        max_len = max(r.shape[0] for r in new_rows)
        pad_id = getattr(model.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(model.config, "eos_token_id", 0)

        new_input_ids = torch.full(
            (batch_size, max_len), pad_id, dtype=input_ids.dtype, device=device,
        )
        new_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        for b, r in enumerate(new_rows):
            new_input_ids[b, : r.shape[0]] = r
            new_attention_mask[b, : r.shape[0]] = 1
        model_kwargs["attention_mask"] = new_attention_mask

    # Prune-aware mode: positions are renumbered contiguously, so we don't
    # thread `position_ids` explicitly — Gemma 4's forward derives them from
    # `cache_position` + cached length.
    model_kwargs.pop("position_ids", None)

    if can_retain_cache:
        if use_layers_api and len(cache.layers) > 0:  # type: ignore[union-attr]
            # Find the first layer that actually has a cache entry to read
            # old_seq_len from (KV-shared layers may be empty).
            old_seq_len = 0
            for lyr in cache.layers:  # type: ignore[union-attr]
                k = getattr(lyr, "keys", None)
                if k is not None and k.numel() > 0:
                    old_seq_len = k.shape[2]
                    break
        else:
            old_seq_len = cache.key_cache[0].shape[2]  # type: ignore[union-attr]

        new_cache_seq = _truncate_kv_cache_layer_aware(
            cache=cache,  # type: ignore[arg-type]
            prune_map=prune_map,
            batch_size=batch_size,
            old_seq_len=old_seq_len,
        )

        model_kwargs["cache_position"] = torch.arange(
            new_cache_seq, max_len, dtype=torch.int64, device=device,
        )
    else:
        model_kwargs.pop("past_key_values", None)
        model_kwargs["cache_position"] = torch.arange(
            max_len, dtype=torch.int64, device=device,
        )

    return new_input_ids, model_kwargs  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sample loop (prune-aware only)
# ---------------------------------------------------------------------------

def _sample(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    open_token_id: int,
    close_token_id: int,
    return_token_id: int,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    return_unpruned_output: bool = False,
    **model_kwargs,
):
    """Token-by-token decode with prune-aware hierarchical CoT.

    Stack semantics: each `<|channel>` opens a frame; `<channel|>` records
    the close position on the top frame; `<return|>` pops the top frame and
    triggers a prune of that frame's content.
    """
    pad_token_id = generation_config._pad_token_tensor  # type: ignore[attr-defined]
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(c, "eos_token_id") for c in stopping_criteria)
    do_sample = generation_config.do_sample

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    stacks: list[list[list[Optional[int]]]] = [[] for _ in range(batch_size)]

    if return_unpruned_output:
        unpruned_ids = [input_ids[b].tolist() for b in range(batch_size)]

    model_forward = (
        model.get_compiled_call(generation_config.compile_config)
        if GenerationMixin._valid_auto_compile_criteria(model, model_kwargs, generation_config)
        else model.__call__
    )

    # Pre-allocate input_ids buffer to avoid O(n^2) torch.cat copies.
    _pad_id_scalar = pad_token_id.item() if isinstance(pad_token_id, torch.Tensor) else pad_token_id
    _max_new = (
        generation_config.max_new_tokens
        if generation_config.max_new_tokens is not None
        else generation_config.max_length
    )
    _buf_len = input_ids.shape[1] + _max_new
    _ids_buf = torch.full(
        (batch_size, _buf_len), _pad_id_scalar,
        dtype=input_ids.dtype, device=input_ids.device,
    )
    _ids_buf[:, : input_ids.shape[1]] = input_ids
    _cur_len = input_ids.shape[1]
    input_ids = _ids_buf[:, :_cur_len]  # type: ignore[assignment]

    if not generation_config.is_assistant:
        outputs = GenerationMixin._prefill(model, input_ids, generation_config, model_kwargs)
        prefill_consumed = False
    else:
        model_kwargs = GenerationMixin._get_initial_cache_position(
            model, input_ids.shape[1], input_ids.device, model_kwargs
        )
        prefill_consumed = True

    while GenerationMixin._has_unfinished_sequences(
        model, this_peer_finished, synced_gpus, device=input_ids.device,
    ):
        if prefill_consumed:
            model_inputs = _prepare_inputs_for_generation(model, input_ids, **model_kwargs)
            with GenerationMixin._optimize_model_for_decode(model):
                outputs = model_forward(**model_inputs, return_dict=True)
        prefill_consumed = True

        model_kwargs = _update_model_kwargs_for_generation(
            model,
            outputs,  # type: ignore[arg-type]
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].float()  # type: ignore[union-attr]
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        _ids_buf[:, _cur_len] = next_tokens
        _cur_len += 1
        input_ids = _ids_buf[:, :_cur_len]  # type: ignore[assignment]

        _next_cpu = None
        def _get_next_cpu():
            nonlocal _next_cpu
            if _next_cpu is None:
                _next_cpu = next_tokens.cpu()
            return _next_cpu

        if streamer is not None:
            streamer.put(_get_next_cpu())

        pos = input_ids.shape[1] - 1

        if return_unpruned_output:
            _cpu = _get_next_cpu()
            for b in range(batch_size):
                unpruned_ids[b].append(_cpu[b].item())

        # Special-token bookkeeping.  Single GPU sync via .any().item().
        _is_open = next_tokens == open_token_id
        _is_close = next_tokens == close_token_id
        _is_return = next_tokens == return_token_id
        _any_special = (_is_open | _is_close | _is_return).any()

        if _any_special.item():
            for b in _is_open.nonzero(as_tuple=True)[0].tolist():
                stacks[b].append([pos, None, None])

            for b in _is_close.nonzero(as_tuple=True)[0].tolist():
                if stacks[b]:
                    stacks[b][-1][1] = pos

            for b in _is_return.nonzero(as_tuple=True)[0].tolist():
                if stacks[b]:
                    stacks[b][-1][2] = pos

            return_indices = _is_return.nonzero(as_tuple=True)[0].tolist()
            prune_candidates = [idx for idx in return_indices if stacks[idx]]
        else:
            prune_candidates = []

        if prune_candidates:
            input_ids, model_kwargs = _prune_model_inputs(
                model,
                prune_input_candidates=prune_candidates,
                prune_input_locations=[[stacks[b].pop()] for b in prune_candidates],  # type: ignore[arg-type]
                input_ids=input_ids,
                model_kwargs=model_kwargs,
            )
            _cur_len = input_ids.shape[1]
            _ids_buf[:, :_cur_len] = input_ids
            input_ids = _ids_buf[:, :_cur_len]  # type: ignore[assignment]

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)  # type: ignore[arg-type]
        this_peer_finished = not unfinished_sequences.any()

        del outputs  # type: ignore[possibly-undefined]

    if streamer is not None:
        streamer.end()

    if return_unpruned_output:
        max_len = max(len(ids) for ids in unpruned_ids)
        pad_id = pad_token_id.item() if isinstance(pad_token_id, torch.Tensor) else pad_token_id
        unpruned_tensor = torch.full(
            (batch_size, max_len), pad_id,
            dtype=input_ids.dtype, device=input_ids.device,
        )
        for b, ids in enumerate(unpruned_ids):
            unpruned_tensor[b, : len(ids)] = torch.tensor(ids, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = unpruned_tensor  # type: ignore[assignment]

    if return_dict_in_generate:
        cache = None
        if any(k in model_kwargs for k in ALL_CACHE_NAMES):
            cache_key = next(k for k in ALL_CACHE_NAMES if k in model_kwargs)
            cache = model_kwargs[cache_key]
        if model.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                decoder_attentions=decoder_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
        return GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=cache,
        )
    return input_ids


def generate(
    model,
    tokenizer: PreTrainedTokenizerBase,
    return_unpruned_output: bool = False,
    **kwargs,
):
    """Top-level entry point for prune-aware Gemma 4 generation.

    Resolves the three marker token IDs from the tokenizer and dispatches to
    `_sample` via `GenerationMixin.generate`'s `custom_generate` hook.
    """
    custom_generate = kwargs.pop("custom_generate", _sample)
    open_token_id = tokenizer.convert_tokens_to_ids(CHANNEL_OPEN_TOKEN)
    close_token_id = tokenizer.convert_tokens_to_ids(CHANNEL_CLOSE_TOKEN)
    return_token_id = tokenizer.convert_tokens_to_ids(RETURN_TOKEN)

    return GenerationMixin.generate(
        model,
        custom_generate=custom_generate,
        open_token_id=open_token_id,
        close_token_id=close_token_id,
        return_token_id=return_token_id,
        return_unpruned_output=return_unpruned_output,
        **kwargs,
    )
