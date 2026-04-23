"""Tests for KV cache retention after pruning.

These tests verify that the KV cache is correctly pruned to retain only the
prefix (tokens before the thought block) and that the caller receives the
right cache_position to re-process solution tokens.
"""

import torch
import pytest
from transformers import DynamicCache

from .generate import (
    _retain_and_prune_kv_cache,
    _prune_model_inputs,
)


# ---------------------------------------------------------------------------
# Test parameters matching Qwen2.5-Math-1.5B
# ---------------------------------------------------------------------------
HEAD_DIM = 128
NUM_HEADS = 2       # num_key_value_heads for Qwen2.5-Math-1.5B
NUM_LAYERS = 2      # reduced for test speed (real model has 28)
BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_num_layers(cache: DynamicCache) -> int:
    if hasattr(cache, 'layers'):
        return len(cache.layers)
    return len(cache.key_cache)  # type: ignore[attr-defined]


def _cache_keys(cache: DynamicCache, layer_idx: int) -> torch.Tensor:
    if hasattr(cache, 'layers') and len(cache.layers) > layer_idx:
        return cache.layers[layer_idx].keys
    return cache.key_cache[layer_idx]  # type: ignore[attr-defined]


def _cache_values(cache: DynamicCache, layer_idx: int) -> torch.Tensor:
    if hasattr(cache, 'layers') and len(cache.layers) > layer_idx:
        return cache.layers[layer_idx].values
    return cache.value_cache[layer_idx]  # type: ignore[attr-defined]


def _make_cache(batch_size: int, num_layers: int, num_heads: int, seq_len: int, head_dim: int):
    """Create a DynamicCache with random KV entries (compatible with transformers ≥5.x)."""
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.update(k, v, layer_idx=layer_idx)
    return cache


def _make_mock_model(head_dim=HEAD_DIM, num_heads=12, pad_token_id=0):
    """Create a minimal mock model with the config attributes needed by _prune_model_inputs."""
    from unittest.mock import MagicMock
    model = MagicMock()
    model.config.hidden_size = head_dim * num_heads
    model.config.num_attention_heads = num_heads
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = 2
    return model


# ---------------------------------------------------------------------------
# _retain_and_prune_kv_cache
# ---------------------------------------------------------------------------

class TestRetainAndPruneKvCache:
    def test_keeps_only_prefix(self):
        """Should keep only [0..thought_pos] in the cache."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        new_seq = _retain_and_prune_kv_cache(
            cache, prune_map, BATCH_SIZE, seq_len,
        )

        # Only prefix [0..5] = 6 entries
        assert new_seq == thought_pos + 1
        assert _cache_keys(cache, 0).shape == (BATCH_SIZE, NUM_HEADS, 6, HEAD_DIM)
        assert _cache_values(cache, 0).shape == (BATCH_SIZE, NUM_HEADS, 6, HEAD_DIM)

    def test_prefix_keys_unchanged(self):
        """Prefix keys should be preserved exactly (no rotation applied)."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        orig_k = _cache_keys(cache, 0)[:, :, :thought_pos + 1, :].clone()

        _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, seq_len)

        assert torch.allclose(_cache_keys(cache, 0)[:, :, :thought_pos + 1, :], orig_k)

    def test_prefix_values_unchanged(self):
        """Prefix values should be preserved exactly."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        orig_v = _cache_values(cache, 0)[:, :, :thought_pos + 1, :].clone()

        _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, seq_len)

        assert torch.allclose(_cache_values(cache, 0)[:, :, :thought_pos + 1, :], orig_v)

    def test_non_pruned_keeps_all(self):
        """Non-pruned batch elements should keep all entries."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        prune_map: dict[int, tuple[int, int]] = {}

        orig_k = _cache_keys(cache, 0).clone()

        new_seq = _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, seq_len)

        assert new_seq == seq_len
        assert torch.allclose(_cache_keys(cache, 0), orig_k)

    def test_prune_at_beginning(self):
        """Pruning a block starting at position 0."""
        seq_len = 15
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        # [THOUGHT] at 0, [SOLUTION] at 3 → keep only [0]
        prune_map = {0: (0, 3)}

        new_seq = _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, seq_len)

        assert new_seq == 1

    def test_empty_cache(self):
        """Empty cache should not crash."""
        cache = DynamicCache()
        prune_map = {0: (5, 10)}

        new_seq = _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, 0)
        assert new_seq == 0

    def test_all_layers_pruned(self):
        """All layers should be pruned consistently."""
        seq_len = 20
        cache = _make_cache(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        thought_pos, solution_pos = 5, 10
        prune_map = {0: (thought_pos, solution_pos)}

        _retain_and_prune_kv_cache(cache, prune_map, BATCH_SIZE, seq_len)

        for layer_idx in range(NUM_LAYERS):
            assert _cache_keys(cache, layer_idx).shape[2] == thought_pos + 1
            assert _cache_values(cache, layer_idx).shape[2] == thought_pos + 1


# ---------------------------------------------------------------------------
# _prune_model_inputs with retain_kv_cache
# ---------------------------------------------------------------------------

class TestPruneModelInputsWithCacheRetention:
    """Test that _prune_model_inputs correctly retains KV cache."""

    def test_cache_retained(self):
        """With retain_kv_cache=True, cache should keep only the prefix."""
        model = _make_mock_model()
        seq_len = 20

        input_ids = torch.arange(seq_len).unsqueeze(0).long()
        thought_pos, solution_pos, return_pos = 5, 10, 19

        cache = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        model_kwargs = {
            'past_key_values': cache,
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }

        new_input_ids, new_kwargs = _prune_model_inputs(
            model,
            prune_input_candidates=[0],
            prune_input_locations=[[(thought_pos, solution_pos, return_pos)]],
            input_ids=input_ids,
            prune_aware=True,
            model_kwargs=model_kwargs,
            retain_kv_cache=True,
        )

        # Cache should still be present (not discarded)
        assert 'past_key_values' in new_kwargs
        retained_cache = new_kwargs['past_key_values']
        assert isinstance(retained_cache, DynamicCache)
        assert _cache_num_layers(retained_cache) == NUM_LAYERS

        # input_ids should be pruned
        pruned_len = seq_len - (solution_pos - thought_pos)  # 20 - 5 = 15
        assert new_input_ids.shape[1] == pruned_len

        # Cache should have only the prefix (thought_pos + 1 entries)
        assert _cache_keys(retained_cache, 0).shape[2] == thought_pos + 1

        # cache_position should cover solution tokens for re-processing
        expected_cache_pos = list(range(thought_pos + 1, pruned_len))
        assert new_kwargs['cache_position'].tolist() == expected_cache_pos

    def test_cache_discarded_when_disabled(self):
        """With retain_kv_cache=False, cache should be discarded (old behavior)."""
        model = _make_mock_model()
        seq_len = 20

        input_ids = torch.arange(seq_len).unsqueeze(0).long()
        thought_pos, solution_pos, return_pos = 5, 10, 19

        cache = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        model_kwargs = {
            'past_key_values': cache,
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }

        new_input_ids, new_kwargs = _prune_model_inputs(
            model,
            prune_input_candidates=[0],
            prune_input_locations=[[(thought_pos, solution_pos, return_pos)]],
            input_ids=input_ids,
            prune_aware=True,
            model_kwargs=model_kwargs,
            retain_kv_cache=False,
        )

        # Cache should be discarded
        assert 'past_key_values' not in new_kwargs

        # cache_position should be full range (for prefill)
        pruned_len = seq_len - (solution_pos - thought_pos)
        assert new_kwargs['cache_position'].tolist() == list(range(pruned_len))

    def test_input_ids_same_regardless_of_cache_mode(self):
        """The pruned input_ids should be the same regardless of cache retention mode."""
        model = _make_mock_model()
        seq_len = 20

        input_ids = torch.arange(seq_len).unsqueeze(0).long()
        thought_pos, solution_pos, return_pos = 5, 10, 19

        # Run with cache retention
        cache1 = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        kwargs1 = {
            'past_key_values': cache1,
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }
        ids_retained, _ = _prune_model_inputs(
            model, [0], [[(thought_pos, solution_pos, return_pos)]],
            input_ids.clone(), True, kwargs1, retain_kv_cache=True,
        )

        # Run without cache retention
        cache2 = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        kwargs2 = {
            'past_key_values': cache2,
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }
        ids_discarded, _ = _prune_model_inputs(
            model, [0], [[(thought_pos, solution_pos, return_pos)]],
            input_ids.clone(), True, kwargs2, retain_kv_cache=False,
        )

        assert torch.equal(ids_retained, ids_discarded)


# ---------------------------------------------------------------------------
# Integration tests: prune-aware + retain_kv_cache
# ---------------------------------------------------------------------------

class TestPruneAwareRetainKvCacheIntegration:
    """Integration tests for prune_aware=True, retain_kv_cache=True covering
    single/multiple prune events, heterogeneous batches, and return_unpruned_output."""

    def test_single_prune_input_ids_and_cache_position(self):
        """After one prune: input_ids = prefix + solution, cache has prefix only,
        cache_position covers [prefix_len, pruned_len) for re-processing."""
        model = _make_mock_model()
        seq_len = 20
        # Tokens: [0..19], thought at 5, solution at 10, return at 19
        input_ids = torch.arange(seq_len).unsqueeze(0).long()
        thought_pos, solution_pos, return_pos = 5, 10, 19

        cache = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        orig_prefix_k = _cache_keys(cache, 0)[:, :, :thought_pos + 1, :].clone()

        model_kwargs = {
            'past_key_values': cache,
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }

        new_ids, new_kwargs = _prune_model_inputs(
            model, [0], [[(thought_pos, solution_pos, return_pos)]],
            input_ids, prune_aware=True, model_kwargs=model_kwargs,
            retain_kv_cache=True,
        )

        # input_ids = prefix [0..5] + solution [11..19] = 6 + 9 = 15 tokens
        pruned_len = thought_pos + 1 + (seq_len - solution_pos - 1)
        assert new_ids.shape == (1, pruned_len)
        assert new_ids[0, :thought_pos + 1].tolist() == list(range(thought_pos + 1))
        assert new_ids[0, thought_pos + 1:].tolist() == list(range(solution_pos + 1, seq_len))

        # Cache has exactly prefix entries
        retained_cache = new_kwargs['past_key_values']
        assert _cache_keys(retained_cache, 0).shape[2] == thought_pos + 1
        # Prefix keys preserved exactly
        assert torch.allclose(_cache_keys(retained_cache, 0), orig_prefix_k)

        # cache_position covers re-processing range
        expected_cp = list(range(thought_pos + 1, pruned_len))
        assert new_kwargs['cache_position'].tolist() == expected_cp

    def test_heterogeneous_batch_padding_is_zeroed(self):
        """batch_size=2, only element 0 pruned. Padding positions in KV cache
        must be actual zeros, not copies of position 0 (Bug 2 regression)."""
        model = _make_mock_model()
        batch_size = 2
        seq_len = 20
        thought_pos, solution_pos, return_pos = 5, 10, 19

        input_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).clone()
        cache = _make_cache(batch_size, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)

        model_kwargs = {
            'past_key_values': cache,
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }

        # Only batch element 0 is pruned
        new_ids, new_kwargs = _prune_model_inputs(
            model,
            prune_input_candidates=[0],
            prune_input_locations=[[(thought_pos, solution_pos, return_pos)]],
            input_ids=input_ids,
            prune_aware=True,
            model_kwargs=model_kwargs,
            retain_kv_cache=True,
        )

        retained_cache = new_kwargs['past_key_values']
        # Element 0: prefix_len = thought_pos + 1 = 6
        # Element 1: not pruned, keeps all seq_len = 20
        # max_new_seq = 20
        keys_0 = _cache_keys(retained_cache, 0)
        new_seq = keys_0.shape[2]
        assert new_seq == seq_len  # max(6, 20) = 20

        prefix_len_0 = thought_pos + 1  # 6
        # Padding positions [6:20) for batch element 0 must be zero
        padding = keys_0[0, :, prefix_len_0:, :]
        assert torch.all(padding == 0), (
            f"Padding positions should be zero but got nonzero values"
        )

    def test_heterogeneous_batch_both_pruned_different_lengths(self):
        """batch_size=2, both elements pruned to different prefix lengths.
        Each element's KV cache prefix must be correct, padding zeroed."""
        model = _make_mock_model()
        batch_size = 2
        seq_len = 20

        input_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).clone()
        cache = _make_cache(batch_size, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        orig_prefix_k0 = _cache_keys(cache, 0)[0:1, :, :4, :].clone()  # prefix len 4 for b=0
        orig_prefix_k1 = _cache_keys(cache, 0)[1:2, :, :8, :].clone()  # prefix len 8 for b=1

        model_kwargs = {
            'past_key_values': cache,
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }

        # b=0: thought at 3, solution at 8 → prefix_len=4
        # b=1: thought at 7, solution at 12 → prefix_len=8
        new_ids, new_kwargs = _prune_model_inputs(
            model,
            prune_input_candidates=[0, 1],
            prune_input_locations=[
                [(3, 8, 19)],
                [(7, 12, 19)],
            ],
            input_ids=input_ids,
            prune_aware=True,
            model_kwargs=model_kwargs,
            retain_kv_cache=True,
        )

        retained_cache = new_kwargs['past_key_values']
        keys = _cache_keys(retained_cache, 0)
        max_prefix = 8  # max(4, 8)
        assert keys.shape[2] == max_prefix

        # b=0: prefix [0:4] preserved, positions [4:8] zeroed
        assert torch.allclose(keys[0:1, :, :4, :], orig_prefix_k0)
        assert torch.all(keys[0, :, 4:, :] == 0)

        # b=1: prefix [0:8] preserved (no padding needed)
        assert torch.allclose(keys[1:2, :, :8, :], orig_prefix_k1)

    def test_return_unpruned_output_tracking(self):
        """Pruned input_ids should contain prefix + post-solution tokens,
        with the thought..solution range removed. The removed tokens would
        appear only in the unpruned tracking list."""
        model = _make_mock_model()
        seq_len = 15
        # Tokens: [100, 101, ..., 114]
        input_ids = (torch.arange(seq_len) + 100).unsqueeze(0).long()
        thought_pos, solution_pos, return_pos = 3, 7, 14

        cache = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        model_kwargs = {
            'past_key_values': cache,
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long),
            'cache_position': torch.tensor([seq_len - 1], dtype=torch.int64),
        }

        new_ids, new_kwargs = _prune_model_inputs(
            model, [0], [[(thought_pos, solution_pos, return_pos)]],
            input_ids, prune_aware=True, model_kwargs=model_kwargs,
            retain_kv_cache=True,
        )

        # Pruned: prefix [100..103] + post-solution [108..114]
        expected_pruned = list(range(100, 104)) + list(range(108, 115))
        assert new_ids[0].tolist() == expected_pruned

        # The removed range [104..107] (thought content + solution marker) is
        # what would go into unpruned_ids tracking in the generation loop
        removed = list(range(104, 108))
        original = list(range(100, 115))
        # Verify: pruned + removed = original (modulo ordering)
        reconstructed = sorted(new_ids[0].tolist() + removed)
        assert reconstructed == original

    def test_two_sequential_prune_events(self):
        """Two sequential prunes on same element. After second prune, cache
        prefix length should reflect only the second prune's prefix."""
        model = _make_mock_model()

        # --- First prune ---
        seq_len_1 = 20
        input_ids_1 = torch.arange(seq_len_1).unsqueeze(0).long()
        cache_1 = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len_1, HEAD_DIM)
        kwargs_1 = {
            'past_key_values': cache_1,
            'attention_mask': torch.ones(1, seq_len_1, dtype=torch.long),
            'cache_position': torch.tensor([seq_len_1 - 1], dtype=torch.int64),
        }
        # thought at 3, solution at 8, return at 19 → prefix_len=4
        new_ids_1, new_kwargs_1 = _prune_model_inputs(
            model, [0], [[(3, 8, 19)]],
            input_ids_1, prune_aware=True, model_kwargs=kwargs_1,
            retain_kv_cache=True,
        )
        cache_after_1 = new_kwargs_1['past_key_values']
        assert _cache_keys(cache_after_1, 0).shape[2] == 4  # prefix [0..3]

        # --- Second prune on the result ---
        # Simulate: after first prune we have 15 tokens, generation continues
        # to seq_len_2 = 25, then a second thought block appears
        seq_len_2 = 25
        input_ids_2 = torch.arange(seq_len_2).unsqueeze(0).long()
        # Extend cache to seq_len_2
        cache_2 = _make_cache(1, NUM_LAYERS, NUM_HEADS, seq_len_2, HEAD_DIM)
        kwargs_2 = {
            'past_key_values': cache_2,
            'attention_mask': torch.ones(1, seq_len_2, dtype=torch.long),
            'cache_position': torch.tensor([seq_len_2 - 1], dtype=torch.int64),
        }
        # thought at 6, solution at 12, return at 24 → prefix_len=7
        new_ids_2, new_kwargs_2 = _prune_model_inputs(
            model, [0], [[(6, 12, 24)]],
            input_ids_2, prune_aware=True, model_kwargs=kwargs_2,
            retain_kv_cache=True,
        )
        cache_after_2 = new_kwargs_2['past_key_values']
        # Second prune: prefix is [0..6] = 7 entries (independent of first prune)
        assert _cache_keys(cache_after_2, 0).shape[2] == 7

        # cache_position for re-processing: [7, pruned_len)
        pruned_len_2 = 7 + (seq_len_2 - 12 - 1)  # 7 + 12 = 19
        expected_cp = list(range(7, pruned_len_2))
        assert new_kwargs_2['cache_position'].tolist() == expected_cp

    def test_gather_padding_regression_bug2(self):
        """Direct regression test for Bug 2: heterogeneous gather must produce
        zeros in padding positions, not copies of position-0 KV values."""
        batch_size = 2
        seq_len = 10

        # Create cache with known non-zero values at position 0
        cache = _make_cache(batch_size, NUM_LAYERS, NUM_HEADS, seq_len, HEAD_DIM)
        # Ensure position 0 has non-zero values (set to 1.0 for easy checking)
        for layer_idx in range(NUM_LAYERS):
            keys = _cache_keys(cache, layer_idx)
            vals = _cache_values(cache, layer_idx)
            keys[:, :, 0, :] = 1.0
            vals[:, :, 0, :] = 1.0

        # b=0: prune to prefix_len=3, b=1: no prune (keep all 10)
        prune_map = {0: (2, 5)}  # thought at 2, solution at 5 → prefix [0..2] = 3

        new_seq = _retain_and_prune_kv_cache(cache, prune_map, batch_size, seq_len)

        assert new_seq == seq_len  # max(3, 10) = 10

        for layer_idx in range(NUM_LAYERS):
            keys = _cache_keys(cache, layer_idx)
            vals = _cache_values(cache, layer_idx)

            # b=0: positions [3:10) must be zero (not copies of pos 0)
            assert torch.all(keys[0, :, 3:, :] == 0), (
                f"Layer {layer_idx}: padding keys should be zero"
            )
            assert torch.all(vals[0, :, 3:, :] == 0), (
                f"Layer {layer_idx}: padding vals should be zero"
            )

            # b=0: prefix [0:3] should still have data
            assert keys[0, :, 0, :].sum() != 0, "Prefix position 0 should be non-zero"

            # b=1: all 10 positions should be preserved (non-pruned)
            assert keys[1].shape[1] == seq_len
