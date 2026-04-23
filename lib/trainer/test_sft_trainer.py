"""Integration tests for HCotSFTTrainer.compute_loss — staged pruning."""

import torch
import pytest
from unittest.mock import MagicMock, patch

from transformers import GPT2Config, GPT2LMHeadModel

from .sft_trainer import HCotSFTTrainer

THOUGHT_ID = 100
SOLUTION_ID = 101
RETURN_ID = 102
PAD_ID = 0

VOCAB_SIZE = 200


def _tiny_model():
    config = GPT2Config(
        vocab_size=VOCAB_SIZE, n_embd=32, n_head=2, n_layer=1, n_positions=64,
    )
    return GPT2LMHeadModel(config)


def _make_trainer(prune_aware=True):
    """Build an HCotSFTTrainer with minimal Trainer plumbing (no datasets/optimizer)."""
    t = object.__new__(HCotSFTTrainer)
    t.thought_token_id = THOUGHT_ID
    t.solution_token_id = SOLUTION_ID
    t.return_token_id = RETURN_ID
    t.prune_aware = prune_aware

    # Minimal attributes that Trainer.compute_loss reads
    mock_pc = MagicMock()
    mock_pc.pad_token_id = PAD_ID
    t.processing_class = mock_pc
    t.label_smoother = None
    t.compute_loss_func = None
    t.model_accepts_loss_kwargs = False
    t.accelerator = MagicMock()
    t.accelerator.parallelism_config = None

    # Mock training args accessed in Trainer.compute_loss
    t.args = MagicMock()
    t.args.average_tokens_across_devices = False

    return t


# ---------------------------------------------------------------------------
# Staged (prune_aware=True)
# ---------------------------------------------------------------------------


class TestStagedComputeLoss:
    """Smoke tests for _compute_loss_staged via compute_loss(prune_aware=True)."""

    def test_single_block_returns_scalar_loss(self):
        model = _tiny_model()
        trainer = _make_trainer(prune_aware=True)

        # A B [TH] c d [SOL] e [RET] f g
        ids = torch.tensor([[10, 11, THOUGHT_ID, 12, 13, SOLUTION_ID, 14, RETURN_ID, 15, 16]])
        labels = ids.clone()
        mask = torch.ones_like(ids)

        inputs = {'input_ids': ids, 'attention_mask': mask, 'labels': labels}
        loss = trainer.compute_loss(model, inputs)

        assert loss.dim() == 0, "loss should be scalar"
        assert loss.requires_grad
        assert loss.item() > 0

    def test_no_blocks_single_stage(self):
        model = _tiny_model()
        trainer = _make_trainer(prune_aware=True)

        ids = torch.tensor([[10, 11, 12, 13]])
        labels = ids.clone()
        mask = torch.ones_like(ids)

        inputs = {'input_ids': ids, 'attention_mask': mask, 'labels': labels}
        loss = trainer.compute_loss(model, inputs)

        assert loss.dim() == 0
        assert loss.requires_grad

    def test_nested_blocks(self):
        model = _tiny_model()
        trainer = _make_trainer(prune_aware=True)

        # A [TH1] c [TH2] d [SOL2] e [RET2] f [SOL1] g [RET1] h
        ids = torch.tensor([[10, THOUGHT_ID, 12, THOUGHT_ID, 13, SOLUTION_ID,
                             14, RETURN_ID, 15, SOLUTION_ID, 16, RETURN_ID, 17]])
        labels = ids.clone()
        mask = torch.ones_like(ids)

        inputs = {'input_ids': ids, 'attention_mask': mask, 'labels': labels}
        loss = trainer.compute_loss(model, inputs)

        assert loss.dim() == 0
        assert loss.requires_grad

    def test_batch_of_two(self):
        """Two sequences in one batch — different number of blocks."""
        model = _tiny_model()
        trainer = _make_trainer(prune_aware=True)

        # Sequence 0: single block (10 tokens)
        seq0 = [10, 11, THOUGHT_ID, 12, 13, SOLUTION_ID, 14, RETURN_ID, 15, 16]
        # Sequence 1: no blocks (shorter, padded)
        seq1 = [20, 21, 22, 23, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]

        ids = torch.tensor([seq0, seq1])
        labels = ids.clone()
        labels[1, 4:] = -100  # pad labels
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        ])

        inputs = {'input_ids': ids, 'attention_mask': mask, 'labels': labels}
        loss = trainer.compute_loss(model, inputs)

        assert loss.dim() == 0
        assert loss.requires_grad

    def test_gradient_flows(self):
        """Loss from staged forward supports backward()."""
        model = _tiny_model()
        trainer = _make_trainer(prune_aware=True)

        ids = torch.tensor([[10, 11, THOUGHT_ID, 12, 13, SOLUTION_ID, 14, RETURN_ID, 15, 16]])
        labels = ids.clone()
        mask = torch.ones_like(ids)

        inputs = {'input_ids': ids, 'attention_mask': mask, 'labels': labels}
        loss = trainer.compute_loss(model, inputs)
        loss.backward()

        # At least some parameters should have gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_return_outputs(self):
        model = _tiny_model()
        trainer = _make_trainer(prune_aware=True)

        ids = torch.tensor([[10, 11, THOUGHT_ID, 12, 13, SOLUTION_ID, 14, RETURN_ID, 15, 16]])
        labels = ids.clone()
        mask = torch.ones_like(ids)

        inputs = {'input_ids': ids, 'attention_mask': mask, 'labels': labels}
        result = trainer.compute_loss(model, inputs, return_outputs=True)

        assert isinstance(result, tuple)
        loss, outputs = result
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Fallback (prune_aware=False)
# ---------------------------------------------------------------------------


class TestFallbackComputeLoss:
    """prune_aware=False uses the single-pass hierarchical mask path."""

    def test_single_block(self):
        # _prepare_attention_mask emits bfloat16, so model must match
        model = _tiny_model().to(torch.bfloat16)
        trainer = _make_trainer(prune_aware=False)

        ids = torch.tensor([[10, 11, THOUGHT_ID, 12, 13, SOLUTION_ID, 14, RETURN_ID, 15, 16]])
        labels = ids.clone()
        mask = torch.ones_like(ids)

        inputs = {'input_ids': ids, 'attention_mask': mask, 'labels': labels}
        loss = trainer.compute_loss(model, inputs)

        assert loss.dim() == 0
        assert loss.requires_grad

    def test_no_blocks(self):
        model = _tiny_model().to(torch.bfloat16)
        trainer = _make_trainer(prune_aware=False)

        ids = torch.tensor([[10, 11, 12, 13]])
        labels = ids.clone()
        mask = torch.ones_like(ids)

        inputs = {'input_ids': ids, 'attention_mask': mask, 'labels': labels}
        loss = trainer.compute_loss(model, inputs)

        assert loss.dim() == 0
        assert loss.requires_grad

    def test_mask_shape_matches_model_dtype(self):
        """The 4D mask dtype must match the model parameter dtype."""
        model = _tiny_model().to(torch.float32)
        trainer = _make_trainer(prune_aware=False)
        trainer.model = model

        ids = torch.tensor([[10, 11, THOUGHT_ID, 12, 13, SOLUTION_ID, 14, RETURN_ID, 15, 16]])
        mask = torch.ones_like(ids)
        inputs = {'input_ids': ids, 'attention_mask': mask, 'labels': ids.clone()}

        attn_mask = trainer._prepare_attention_mask(inputs)
        assert attn_mask.dtype == torch.float32

    def test_hierarchical_mask_affects_logits(self):
        """Verify the 4D mask actually changes model output.

        When the hierarchical mask blocks thought content after [RETURN],
        logits at post-[RETURN] positions must differ from a plain causal
        run.  If they are identical, the mask is being silently ignored
        (e.g. by Flash Attention or SDPA with is_causal=True).
        """
        model = _tiny_model().to(torch.bfloat16)
        trainer = _make_trainer(prune_aware=False)
        trainer.model = model

        # A B [TH] c d [SOL] e [RET] f g
        ids = torch.tensor([[10, 11, THOUGHT_ID, 12, 13, SOLUTION_ID, 14, RETURN_ID, 15, 16]])
        mask_2d = torch.ones_like(ids)

        # Run with standard causal mask (no hierarchical masking)
        with torch.no_grad():
            out_causal = model(input_ids=ids, attention_mask=mask_2d)

        # Run with hierarchical mask
        inputs = {'input_ids': ids, 'attention_mask': mask_2d.clone(), 'labels': ids.clone()}
        attn_4d = trainer._prepare_attention_mask(inputs)
        with torch.no_grad():
            out_masked = model(input_ids=ids, attention_mask=attn_4d)

        # Logits AFTER [RETURN] must differ — the mask blocks thought content
        # that was visible under the plain causal mask.
        return_pos = 7  # [RETURN] is at index 7
        post_return_causal = out_causal.logits[:, return_pos + 1:, :]
        post_return_masked = out_masked.logits[:, return_pos + 1:, :]
        assert not torch.allclose(
            post_return_causal, post_return_masked, atol=1e-4
        ), (
            "Logits after [RETURN] are identical with and without the "
            "hierarchical mask — the 4D mask is being silently ignored! "
            "Ensure the model uses attn_implementation='eager' or 'sdpa'."
        )
