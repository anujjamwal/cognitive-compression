from dataclasses import field
import logging
from typing import Any
import torch
from torch import nn
from transformers import Trainer, AutoProcessor, PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.processing_utils import ProcessorMixin
from trl import trainer
from trl.trainer.utils import get_config_model_id
import utils

logger = logging.getLogger(__name__)

# Attention implementations that are known to correctly handle custom 4D
# attention masks.  Flash-Attention-2 only supports 2D (padding) masks and
# will silently ignore a full 4D mask, which completely breaks the
# hierarchical masking required by prune_aware=False training.
_4D_MASK_SAFE_ATTN_IMPLEMENTATIONS = {"eager", "sdpa"}


class HCotSFTTrainer(trainer.sft_trainer.SFTTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: trainer.sft_config.SFTConfig | TrainingArguments | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        prune_aware: bool = True,
        **kwargs
    ):
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer # type: ignore
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        self.thought_token_id = tokenizer.convert_tokens_to_ids("[THOUGHT]")
        self.solution_token_id = tokenizer.convert_tokens_to_ids("[SOLUTION]")
        self.return_token_id = tokenizer.convert_tokens_to_ids("[RETURN]")
        self.prune_aware = prune_aware

        if not prune_aware:
            self._validate_attn_implementation(model)

        super().__init__(model, args=args, processing_class=processing_class, **kwargs)

    @staticmethod
    def _validate_attn_implementation(model: PreTrainedModel):
        """Ensure the model uses an attention implementation that respects 4D masks.

        prune_aware=False relies on a custom 4D hierarchical attention mask to
        simulate pruning during training.  Flash-Attention-2 only supports 2D
        padding masks and will silently drop the hierarchical pattern, making
        training equivalent to plain SFT.  This causes the model to never
        learn post-pruning continuations, producing out-of-order / invalid
        special-token sequences at inference time.
        """
        attn_impl = getattr(model.config, "_attn_implementation", None)
        if attn_impl and attn_impl not in _4D_MASK_SAFE_ATTN_IMPLEMENTATIONS:
            raise ValueError(
                f"prune_aware=False requires an attention implementation that "
                f"respects custom 4D masks. The model uses "
                f"'{attn_impl}' which is not supported. "
                f"Load the model with "
                f'attn_implementation="eager" or "sdpa" instead. '
                f"Supported: {sorted(_4D_MASK_SAFE_ATTN_IMPLEMENTATIONS)}"
            )
        if attn_impl:
            logger.info(
                "prune_aware=False: attention implementation '%s' is compatible "
                "with 4D hierarchical masks.",
                attn_impl,
            )
        else:
            logger.warning(
                "Could not determine the model's attention implementation. "
                "prune_aware=False requires 'eager' or 'sdpa' to correctly "
                "apply the 4D hierarchical mask. If using flash_attention_2, "
                "the mask will be silently ignored and training will degrade "
                "to plain SFT."
            )


    def _prepare_attention_mask(self, inputs: dict[str, torch.Tensor|Any], dtype: torch.dtype | None = None):
        input_ids = inputs['input_ids']
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Use the model's parameter dtype so the mask matches activations.
        # Falls back to bfloat16 when the dtype cannot be determined.
        if dtype is None:
            try:
                dtype = next(self.model.parameters()).dtype
            except (StopIteration, AttributeError):
                dtype = torch.bfloat16

        batch_blocks = utils.find_cot_blocks(
            input_ids, self.thought_token_id, self.solution_token_id, self.return_token_id
        )

        # Create the default mask with lower triangle
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )

        padding_mask = inputs['attention_mask']
        masks = []
        for b in range(batch_size):
            mask = causal.clone()
            for thought_pos, solution_pos, return_pos in batch_blocks[b]:
                # By reduction rule we maintain [THOUGHT] and [RETURN] tokens.
                mask[return_pos + 1:, thought_pos+1:solution_pos+1] = False

            mask = mask & padding_mask[b].bool().unsqueeze(0)
            masks.append(mask)

        mask_tensor = torch.stack(masks).unsqueeze(1)
        float_mask = torch.zeros(
            batch_size, 1, seq_len, seq_len, dtype=dtype, device=device
        )
        float_mask.masked_fill_(~mask_tensor, torch.finfo(dtype).min)
        return float_mask

    def _compute_loss_staged(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Staged pruning training: multiple forward passes mirroring inference.

        Each pruning event (at [RETURN]) creates a new stage where thought
        content is physically removed and the model does a fresh forward on
        the pruned sequence with contiguous positions — exactly matching
        inference behaviour.
        """
        input_ids = inputs['input_ids']
        labels = inputs.get('labels', input_ids)
        attention_mask = inputs['attention_mask']
        batch_size = input_ids.shape[0]
        device = input_ids.device

        batch_blocks = utils.find_cot_blocks(
            input_ids, self.thought_token_id, self.solution_token_id, self.return_token_id
        )

        # Build all stages for every batch element
        all_stages = []
        for b in range(batch_size):
            stages = utils.build_stages(
                input_ids[b], labels[b], attention_mask[b], batch_blocks[b]
            )
            all_stages.append(stages)

        max_num_stages = max(len(s) for s in all_stages)

        total_loss = None
        first_outputs = None
        pad_id = getattr(self.processing_class, 'pad_token_id', 0) or 0

        for stage_idx in range(max_num_stages):
            # Collect sequences from batch elements that participate in this stage
            participating = []
            for b in range(batch_size):
                if stage_idx < len(all_stages[b]):
                    participating.append(b)

            if not participating:
                continue

            # Pad to uniform length and stack into a batch
            max_len = max(all_stages[b][stage_idx][0].shape[0] for b in participating)
            n = len(participating)

            batch_ids = torch.full((n, max_len), pad_id, dtype=input_ids.dtype, device=device)
            batch_labels = torch.full((n, max_len), -100, dtype=input_ids.dtype, device=device)
            batch_mask = torch.zeros((n, max_len), dtype=attention_mask.dtype, device=device)

            for local_idx, b_idx in enumerate(participating):
                s_ids, s_labels, s_mask = all_stages[b_idx][stage_idx]

                batch_ids[local_idx, :s_ids.shape[0]] = s_ids
                batch_labels[local_idx, :s_labels.shape[0]] = s_labels
                batch_mask[local_idx, :s_mask.shape[0]] = s_mask

            stage_inputs: dict[str, torch.Tensor | Any] = {
                'input_ids': batch_ids,
                'attention_mask': batch_mask,
                'labels': batch_labels,
            }
            stage_loss, stage_outputs = Trainer.compute_loss(
                self, model, stage_inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch,
            )

            total_loss = stage_loss if total_loss is None else total_loss + stage_loss
            if first_outputs is None:
                first_outputs = stage_outputs

        if total_loss is None:
            # Edge case: no stages produced (shouldn't happen with valid data)
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return (total_loss, first_outputs) if return_outputs else total_loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:

        if self.prune_aware:
            return self._compute_loss_staged(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # Fallback: single-pass with hierarchical attention mask
        inputs['attention_mask'] = self._prepare_attention_mask(inputs)

        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        return (loss, outputs) if return_outputs else loss
