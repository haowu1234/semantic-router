#!/usr/bin/env python3
"""
Data collators for DSL model training.

Handles padding and batching for different training stages.
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers import PreTrainedTokenizer


@dataclass
class DSLDataCollator:
    """
    Data collator for Stage 1 (syntax pre-training) and Stage 2 (SFT).
    
    Handles dynamic padding to longest sequence in batch.
    """
    
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Separate labels from features
        labels = [f.pop('labels') for f in features] if 'labels' in features[0] else None
        
        # Pad input features
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        
        # Pad labels separately
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            
            padded_labels = []
            for label in labels:
                padding_length = max_label_length - len(label)
                padded_label = label + [self.label_pad_token_id] * padding_length
                padded_labels.append(padded_label)
            
            batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch


SFTDataCollator = DSLDataCollator  # Same implementation


@dataclass
class DPODataCollator:
    """
    Data collator for Stage 3 (DPO training).
    
    Handles padding for prompt, chosen, and rejected sequences.
    """
    
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Extract different parts
        prompt_input_ids = [f['prompt_input_ids'] for f in features]
        prompt_attention_mask = [f['prompt_attention_mask'] for f in features]
        chosen_input_ids = [f['chosen_input_ids'] for f in features]
        chosen_attention_mask = [f['chosen_attention_mask'] for f in features]
        rejected_input_ids = [f['rejected_input_ids'] for f in features]
        rejected_attention_mask = [f['rejected_attention_mask'] for f in features]
        
        # Pad each part
        batch = {}
        
        # Pad prompts
        padded_prompts = self._pad_sequences(prompt_input_ids, prompt_attention_mask, 'prompt')
        batch.update(padded_prompts)
        
        # Pad chosen
        padded_chosen = self._pad_sequences(chosen_input_ids, chosen_attention_mask, 'chosen')
        batch.update(padded_chosen)
        
        # Pad rejected
        padded_rejected = self._pad_sequences(rejected_input_ids, rejected_attention_mask, 'rejected')
        batch.update(padded_rejected)
        
        return batch
    
    def _pad_sequences(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        """Pad a list of sequences."""
        max_length = max(len(ids) for ids in input_ids)
        
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        
        if self.max_length is not None:
            max_length = min(max_length, self.max_length)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        pad_token_id = self.tokenizer.pad_token_id or 0
        
        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            
            # Right padding
            padded_ids = ids + [pad_token_id] * padding_length
            padded_mask = mask + [0] * padding_length
            
            # Truncate if needed
            padded_input_ids.append(padded_ids[:max_length])
            padded_attention_mask.append(padded_mask[:max_length])
        
        return {
            f'{prefix}_input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            f'{prefix}_attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
        }
