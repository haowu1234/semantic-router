#!/usr/bin/env python3
"""
Dataset classes for DSL model training.

Supports three formats:
1. Stage 1 (Completion): Pure DSL for syntax pre-training
2. Stage 2 (SFT): Instruction-Input-Output for NL→DSL
3. Stage 3 (DPO): Preference pairs (chosen, rejected)
"""

import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import Dataset as HFDataset


def load_dataset_from_jsonl(file_path: str | Path) -> list[dict]:
    """Load dataset from JSONL file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


@dataclass
class DSLDataset(Dataset):
    """
    Stage 1: Pure DSL completion dataset for syntax pre-training.
    
    Each sample contains:
    - id: unique identifier
    - dsl: DSL configuration text
    - complexity: L1-L5 complexity level
    """
    
    samples: list[dict]
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    template: str = "{dsl}"
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Format text using template
        text = self.template.format(dsl=sample['dsl'])
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # For causal LM, labels = input_ids
        encodings['labels'] = encodings['input_ids'].copy()
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': encodings['labels'],
        }
    
    @classmethod
    def from_jsonl(
        cls,
        file_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        template: str = "{dsl}",
        max_samples: Optional[int] = None,
    ) -> "DSLDataset":
        """Create dataset from JSONL file."""
        samples = load_dataset_from_jsonl(file_path)
        if max_samples is not None:
            samples = samples[:max_samples]
        return cls(
            samples=samples,
            tokenizer=tokenizer,
            max_length=max_length,
            template=template,
        )


@dataclass
class SFTDataset(Dataset):
    """
    Stage 2: SFT dataset for NL→DSL instruction following.
    
    Each sample contains:
    - id: unique identifier
    - instruction: task instruction
    - input: natural language description
    - output: target DSL configuration
    - style: NL style (en_formal, zh_casual, etc.)
    - complexity: L1-L5 complexity level
    """
    
    samples: list[dict]
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    system_prompt: str = "You are a Signal DSL configuration generator."
    default_instruction: str = "Convert the following natural language description into Signal DSL configuration."
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        instruction = sample.get('instruction', self.default_instruction)
        input_text = sample.get('input', '')
        output_text = sample.get('output', sample.get('dsl', ''))
        
        # Build conversation WITHOUT assistant response for prompt
        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{instruction}\n\n{input_text}"},
        ]
        
        # Get prompt text with generation prompt
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,  # Adds <|im_start|>assistant\n
        )
        
        # Build full conversation
        full_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{instruction}\n\n{input_text}"},
            {"role": "assistant", "content": output_text},
        ]
        
        # Full text
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Tokenize prompt to get exact length
        prompt_encodings = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        prompt_length = len(prompt_encodings['input_ids'])
        
        # Tokenize full text
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # Create labels: mask prompt tokens with -100
        labels = encodings['input_ids'].copy()
        
        # Mask prompt tokens (keep only assistant response for loss)
        for i in range(min(prompt_length, len(labels))):
            labels[i] = -100
        
        # Safety check: ensure we have some non-masked labels
        non_masked = sum(1 for l in labels if l != -100)
        if non_masked == 0:
            # Fallback: use all tokens for loss (shouldn't happen normally)
            labels = encodings['input_ids'].copy()
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels,
        }
    
    @classmethod
    def from_jsonl(
        cls,
        file_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        system_prompt: Optional[str] = None,
        default_instruction: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> "SFTDataset":
        """Create dataset from JSONL file."""
        samples = load_dataset_from_jsonl(file_path)
        if max_samples is not None:
            samples = samples[:max_samples]
        
        kwargs = {
            'samples': samples,
            'tokenizer': tokenizer,
            'max_length': max_length,
        }
        if system_prompt is not None:
            kwargs['system_prompt'] = system_prompt
        if default_instruction is not None:
            kwargs['default_instruction'] = default_instruction
            
        return cls(**kwargs)


@dataclass  
class DPODataset:
    """
    Stage 3: DPO dataset for preference learning.
    
    Returns a HuggingFace Dataset compatible with trl.DPOTrainer.
    
    TRL DPOTrainer expects data in one of these formats:
    - Standard format: {"prompt": str, "chosen": str, "rejected": str}
    - Conversational format: {"prompt": list[dict], "chosen": list[dict], "rejected": list[dict]}
    
    We use the conversational format for better compatibility.
    """
    
    samples: list[dict]
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    system_prompt: str = "You are a Signal DSL configuration generator. Generate valid Signal DSL configurations."
    _hf_dataset: Optional[HFDataset] = None
    
    def __post_init__(self):
        """Convert to HuggingFace Dataset format after initialization."""
        self._hf_dataset = self._create_hf_dataset()
    
    def _create_hf_dataset(self) -> HFDataset:
        """
        Create HuggingFace Dataset with proper format for DPOTrainer.
        
        Uses conversational format where prompt/chosen/rejected are message lists.
        This is the recommended format for TRL >= 0.8.0.
        """
        processed_samples = []
        
        for sample in self.samples:
            user_prompt = sample.get('prompt', 'Generate a valid Signal DSL configuration.')
            chosen = sample['chosen']
            rejected = sample['rejected']
            
            # Skip invalid samples
            if not chosen or not rejected:
                continue
            
            # Build prompt as message list (without assistant response)
            prompt_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            # Chosen and rejected as message lists (assistant responses only)
            chosen_messages = [
                {"role": "assistant", "content": chosen},
            ]
            rejected_messages = [
                {"role": "assistant", "content": rejected},
            ]
            
            processed_samples.append({
                'prompt': prompt_messages,
                'chosen': chosen_messages,
                'rejected': rejected_messages,
            })
        
        return HFDataset.from_list(processed_samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """For compatibility with PyTorch Dataset interface."""
        return self._hf_dataset[idx]
    
    def get_hf_dataset(self) -> HFDataset:
        """Get the underlying HuggingFace Dataset."""
        return self._hf_dataset
    
    # Delegate common Dataset methods to HF Dataset
    def map(self, *args, **kwargs):
        """Delegate map to HF Dataset."""
        return self._hf_dataset.map(*args, **kwargs)
    
    def filter(self, *args, **kwargs):
        """Delegate filter to HF Dataset."""
        return self._hf_dataset.filter(*args, **kwargs)
    
    def select(self, *args, **kwargs):
        """Delegate select to HF Dataset."""
        return self._hf_dataset.select(*args, **kwargs)
    
    def shuffle(self, *args, **kwargs):
        """Delegate shuffle to HF Dataset."""
        return self._hf_dataset.shuffle(*args, **kwargs)
    
    @property
    def column_names(self):
        """Return column names from HF Dataset."""
        return self._hf_dataset.column_names
    
    @classmethod
    def from_jsonl(
        cls,
        file_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        system_prompt: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> "DPODataset":
        """Create dataset from JSONL file."""
        samples = load_dataset_from_jsonl(file_path)
        if max_samples is not None:
            samples = samples[:max_samples]
        
        kwargs = {
            'samples': samples,
            'tokenizer': tokenizer,
            'max_length': max_length,
        }
        if system_prompt is not None:
            kwargs['system_prompt'] = system_prompt
            
        return cls(**kwargs)


def create_dataset(
    stage: str,
    file_path: str | Path,
    tokenizer: PreTrainedTokenizer,
    config: dict,
) -> Dataset:
    """Factory function to create appropriate dataset for each stage."""
    
    max_length = config.get('model', {}).get('max_length', 2048)
    max_samples = config.get('data', {}).get('max_train_samples')
    
    if stage == "stage1_syntax_pt":
        template = config.get('data', {}).get('template', '{dsl}')
        return DSLDataset.from_jsonl(
            file_path=file_path,
            tokenizer=tokenizer,
            max_length=max_length,
            template=template,
            max_samples=max_samples,
        )
    
    elif stage == "stage2_sft":
        system_prompt = config.get('data', {}).get('system_prompt')
        default_instruction = config.get('data', {}).get('default_instruction')
        return SFTDataset.from_jsonl(
            file_path=file_path,
            tokenizer=tokenizer,
            max_length=max_length,
            system_prompt=system_prompt,
            default_instruction=default_instruction,
            max_samples=max_samples,
        )
    
    elif stage == "stage3_dpo":
        system_prompt = config.get('data', {}).get('system_prompt')
        return DPODataset.from_jsonl(
            file_path=file_path,
            tokenizer=tokenizer,
            max_length=max_length,
            system_prompt=system_prompt,
            max_samples=max_samples,
        )
    
    else:
        raise ValueError(f"Unknown stage: {stage}")
