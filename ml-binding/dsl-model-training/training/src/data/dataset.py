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
        
        # Build conversation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{instruction}\n\n{input_text}"},
            {"role": "assistant", "content": output_text},
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # Create labels: mask prompt tokens with -100
        labels = encodings['input_ids'].copy()
        
        # Find where assistant response starts
        # For Qwen template: <|im_start|>assistant\n
        assistant_marker = "<|im_start|>assistant"
        text_up_to_assistant = text.split(assistant_marker)[0] + assistant_marker
        prompt_length = len(self.tokenizer.encode(text_up_to_assistant, add_special_tokens=False))
        
        # Mask prompt tokens
        for i in range(min(prompt_length, len(labels))):
            labels[i] = -100
        
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
class DPODataset(Dataset):
    """
    Stage 3: DPO dataset for preference learning.
    
    Each sample contains:
    - id: unique identifier
    - prompt: the prompt/question
    - chosen: preferred (correct) response
    - rejected: dispreferred (incorrect) response
    - mutation_type: type of error in rejected
    - mutation_category: category of error
    """
    
    samples: list[dict]
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    system_prompt: str = "You are a Signal DSL configuration generator. Generate valid Signal DSL configurations."
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        prompt = sample.get('prompt', 'Generate a valid Signal DSL configuration.')
        chosen = sample['chosen']
        rejected = sample['rejected']
        
        # Build prompt with system message
        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        # Apply chat template for prompt only
        formatted_prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize prompt
        prompt_encodings = self.tokenizer(
            formatted_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # Tokenize chosen response
        chosen_encodings = self.tokenizer(
            chosen,
            max_length=self.max_length - len(prompt_encodings['input_ids']),
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # Tokenize rejected response
        rejected_encodings = self.tokenizer(
            rejected,
            max_length=self.max_length - len(prompt_encodings['input_ids']),
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        return {
            'prompt_input_ids': prompt_encodings['input_ids'],
            'prompt_attention_mask': prompt_encodings['attention_mask'],
            'chosen_input_ids': chosen_encodings['input_ids'],
            'chosen_attention_mask': chosen_encodings['attention_mask'],
            'rejected_input_ids': rejected_encodings['input_ids'],
            'rejected_attention_mask': rejected_encodings['attention_mask'],
        }
    
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
