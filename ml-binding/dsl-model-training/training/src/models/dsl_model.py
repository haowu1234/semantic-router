#!/usr/bin/env python3
"""
Model loading and management utilities for DSL model training.

Supports:
- Base model loading (Qwen, DeepSeek, etc.)
- LoRA/PEFT configuration
- Model saving and merging
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


def load_model_and_tokenizer(
    model_name: str,
    config: dict,
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base model and tokenizer.
    
    Args:
        model_name: HuggingFace model name or local path
        config: Configuration dictionary
        device_map: Device mapping strategy
        load_in_8bit: Whether to load in 8-bit quantization
        load_in_4bit: Whether to load in 4-bit quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = config.get('model', {})
    tokenizer_config = config.get('tokenizer', {})
    
    # Quantization config
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Determine torch dtype
    torch_dtype_str = model_config.get('torch_dtype', 'bfloat16')
    torch_dtype = getattr(torch, torch_dtype_str, torch.bfloat16)
    
    # Attention implementation
    attn_implementation = model_config.get('attn_implementation', None)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get('trust_remote_code', True),
        revision=model_config.get('revision', 'main'),
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Apply tokenizer config
    if tokenizer_config.get('padding_side'):
        tokenizer.padding_side = tokenizer_config['padding_side']
    
    # Load model
    model_kwargs = {
        'trust_remote_code': model_config.get('trust_remote_code', True),
        'revision': model_config.get('revision', 'main'),
        'torch_dtype': torch_dtype,
        'device_map': device_map,
    }
    
    if quantization_config is not None:
        model_kwargs['quantization_config'] = quantization_config
    
    if attn_implementation:
        model_kwargs['attn_implementation'] = attn_implementation
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    
    # Enable gradient checkpointing if specified
    if config.get('hardware', {}).get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
    
    return model, tokenizer


def create_peft_model(
    model: PreTrainedModel,
    config: dict,
    is_trainable: bool = True,
) -> PeftModel:
    """
    Create PEFT (LoRA) model from base model.
    
    Args:
        model: Base model
        config: Configuration dictionary with LoRA settings
        is_trainable: Whether to make LoRA layers trainable
    
    Returns:
        PEFT model
    """
    lora_config = config.get('lora', {})
    
    if not lora_config.get('enabled', True):
        return model
    
    # Prepare model for k-bit training if quantized
    if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)
    elif hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Create LoRA config
    peft_config = LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('alpha', 32),
        lora_dropout=lora_config.get('dropout', 0.05),
        target_modules=lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias=lora_config.get('bias', 'none'),
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Create PEFT model
    model = get_peft_model(model, peft_config)
    
    if is_trainable:
        model.print_trainable_parameters()
    
    return model


def load_peft_model(
    base_model: PreTrainedModel,
    peft_model_path: str | Path,
    is_trainable: bool = False,
) -> PeftModel:
    """
    Load existing PEFT model from checkpoint.
    
    Args:
        base_model: Base model
        peft_model_path: Path to PEFT checkpoint
        is_trainable: Whether to make model trainable
    
    Returns:
        Loaded PEFT model
    """
    model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
        is_trainable=is_trainable,
    )
    
    return model


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str | Path,
    safe_serialization: bool = True,
) -> None:
    """
    Save model and tokenizer.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        safe_serialization: Use safetensors format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(
        output_dir,
        safe_serialization=safe_serialization,
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")


def merge_and_save(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str | Path,
    safe_serialization: bool = True,
) -> None:
    """
    Merge LoRA weights into base model and save.
    
    Args:
        model: PEFT model
        tokenizer: Tokenizer
        output_dir: Output directory
        safe_serialization: Use safetensors format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge LoRA weights
    merged_model = model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=safe_serialization,
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"Merged model saved to {output_dir}")


def get_model_size(model: PreTrainedModel) -> dict:
    """Get model parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'total_params_billions': total_params / 1e9,
        'trainable_params_millions': trainable_params / 1e6,
    }
