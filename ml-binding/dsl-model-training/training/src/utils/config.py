#!/usr/bin/env python3
"""
Configuration loading and management utilities.
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        OmegaConf DictConfig object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Handle defaults (inheritance)
    if 'defaults' in config_dict:
        defaults = config_dict.pop('defaults')
        base_config = {}
        
        for default in defaults:
            if isinstance(default, str):
                # Load base config from same directory
                base_path = config_path.parent / f"{default}.yaml"
                if base_path.exists():
                    base_config = load_config(base_path)
        
        # Merge: base <- current
        config_dict = merge_configs(dict(base_config), config_dict)
    
    return OmegaConf.create(config_dict)


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep merge two config dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration (takes precedence)
    
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def config_to_dict(config: DictConfig) -> dict:
    """Convert OmegaConf config to plain dict."""
    return OmegaConf.to_container(config, resolve=True)


def save_config(config: DictConfig | dict, output_path: str | Path) -> None:
    """Save configuration to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(config, DictConfig):
        config = config_to_dict(config)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_training_args_from_config(config: DictConfig) -> dict:
    """Extract training arguments from config."""
    training = dict(config.get('training', {}))
    
    # Add model-related args
    if 'model' in config:
        training['max_length'] = config.model.get('max_length', 2048)
    
    # Add hardware args
    if 'hardware' in config:
        training['gradient_checkpointing'] = config.hardware.get('gradient_checkpointing', False)
    
    return training
