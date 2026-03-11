#!/usr/bin/env python3
"""
SFT Trainer for Stage 2 (NL→DSL instruction fine-tuning).

Uses trl's SFTTrainer for efficient supervised fine-tuning.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from torch.utils.data import Dataset
from trl import SFTTrainer, SFTConfig

from data.collator import SFTDataCollator


class DSLSFTTrainer:
    """
    SFT Trainer for Stage 2: NL→DSL instruction following.
    
    Uses trl's SFTTrainer with:
    - Instruction masking (only compute loss on output)
    - Chat template formatting
    - Packing for efficiency (optional)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: dict,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        
        # Build training arguments
        self.training_args = self._build_sft_config()
        
        # Initialize trainer
        self.trainer = self._build_trainer()
    
    def _build_sft_config(self) -> SFTConfig:
        """Build SFT-specific training configuration."""
        training_config = self.config.get('training', {})
        data_config = self.config.get('data', {})
        wandb_config = self.config.get('wandb', {})
        
        output_dir = training_config.get('output_dir', './checkpoints/stage2_sft')
        
        # SFT specific args
        sft_config = SFTConfig(
            output_dir=output_dir,
            
            # Batch size
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 8),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
            
            # Learning rate
            learning_rate=training_config.get('learning_rate', 2e-5),
            weight_decay=training_config.get('weight_decay', 0.01),
            
            # Epochs
            num_train_epochs=training_config.get('num_train_epochs', 3),
            max_steps=training_config.get('max_steps', -1),
            
            # Warmup (warmup_steps takes priority over warmup_ratio)
            warmup_steps=training_config.get('warmup_steps', 0),
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
            
            # Optimization
            optim=training_config.get('optim', 'adamw_torch'),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            
            # Logging
            logging_steps=training_config.get('logging_steps', 10),
            
            # Checkpointing
            save_steps=training_config.get('save_steps', 500),
            save_total_limit=training_config.get('save_total_limit', 3),
            
            # Evaluation
            eval_strategy=training_config.get('eval_strategy', 'steps'),
            eval_steps=training_config.get('eval_steps', 500),
            
            # Best model
            load_best_model_at_end=training_config.get('load_best_model_at_end', True),
            metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=training_config.get('greater_is_better', False),
            
            # Precision
            bf16=training_config.get('bf16', True),
            fp16=training_config.get('fp16', False),
            
            # Reproducibility
            seed=training_config.get('seed', 42),
            data_seed=training_config.get('data_seed', 42),
            
            # Gradient checkpointing - 默认禁用，bf16+checkpointing+长序列在ROCm上会导致NaN
            gradient_checkpointing=self.config.get('hardware', {}).get('gradient_checkpointing', False),
            
            # SFT specific
            max_seq_length=self.config.get('model', {}).get('max_length', 2048),
            packing=False,  # Disable packing for variable length sequences
            
            # Wandb
            report_to=['wandb'] if wandb_config.get('enabled', False) else ['tensorboard'],
            run_name=wandb_config.get('name'),
        )
        
        return sft_config
    
    def _build_trainer(self) -> SFTTrainer:
        """Build trl SFTTrainer."""
        data_config = self.config.get('data', {})
        
        # Create data collator for proper padding
        data_collator = SFTDataCollator(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.config.get('model', {}).get('max_length', 2048),
        )
        
        trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        return trainer
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> dict:
        """Run SFT training."""
        result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        self.save()
        
        return result.metrics
    
    def evaluate(self) -> dict:
        """Run evaluation."""
        return self.trainer.evaluate()
    
    def save(self, output_dir: Optional[str] = None) -> None:
        """Save model and tokenizer."""
        if output_dir is None:
            output_dir = self.training_args.output_dir
        
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
