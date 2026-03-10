#!/usr/bin/env python3
"""
Base trainer for DSL model training.

Wraps HuggingFace Trainer with DSL-specific functionality.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

from ..data.collator import DSLDataCollator


class DSLTrainer:
    """
    Base trainer for DSL model training (Stage 1 and general usage).
    
    Handles:
    - Training loop setup
    - Checkpointing
    - Logging to wandb/tensorboard
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
        self.training_args = self._build_training_args()
        
        # Build data collator
        self.data_collator = DSLDataCollator(
            tokenizer=tokenizer,
            padding=True,
            max_length=config.get('model', {}).get('max_length', 2048),
        )
        
        # Initialize trainer
        self.trainer = self._build_trainer()
    
    def _build_training_args(self) -> TrainingArguments:
        """Build HuggingFace TrainingArguments from config."""
        training_config = self.config.get('training', {})
        wandb_config = self.config.get('wandb', {})
        
        # Determine output directory
        output_dir = training_config.get('output_dir', './checkpoints')
        
        # Build args
        args = TrainingArguments(
            output_dir=output_dir,
            
            # Batch size
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 8),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
            
            # Learning rate
            learning_rate=training_config.get('learning_rate', 2e-5),
            weight_decay=training_config.get('weight_decay', 0.01),
            
            # Epochs/Steps
            num_train_epochs=training_config.get('num_train_epochs', 3),
            max_steps=training_config.get('max_steps', -1),
            
            # Warmup
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
            
            # Optimization
            optim=training_config.get('optim', 'adamw_torch'),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            
            # Logging
            logging_steps=training_config.get('logging_steps', 10),
            logging_first_step=True,
            
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
            
            # Gradient checkpointing
            gradient_checkpointing=self.config.get('hardware', {}).get('gradient_checkpointing', True),
            
            # Wandb
            report_to=['wandb'] if wandb_config.get('enabled', False) else ['tensorboard'],
            run_name=wandb_config.get('name'),
        )
        
        return args
    
    def _build_trainer(self) -> Trainer:
        """Build HuggingFace Trainer."""
        callbacks = []
        
        # Add early stopping if configured
        if self.config.get('training', {}).get('early_stopping_patience'):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['training']['early_stopping_patience']
                )
            )
        
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks if callbacks else None,
        )
        
        return trainer
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> dict:
        """
        Run training.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        
        Returns:
            Training metrics
        """
        result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        self.save()
        
        return result.metrics
    
    def evaluate(self) -> dict:
        """Run evaluation on eval dataset."""
        return self.trainer.evaluate()
    
    def save(self, output_dir: Optional[str] = None) -> None:
        """Save model and tokenizer."""
        if output_dir is None:
            output_dir = self.training_args.output_dir
        
        # Save model
        self.trainer.save_model(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Training complete",
        private: bool = True,
    ) -> None:
        """Push model to HuggingFace Hub."""
        self.trainer.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
        )
