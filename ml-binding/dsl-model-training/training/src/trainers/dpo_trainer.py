#!/usr/bin/env python3
"""
DPO Trainer for Stage 3 (Preference Learning).

Uses trl's DPOTrainer for Direct Preference Optimization.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from trl import DPOTrainer, DPOConfig
from peft import PeftModel


class DSLDPOTrainer:
    """
    DPO Trainer for Stage 3: Preference learning to avoid common errors.
    
    Uses trl's DPOTrainer to learn from (chosen, rejected) pairs:
    - chosen: valid DSL configurations
    - rejected: DSL with various errors (syntax, reference, etc.)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: Optional[PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        config: dict,
        train_dataset: Optional[Union[HFDataset, TorchDataset]] = None,
        eval_dataset: Optional[Union[HFDataset, TorchDataset]] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Build DPO config
        self.dpo_config = self._build_dpo_config()
        
        # Initialize trainer
        self.trainer = self._build_trainer()
    
    def _build_dpo_config(self) -> DPOConfig:
        """Build DPO-specific training configuration."""
        training_config = self.config.get('training', {})
        dpo_config = self.config.get('dpo', {})
        wandb_config = self.config.get('wandb', {})
        
        output_dir = training_config.get('output_dir', './checkpoints/stage3_dpo')
        
        config = DPOConfig(
            output_dir=output_dir,
            
            # DPO specific
            beta=dpo_config.get('beta', 0.1),
            loss_type=dpo_config.get('loss_type', 'sigmoid'),
            label_smoothing=dpo_config.get('label_smoothing', 0.0),
            
            # Batch size (smaller for DPO due to memory)
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 2),
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
            
            # Learning rate (lower for DPO)
            learning_rate=training_config.get('learning_rate', 5e-6),
            weight_decay=training_config.get('weight_decay', 0.01),
            
            # Epochs (fewer for DPO)
            num_train_epochs=training_config.get('num_train_epochs', 1),
            max_steps=training_config.get('max_steps', -1),
            
            # Warmup
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
            eval_strategy=training_config.get('eval_strategy', 'epoch'),
            
            # Precision
            bf16=training_config.get('bf16', True),
            fp16=training_config.get('fp16', False),
            
            # Reproducibility
            seed=training_config.get('seed', 42),
            
            # Gradient checkpointing
            gradient_checkpointing=self.config.get('hardware', {}).get('gradient_checkpointing', False),
            
            # Max length
            max_length=self.config.get('model', {}).get('max_length', 2048),
            max_prompt_length=1024,
            
            # Wandb
            report_to=['wandb'] if wandb_config.get('enabled', False) else ['tensorboard'],
            run_name=wandb_config.get('name'),
        )
        
        return config
    
    def _build_trainer(self) -> DPOTrainer:
        """Build trl DPOTrainer."""
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Set padding side for DPO (left padding is recommended)
        self.tokenizer.padding_side = 'left'
        
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=self.dpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        return trainer
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> dict:
        """Run DPO training."""
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
            output_dir = self.dpo_config.output_dir
        
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    @staticmethod
    def prepare_dpo_dataset(
        positive_file: str | Path,
        negative_file: str | Path,
        output_file: str | Path,
    ) -> int:
        """
        Prepare DPO dataset from positive and negative samples.
        
        Args:
            positive_file: JSONL file with valid DSL samples
            negative_file: JSONL file with negative (mutated) samples
            output_file: Output JSONL file in DPO format
        
        Returns:
            Number of preference pairs created
        """
        import json
        from pathlib import Path
        
        positive_file = Path(positive_file)
        negative_file = Path(negative_file)
        output_file = Path(output_file)
        
        # Load positive samples
        positives = {}
        with open(positive_file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                positives[sample['id']] = sample['dsl']
        
        # Load negative samples and match with positives
        pairs = []
        with open(negative_file, 'r') as f:
            for line in f:
                neg_sample = json.loads(line)
                
                # Get original ID (remove mutation suffix)
                orig_id = neg_sample.get('original_id', neg_sample['id'].rsplit('_', 2)[0])
                
                if orig_id in positives:
                    pair = {
                        'id': neg_sample['id'],
                        'prompt': 'Generate a valid Signal DSL configuration.',
                        'chosen': positives[orig_id],
                        'rejected': neg_sample['dsl'],
                        'mutation_type': neg_sample.get('mutation_type', 'unknown'),
                        'mutation_category': neg_sample.get('mutation_category', 'unknown'),
                    }
                    pairs.append(pair)
        
        # Save pairs
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        return len(pairs)
