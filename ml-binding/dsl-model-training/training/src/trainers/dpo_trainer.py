#!/usr/bin/env python3
"""
DPO Trainer for Stage 3 (Preference Learning).

Uses trl's DPOTrainer for Direct Preference Optimization.
"""

from pathlib import Path
from typing import Optional, Union, List

from transformers import PreTrainedModel, PreTrainedTokenizer, TrainerCallback
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from trl import DPOTrainer, DPOConfig
from peft import PeftModel

from .early_stopping import (
    DPOEarlyStoppingCallback, 
    DPOEarlyStoppingConfig,
    create_early_stopping_callback,
)


def validate_and_clean_dpo_dataset(dataset: HFDataset) -> HFDataset:
    """
    Validate and clean DPO dataset to ensure all samples are valid.
    
    Removes samples with None, empty, or non-string values.
    """
    def is_valid_sample(sample):
        prompt = sample.get('prompt')
        chosen = sample.get('chosen')
        rejected = sample.get('rejected')
        
        # Check for None
        if prompt is None or chosen is None or rejected is None:
            return False
        
        # Check types
        if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
            return False
        
        # Check for empty strings
        if not prompt.strip() or not chosen.strip() or not rejected.strip():
            return False
        
        return True
    
    original_len = len(dataset)
    dataset = dataset.filter(is_valid_sample, desc="Validating DPO samples")
    new_len = len(dataset)
    
    if new_len < original_len:
        print(f"Removed {original_len - new_len} invalid samples during validation")
    
    return dataset


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
        early_stopping: Optional[Union[bool, DPOEarlyStoppingConfig, dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Build DPO config
        self.dpo_config = self._build_dpo_config()
        
        # Setup callbacks (including early stopping)
        self.callbacks = callbacks or []
        self.early_stopping_callback = None
        
        if early_stopping:
            self.early_stopping_callback = self._setup_early_stopping(early_stopping)
            self.callbacks.append(self.early_stopping_callback)
        
        # Initialize trainer
        self.trainer = self._build_trainer()
    
    def _setup_early_stopping(
        self, 
        early_stopping: Union[bool, DPOEarlyStoppingConfig, dict]
    ) -> DPOEarlyStoppingCallback:
        """Setup early stopping callback based on configuration."""
        if isinstance(early_stopping, bool) and early_stopping:
            # Use default combined strategy
            return create_early_stopping_callback(
                strategy="combined",
                accuracy_threshold=0.95,
                margin_threshold=2.5,
                max_logps_drift=100.0,
                min_steps=50,
            )
        elif isinstance(early_stopping, DPOEarlyStoppingConfig):
            return DPOEarlyStoppingCallback(early_stopping)
        elif isinstance(early_stopping, dict):
            # Build config from dict
            config = DPOEarlyStoppingConfig(**early_stopping)
            return DPOEarlyStoppingCallback(config)
        else:
            raise ValueError(f"Invalid early_stopping type: {type(early_stopping)}")
    
    def _build_dpo_config(self) -> DPOConfig:
        """Build DPO-specific training configuration."""
        training_config = self.config.get('training', {})
        dpo_config = self.config.get('dpo', {})
        data_config = self.config.get('data', {})
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
            
            # Max length settings - CRITICAL for proper tokenization
            max_length=self.config.get('model', {}).get('max_length', 2048),
            max_prompt_length=data_config.get('max_prompt_length', 1024),
            truncation_mode='keep_end',  # Keep the response end when truncating
            
            # Ensure correct model type detection
            is_encoder_decoder=False,
            
            # Remove columns after tokenization to avoid issues
            remove_unused_columns=False,
            
            # Wandb
            report_to=['wandb'] if wandb_config.get('enabled', False) else ['tensorboard'],
            run_name=wandb_config.get('name'),
        )
        
        return config
    
    def _build_trainer(self) -> DPOTrainer:
        """Build trl DPOTrainer. Let TRL handle tokenization internally."""
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # CRITICAL FIX: Ensure tokenizer has bos_token
        # Some models (like Qwen) don't have bos_token set, which causes:
        # TypeError: 'NoneType' object cannot be interpreted as an integer
        # when TRL tries to add bos_token to the input_ids
        if self.tokenizer.bos_token is None:
            print("WARNING: tokenizer.bos_token is None, setting it to eos_token")
            self.tokenizer.bos_token = self.tokenizer.eos_token
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
        
        # Set padding side to left for decoder-only models (important for DPO)
        self.tokenizer.padding_side = 'left'
        
        # Debug: print tokenizer special tokens
        print(f"Tokenizer special tokens:")
        print(f"  pad_token: {repr(self.tokenizer.pad_token)} (id={self.tokenizer.pad_token_id})")
        print(f"  bos_token: {repr(self.tokenizer.bos_token)} (id={self.tokenizer.bos_token_id})")
        print(f"  eos_token: {repr(self.tokenizer.eos_token)} (id={self.tokenizer.eos_token_id})")
        
        # Validate and clean datasets
        train_dataset = None
        eval_dataset = None
        
        if self.train_dataset is not None:
            print(f"Original train dataset columns: {self.train_dataset.column_names}")
            print(f"Original train dataset size: {len(self.train_dataset)}")
            
            # Show sample info
            if len(self.train_dataset) > 0:
                first_sample = self.train_dataset[0]
                print(f"First sample keys: {list(first_sample.keys())}")
                for k, v in first_sample.items():
                    if isinstance(v, str):
                        print(f"  {k}: type=str, len={len(v)}, preview={repr(v[:50])}...")
                    else:
                        print(f"  {k}: type={type(v).__name__}, value={v}")
            
            # Validate and clean
            print("Validating train dataset...")
            train_dataset = validate_and_clean_dpo_dataset(self.train_dataset)
            print(f"Validated train dataset size: {len(train_dataset)}")
            
        if self.eval_dataset is not None:
            print("Validating eval dataset...")
            eval_dataset = validate_and_clean_dpo_dataset(self.eval_dataset)
            print(f"Validated eval dataset size: {len(eval_dataset)}")
        
        # Build trainer - TRL will handle tokenization of raw text data
        # Note: newer TRL versions use 'processing_class' instead of 'tokenizer'
        # We try both for compatibility
        try:
            trainer = DPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=self.dpo_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=self.tokenizer,  # New TRL API
                callbacks=self.callbacks if self.callbacks else None,
            )
        except TypeError:
            # Fallback to old API for older TRL versions
            trainer = DPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=self.dpo_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,  # Old TRL API
                callbacks=self.callbacks if self.callbacks else None,
            )
        
        return trainer
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> dict:
        """Run DPO training."""
        result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save early stopping history if enabled
        if self.early_stopping_callback:
            self.early_stopping_callback.save_history(self.dpo_config.output_dir)
        
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
