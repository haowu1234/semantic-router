#!/usr/bin/env python3
"""
Complete training pipeline for DSL model.

Runs all three training stages in sequence:
1. Stage 1: Syntax pre-training
2. Stage 2: SFT (NL→DSL)
3. Stage 3: DPO (preference learning)

Usage:
    python train_all.py --config configs/base.yaml --data-dir ./prepared_data
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from omegaconf import OmegaConf

from utils.config import load_config, merge_configs, config_to_dict
from utils.logger import setup_logger, TrainingLogger
from models.dsl_model import load_model_and_tokenizer, create_peft_model, merge_and_save
from data.dataset import DSLDataset, SFTDataset, DPODataset
from trainers.base_trainer import DSLTrainer
from trainers.sft_trainer import DSLSFTTrainer
from trainers.dpo_trainer import DSLDPOTrainer
from evaluation.evaluator import DSLEvaluator


def find_latest_checkpoint(output_dir: Path) -> str | None:
    """Find the latest checkpoint in the output directory."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    
    # Look for checkpoint-* directories
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort by step number and get latest
    def get_step(p):
        try:
            return int(p.name.split("-")[1])
        except (IndexError, ValueError):
            return 0
    
    checkpoints.sort(key=get_step)
    latest = checkpoints[-1]
    
    return str(latest)


def train_stage1(
    config: dict,
    data_dir: Path,
    output_dir: Path,
    logger: TrainingLogger,
) -> str:
    """Run Stage 1: Syntax pre-training."""
    logger.logger.info("=" * 60)
    logger.logger.info("Stage 1: Syntax Pre-training")
    logger.logger.info("=" * 60)
    
    # Load stage config
    stage_config = load_config(Path(__file__).parent.parent / "configs" / "stage1_pt.yaml")
    config = merge_configs(config_to_dict(config), config_to_dict(stage_config))
    
    # Update output dir
    stage_output = output_dir / "stage1_pt"
    config['training']['output_dir'] = str(stage_output)
    
    # Load model and tokenizer
    logger.logger.info(f"Loading model: {config['model']['name']}")
    model, tokenizer = load_model_and_tokenizer(
        config['model']['name'],
        config,
    )
    
    # Create PEFT model
    model = create_peft_model(model, config)
    
    # Load datasets
    logger.logger.info("Loading datasets...")
    train_dataset = DSLDataset.from_jsonl(
        data_dir / "stage1_syntax_pt.jsonl",
        tokenizer,
        max_length=config['model']['max_length'],
    )
    eval_dataset = DSLDataset.from_jsonl(
        data_dir / "stage1_syntax_pt_eval.jsonl",
        tokenizer,
        max_length=config['model']['max_length'],
    )
    logger.logger.info(f"Train samples: {len(train_dataset)}")
    logger.logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Create trainer
    trainer = DSLTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    logger.logger.info("Starting training...")
    metrics = trainer.train()
    logger.log_eval(metrics)
    
    return str(stage_output)


def train_stage2(
    config: dict,
    data_dir: Path,
    output_dir: Path,
    logger: TrainingLogger,
    stage1_checkpoint: str = None,
    resume_from_checkpoint: str = None,
) -> str:
    """Run Stage 2: SFT training."""
    logger.logger.info("=" * 60)
    logger.logger.info("Stage 2: Supervised Fine-Tuning")
    logger.logger.info("=" * 60)
    
    # Load stage config
    stage_config = load_config(Path(__file__).parent.parent / "configs" / "stage2_sft.yaml")
    config = merge_configs(config_to_dict(config), config_to_dict(stage_config))
    
    # Update output dir
    stage_output = output_dir / "stage2_sft"
    config['training']['output_dir'] = str(stage_output)
    
    # Auto-detect checkpoint for resume
    if resume_from_checkpoint is True or resume_from_checkpoint == "auto":
        resume_from_checkpoint = find_latest_checkpoint(stage_output)
        if resume_from_checkpoint:
            logger.logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            logger.logger.info("No checkpoint found, starting fresh")
            resume_from_checkpoint = None
    
    # Load from stage1 checkpoint if available
    model_name = stage1_checkpoint if stage1_checkpoint else config['model']['name']
    
    logger.logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        config,
    )
    
    # Create PEFT model (or continue from stage1)
    if not stage1_checkpoint:
        model = create_peft_model(model, config)
    
    # Load datasets
    logger.logger.info("Loading datasets...")
    train_dataset = SFTDataset.from_jsonl(
        data_dir / "stage2_sft.jsonl",
        tokenizer,
        max_length=config['model']['max_length'],
    )
    eval_dataset = SFTDataset.from_jsonl(
        data_dir / "stage2_sft_eval.jsonl",
        tokenizer,
        max_length=config['model']['max_length'],
    )
    logger.logger.info(f"Train samples: {len(train_dataset)}")
    logger.logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Create trainer
    trainer = DSLSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    logger.logger.info("Starting training...")
    metrics = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.log_eval(metrics)
    
    return str(stage_output)


def train_stage3(
    config: dict,
    data_dir: Path,
    output_dir: Path,
    logger: TrainingLogger,
    stage2_checkpoint: str = None,
    early_stopping: bool | dict = True,  # Enable early stopping by default
) -> str:
    """Run Stage 3: DPO training."""
    logger.logger.info("=" * 60)
    logger.logger.info("Stage 3: Direct Preference Optimization")
    logger.logger.info("=" * 60)
    
    # Load stage config
    stage_config = load_config(Path(__file__).parent.parent / "configs" / "stage3_dpo.yaml")
    config = merge_configs(config_to_dict(config), config_to_dict(stage_config))
    
    # Update output dir
    stage_output = output_dir / "stage3_dpo"
    config['training']['output_dir'] = str(stage_output)
    
    # Determine base model name (always need base model for PEFT)
    base_model_name = config['model']['name']
    
    # Check if stage2_checkpoint is a PEFT checkpoint
    is_peft_checkpoint = False
    if stage2_checkpoint:
        checkpoint_path = Path(stage2_checkpoint)
        # PEFT checkpoints have adapter_config.json
        if (checkpoint_path / "adapter_config.json").exists():
            is_peft_checkpoint = True
            logger.logger.info(f"Detected PEFT checkpoint at: {stage2_checkpoint}")
        else:
            # It might be a merged/full model
            logger.logger.info(f"Detected full model checkpoint at: {stage2_checkpoint}")
            base_model_name = stage2_checkpoint
    
    # Load base model and tokenizer
    logger.logger.info(f"Loading base model: {base_model_name}")
    model, tokenizer = load_model_and_tokenizer(
        base_model_name,
        config,
    )
    
    # Handle PEFT model loading
    if is_peft_checkpoint:
        # Load PEFT adapter from stage2 checkpoint
        from models.dsl_model import load_peft_model
        logger.logger.info(f"Loading PEFT adapter from: {stage2_checkpoint}")
        model = load_peft_model(model, stage2_checkpoint, is_trainable=True)
        model.print_trainable_parameters()
    else:
        # Create new PEFT model
        logger.logger.info("Creating new PEFT model for DPO training...")
        model = create_peft_model(model, config)
    
    # Load reference model (for DPO) - should match the training model's starting point
    logger.logger.info("Loading reference model...")
    ref_model, _ = load_model_and_tokenizer(
        base_model_name,
        config,
    )
    
    # If we have a PEFT checkpoint, load it for reference model too (but frozen)
    if is_peft_checkpoint:
        from models.dsl_model import load_peft_model
        logger.logger.info(f"Loading PEFT adapter for reference model from: {stage2_checkpoint}")
        ref_model = load_peft_model(ref_model, stage2_checkpoint, is_trainable=False)
    
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Load datasets
    logger.logger.info("Loading datasets...")
    train_dpo_dataset = DPODataset.from_jsonl(
        data_dir / "stage3_dpo.jsonl",
        tokenizer,
        max_length=config['model']['max_length'],
    )
    eval_dpo_dataset = DPODataset.from_jsonl(
        data_dir / "stage3_dpo_eval.jsonl",
        tokenizer,
        max_length=config['model']['max_length'],
    )
    logger.logger.info(f"Train samples: {len(train_dpo_dataset)}")
    logger.logger.info(f"Eval samples: {len(eval_dpo_dataset)}")
    
    # Setup early stopping
    early_stopping_config = None
    if early_stopping:
        if isinstance(early_stopping, dict):
            early_stopping_config = early_stopping
        else:
            # Default early stopping configuration (recommended)
            early_stopping_config = {
                'strategy': 'combined',
                'accuracy_threshold': 0.95,
                'margin_threshold': 2.5,          # Stop if margins get too high
                'max_logps_drift': 100.0,         # Stop if model drifts too far
                'min_steps': 50,                  # Don't stop too early
                'combined_accuracy': 0.90,        # Combined: need at least 90%
                'combined_margin_min': 0.5,       # Combined: margin at least 0.5
                'combined_margin_max': 3.0,       # Combined: margin at most 3.0
                'combined_max_logps_drift': 80.0, # Combined: drift at most 80
                'verbose': True,
            }
        logger.logger.info(f"Early stopping enabled with config: {early_stopping_config}")
    
    # Create trainer (use HF Dataset for DPOTrainer compatibility)
    trainer = DSLDPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dpo_dataset.get_hf_dataset(),
        eval_dataset=eval_dpo_dataset.get_hf_dataset(),
        early_stopping=early_stopping_config,
    )
    
    # Train
    logger.logger.info("Starting training...")
    metrics = trainer.train()
    logger.log_eval(metrics)
    
    return str(stage_output)


def run_evaluation(
    config: dict,
    model_path: str,
    data_dir: Path,
    output_dir: Path,
    logger: TrainingLogger,
) -> None:
    """Run final evaluation."""
    logger.logger.info("=" * 60)
    logger.logger.info("Final Evaluation")
    logger.logger.info("=" * 60)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path, config)
    model.eval()
    
    # Create evaluator
    evaluator = DSLEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    
    # Run evaluation
    result = evaluator.evaluate(
        eval_file=data_dir / "eval_benchmark.jsonl",
        output_file=output_dir / "evaluation_results.json",
    )
    
    # Print summary
    evaluator.print_summary(result)


def main():
    parser = argparse.ArgumentParser(description='Train DSL model (all stages)')
    parser.add_argument('--config', type=Path, default=Path('configs/base.yaml'),
                        help='Base configuration file')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory with prepared training data')
    parser.add_argument('--output-dir', type=Path, default=Path('./checkpoints'),
                        help='Output directory for checkpoints')
    parser.add_argument('--stages', type=str, default='1,2,3',
                        help='Comma-separated list of stages to run (e.g., "1,2" or "3")')
    parser.add_argument('--stage1-checkpoint', type=str, default=None,
                        help='Checkpoint to use for stage 2 (skip stage 1)')
    parser.add_argument('--stage2-checkpoint', type=str, default=None,
                        help='Checkpoint to use for stage 3 (skip stages 1,2)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint in output dir')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip final evaluation')
    parser.add_argument('--log-dir', type=Path, default=Path('./logs'),
                        help='Directory for log files')
    
    # Training hyperparameters (override config)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Per-device train batch size (overrides config)')
    parser.add_argument('--eval-batch-size', type=int, default=None,
                        help='Per-device eval batch size (overrides config, default: same as batch-size)')
    parser.add_argument('--grad-accum', type=int, default=None,
                        help='Gradient accumulation steps (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Max training steps, -1 for full epochs (overrides config)')
    parser.add_argument('--save-steps', type=int, default=None,
                        help='Save checkpoint every N steps (overrides config)')
    
    # Early stopping options (Stage 3 DPO)
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='Disable early stopping for DPO training')
    parser.add_argument('--early-stop-accuracy', type=float, default=0.95,
                        help='Early stop when accuracy reaches this threshold (default: 0.95)')
    parser.add_argument('--early-stop-margin', type=float, default=2.5,
                        help='Early stop when margin exceeds this threshold (default: 2.5)')
    parser.add_argument('--early-stop-drift', type=float, default=100.0,
                        help='Early stop when logps drift exceeds this (default: 100.0)')
    parser.add_argument('--early-stop-min-steps', type=int, default=50,
                        help='Minimum steps before early stopping can trigger (default: 50)')
    
    args = parser.parse_args()
    
    # Setup logging
    base_logger = setup_logger(
        name="dsl_training",
        log_dir=args.log_dir,
    )
    logger = TrainingLogger(base_logger)
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(config_path)
    
    # Apply command-line overrides
    config_dict = config_to_dict(config)
    if 'training' not in config_dict:
        config_dict['training'] = {}
    
    if args.batch_size is not None:
        config_dict['training']['per_device_train_batch_size'] = args.batch_size
        base_logger.info(f"Override: batch_size = {args.batch_size}")
    
    # Set eval batch size (defaults to train batch size if not specified)
    if args.eval_batch_size is not None:
        config_dict['training']['per_device_eval_batch_size'] = args.eval_batch_size
        base_logger.info(f"Override: eval_batch_size = {args.eval_batch_size}")
    elif args.batch_size is not None:
        # Default eval batch size to train batch size for DPO (memory-safe)
        config_dict['training']['per_device_eval_batch_size'] = args.batch_size
        base_logger.info(f"Override: eval_batch_size = {args.batch_size} (same as train)")
    
    if args.grad_accum is not None:
        config_dict['training']['gradient_accumulation_steps'] = args.grad_accum
        base_logger.info(f"Override: gradient_accumulation_steps = {args.grad_accum}")
    if args.learning_rate is not None:
        config_dict['training']['learning_rate'] = args.learning_rate
        base_logger.info(f"Override: learning_rate = {args.learning_rate}")
    if args.max_steps is not None:
        config_dict['training']['max_steps'] = args.max_steps
        base_logger.info(f"Override: max_steps = {args.max_steps}")
    if args.save_steps is not None:
        config_dict['training']['save_steps'] = args.save_steps
        config_dict['training']['eval_steps'] = args.save_steps  # Keep eval aligned
        base_logger.info(f"Override: save_steps = {args.save_steps}")
    
    # Convert back to OmegaConf
    config = OmegaConf.create(config_dict)
    
    logger.log_config(config_to_dict(config))
    
    # Parse stages
    stages = [int(s.strip()) for s in args.stages.split(',')]
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run stages
    stage1_checkpoint = args.stage1_checkpoint
    stage2_checkpoint = args.stage2_checkpoint
    final_checkpoint = None
    
    if 1 in stages and not stage1_checkpoint:
        stage1_checkpoint = train_stage1(
            config=config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            logger=logger,
        )
        final_checkpoint = stage1_checkpoint
    
    if 2 in stages and not stage2_checkpoint:
        stage2_checkpoint = train_stage2(
            config=config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            logger=logger,
            stage1_checkpoint=stage1_checkpoint,
            resume_from_checkpoint="auto" if args.resume else None,
        )
        final_checkpoint = stage2_checkpoint
    
    if 3 in stages:
        # Build early stopping config from command line args
        early_stopping_config = None
        if not args.no_early_stopping:
            early_stopping_config = {
                'strategy': 'combined',
                'accuracy_threshold': args.early_stop_accuracy,
                'margin_threshold': args.early_stop_margin,
                'max_logps_drift': args.early_stop_drift,
                'min_steps': args.early_stop_min_steps,
                'combined_accuracy': 0.90,
                'combined_margin_min': 0.5,
                'combined_margin_max': args.early_stop_margin,
                'combined_max_logps_drift': args.early_stop_drift * 0.8,  # 80% of max
                'verbose': True,
            }
        
        final_checkpoint = train_stage3(
            config=config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            logger=logger,
            stage2_checkpoint=stage2_checkpoint,
            early_stopping=early_stopping_config,
        )
    
    # Final evaluation
    if not args.skip_eval and final_checkpoint:
        run_evaluation(
            config=config,
            model_path=final_checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            logger=logger,
        )
    
    logger.logger.info("=" * 60)
    logger.logger.info("Training Complete!")
    logger.logger.info(f"Final checkpoint: {final_checkpoint}")
    logger.logger.info("=" * 60)


if __name__ == '__main__':
    main()
