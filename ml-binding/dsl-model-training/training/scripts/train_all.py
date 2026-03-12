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
    
    # Load from stage2 checkpoint if available
    model_name = stage2_checkpoint if stage2_checkpoint else config['model']['name']
    
    logger.logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        config,
    )
    
    # CRITICAL FIX: Create PEFT model for training
    # Without this, the model has no trainable parameters and grad_norm will be 0
    logger.logger.info("Creating PEFT model for DPO training...")
    model = create_peft_model(model, config)
    
    # Load reference model (for DPO)
    logger.logger.info("Loading reference model...")
    ref_model, _ = load_model_and_tokenizer(
        model_name,
        config,
    )
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
    
    # Create trainer (use HF Dataset for DPOTrainer compatibility)
    trainer = DSLDPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dpo_dataset.get_hf_dataset(),
        eval_dataset=eval_dpo_dataset.get_hf_dataset(),
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
    parser.add_argument('--grad-accum', type=int, default=None,
                        help='Gradient accumulation steps (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Max training steps, -1 for full epochs (overrides config)')
    parser.add_argument('--save-steps', type=int, default=None,
                        help='Save checkpoint every N steps (overrides config)')
    
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
        final_checkpoint = train_stage3(
            config=config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            logger=logger,
            stage2_checkpoint=stage2_checkpoint,
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
