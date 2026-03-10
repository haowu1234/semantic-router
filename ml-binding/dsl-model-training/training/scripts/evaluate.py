#!/usr/bin/env python3
"""
Evaluation script for trained DSL model.

Usage:
    python evaluate.py --model ./checkpoints/stage3_dpo --eval-data ./prepared_data/eval_benchmark.jsonl
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config
from utils.logger import setup_logger
from models.dsl_model import load_model_and_tokenizer
from evaluation.evaluator import DSLEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate DSL model')
    parser.add_argument('--model', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--eval-data', type=Path, required=True,
                        help='Path to evaluation data (JSONL)')
    parser.add_argument('--config', type=Path, default=Path('configs/base.yaml'),
                        help='Configuration file')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output file for results (JSON)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for generation')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate')
    parser.add_argument('--dsl-parser', type=str, default=None,
                        help='Path to DSL parser binary for validation')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(name="dsl_eval")
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(config_path)
    
    logger.info(f"Loading model from: {args.model}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        str(args.model),
        config,
    )
    model.eval()
    
    # Create evaluator
    evaluator = DSLEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        dsl_parser_path=args.dsl_parser,
    )
    
    # Run evaluation
    output_file = args.output or (args.model / "evaluation_results.json")
    
    logger.info(f"Running evaluation on: {args.eval_data}")
    result = evaluator.evaluate(
        eval_file=args.eval_data,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_file=output_file,
    )
    
    # Print summary
    evaluator.print_summary(result)
    
    logger.info(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
