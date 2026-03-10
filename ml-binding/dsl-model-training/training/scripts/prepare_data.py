#!/usr/bin/env python3
"""
Data preparation script for DSL model training.

Prepares training data for all three stages:
1. Stage 1: Syntax pre-training data
2. Stage 2: SFT data (NL→DSL pairs)
3. Stage 3: DPO data (preference pairs)

Usage:
    python prepare_data.py --data-dir ../data --output-dir ./prepared_data
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm


def prepare_stage1_data(
    synthetic_dir: Path,
    output_dir: Path,
    eval_ratio: float = 0.1,
) -> dict:
    """
    Prepare Stage 1 syntax pre-training data.
    
    Takes synthetic DSL samples and formats them for causal LM training.
    """
    all_samples = []
    
    # Load all synthetic samples
    for jsonl_file in synthetic_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    all_samples.append({
                        'id': sample['id'],
                        'dsl': sample['dsl'],
                        'complexity': sample.get('complexity', 'L3'),
                    })
    
    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - eval_ratio))
    train_samples = all_samples[:split_idx]
    eval_samples = all_samples[split_idx:]
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "stage1_syntax_pt.jsonl", 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open(output_dir / "stage1_syntax_pt_eval.jsonl", 'w', encoding='utf-8') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return {
        'train': len(train_samples),
        'eval': len(eval_samples),
    }


def prepare_stage2_data(
    nl_pairs_dir: Path,
    synthetic_dir: Path,
    output_dir: Path,
    eval_ratio: float = 0.1,
) -> dict:
    """
    Prepare Stage 2 SFT data.
    
    Uses NL-DSL pairs for instruction fine-tuning.
    Falls back to synthetic data if NL pairs not available.
    """
    all_samples = []
    
    # Try to load NL pairs first
    if nl_pairs_dir.exists() and any(nl_pairs_dir.glob("*.jsonl")):
        for jsonl_file in nl_pairs_dir.glob("*.jsonl"):
            style = jsonl_file.stem  # e.g., en_formal, zh_casual
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        all_samples.append({
                            'id': sample.get('id', f"nl_{len(all_samples)}"),
                            'instruction': sample.get('instruction', 'Convert the following natural language description into Signal DSL configuration.'),
                            'input': sample.get('input', sample.get('nl', '')),
                            'output': sample.get('output', sample.get('dsl', '')),
                            'style': sample.get('style', style),
                            'complexity': sample.get('complexity', 'L3'),
                        })
    else:
        # Fallback: create simple instruction data from synthetic
        print("NL pairs not found, creating from synthetic data...")
        for jsonl_file in synthetic_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        # Create a simple instruction
                        all_samples.append({
                            'id': sample['id'],
                            'instruction': 'Generate a Signal DSL configuration.',
                            'input': f"Create a configuration with complexity level {sample.get('complexity', 'L3')}.",
                            'output': sample['dsl'],
                            'style': 'synthetic',
                            'complexity': sample.get('complexity', 'L3'),
                        })
    
    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - eval_ratio))
    train_samples = all_samples[:split_idx]
    eval_samples = all_samples[split_idx:]
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "stage2_sft.jsonl", 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open(output_dir / "stage2_sft_eval.jsonl", 'w', encoding='utf-8') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return {
        'train': len(train_samples),
        'eval': len(eval_samples),
    }


def prepare_stage3_data(
    synthetic_dir: Path,
    negative_dir: Path,
    output_dir: Path,
    eval_ratio: float = 0.1,
) -> dict:
    """
    Prepare Stage 3 DPO data.
    
    Creates preference pairs from positive (synthetic) and negative (mutated) samples.
    """
    # Load positive samples (indexed by ID for matching)
    positives = {}
    for jsonl_file in synthetic_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    positives[sample['id']] = sample['dsl']
    
    # Create preference pairs from negative samples
    all_pairs = []
    
    if negative_dir.exists():
        for jsonl_file in negative_dir.glob("*.jsonl"):
            category = jsonl_file.stem  # e.g., syntax_error, reference_error
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        neg_sample = json.loads(line)
                        
                        # Get original ID
                        orig_id = neg_sample.get('original_id')
                        if not orig_id:
                            # Try to extract from ID
                            parts = neg_sample['id'].rsplit('_', 2)
                            if len(parts) >= 2:
                                orig_id = '_'.join(parts[:-2])
                        
                        if orig_id and orig_id in positives:
                            pair = {
                                'id': neg_sample['id'],
                                'prompt': 'Generate a valid Signal DSL configuration.',
                                'chosen': positives[orig_id],
                                'rejected': neg_sample['dsl'],
                                'mutation_type': neg_sample.get('mutation_type', 'unknown'),
                                'mutation_category': neg_sample.get('mutation_category', category),
                            }
                            all_pairs.append(pair)
    
    # Shuffle and split
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * (1 - eval_ratio))
    train_pairs = all_pairs[:split_idx]
    eval_pairs = all_pairs[split_idx:]
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "stage3_dpo.jsonl", 'w', encoding='utf-8') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    with open(output_dir / "stage3_dpo_eval.jsonl", 'w', encoding='utf-8') as f:
        for pair in eval_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    return {
        'train': len(train_pairs),
        'eval': len(eval_pairs),
    }


def prepare_eval_benchmark(
    synthetic_dir: Path,
    nl_pairs_dir: Path,
    output_dir: Path,
    num_samples: int = 200,
) -> int:
    """
    Create evaluation benchmark from held-out samples.
    
    Stratifies by complexity to ensure coverage.
    """
    samples_by_complexity = defaultdict(list)
    
    # Prefer NL pairs if available
    source_dir = nl_pairs_dir if nl_pairs_dir.exists() and any(nl_pairs_dir.glob("*.jsonl")) else synthetic_dir
    
    for jsonl_file in source_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    complexity = sample.get('complexity', 'L3')
                    samples_by_complexity[complexity].append(sample)
    
    # Sample stratified
    benchmark = []
    samples_per_complexity = num_samples // len(samples_by_complexity)
    
    for complexity, samples in samples_by_complexity.items():
        random.shuffle(samples)
        selected = samples[:samples_per_complexity]
        
        for sample in selected:
            benchmark.append({
                'id': f"bench_{sample.get('id', len(benchmark))}",
                'instruction': sample.get('instruction', 'Convert the following natural language description into Signal DSL configuration.'),
                'input': sample.get('input', sample.get('nl', f"Generate a {complexity} configuration.")),
                'output': sample.get('output', sample.get('dsl', '')),
                'complexity': complexity,
                'style': sample.get('style', 'unknown'),
            })
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_benchmark.jsonl", 'w', encoding='utf-8') as f:
        for sample in benchmark:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return len(benchmark)


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for DSL model')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Root data directory containing synthetic/, nl_pairs/, negative/')
    parser.add_argument('--output-dir', type=Path, default=Path('./prepared_data'),
                        help='Output directory for prepared data')
    parser.add_argument('--eval-ratio', type=float, default=0.1,
                        help='Ratio of data to use for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("DSL Training Data Preparation")
    print("=" * 60)
    
    # Define paths
    synthetic_dir = args.data_dir / "synthetic"
    nl_pairs_dir = args.data_dir / "nl_pairs"
    negative_dir = args.data_dir / "negative"
    
    # Prepare Stage 1
    print("\n[Stage 1] Preparing syntax pre-training data...")
    stage1_stats = prepare_stage1_data(
        synthetic_dir=synthetic_dir,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
    )
    print(f"  Train: {stage1_stats['train']:,}")
    print(f"  Eval:  {stage1_stats['eval']:,}")
    
    # Prepare Stage 2
    print("\n[Stage 2] Preparing SFT data...")
    stage2_stats = prepare_stage2_data(
        nl_pairs_dir=nl_pairs_dir,
        synthetic_dir=synthetic_dir,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
    )
    print(f"  Train: {stage2_stats['train']:,}")
    print(f"  Eval:  {stage2_stats['eval']:,}")
    
    # Prepare Stage 3
    print("\n[Stage 3] Preparing DPO data...")
    stage3_stats = prepare_stage3_data(
        synthetic_dir=synthetic_dir,
        negative_dir=negative_dir,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
    )
    print(f"  Train: {stage3_stats['train']:,}")
    print(f"  Eval:  {stage3_stats['eval']:,}")
    
    # Prepare benchmark
    print("\n[Benchmark] Creating evaluation benchmark...")
    bench_count = prepare_eval_benchmark(
        synthetic_dir=synthetic_dir,
        nl_pairs_dir=nl_pairs_dir,
        output_dir=args.output_dir,
    )
    print(f"  Samples: {bench_count}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nFiles created:")
    for f in sorted(args.output_dir.glob("*.jsonl")):
        line_count = sum(1 for _ in open(f))
        print(f"  {f.name}: {line_count:,} samples")


if __name__ == '__main__':
    main()
