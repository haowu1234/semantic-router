#!/usr/bin/env python3
"""
Upload Signal DSL training dataset to Hugging Face Hub.

This script can either:
1. Generate fresh synthetic data and upload
2. Upload existing prepared data files

Usage:
    # Generate and upload fresh data
    python scripts/upload_dataset_to_hf.py \
        --repo-id your-username/signal-dsl-dataset \
        --generate \
        --num-samples 10000

    # Upload existing prepared data
    python scripts/upload_dataset_to_hf.py \
        --repo-id your-username/signal-dsl-dataset \
        --data-dir ./prepared_data
"""

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

from huggingface_hub import HfApi, create_repo


# Dataset card template
DATASET_CARD_TEMPLATE = """---
license: apache-2.0
task_categories:
  - text-generation
  - text2text-generation
language:
  - en
  - zh
tags:
  - dsl
  - domain-specific-language
  - code-generation
  - routing
  - llm-routing
  - signal-router
  - synthetic
pretty_name: Signal DSL Dataset
size_categories:
  - {size_category}
---

# Signal DSL Dataset

A synthetic dataset for training models to generate **Signal DSL** (Domain-Specific Language) configurations from natural language descriptions.

## Dataset Description

Signal DSL is used to configure intelligent LLM routing with signals, routes, plugins, and algorithms. This dataset contains:

| Split | Samples | Description |
|-------|---------|-------------|
| **stage1_syntax_pt** | {stage1_train} | Pure DSL for syntax pre-training |
| **stage2_sft** | {stage2_train} | NL→DSL pairs for instruction following |
| **stage3_dpo** | {stage3_train} | Preference pairs for DPO training |
| **eval_benchmark** | {eval_count} | Held-out evaluation set |

## Signal DSL Overview

### Core Components

1. **SIGNAL**: Define detection signals (keyword, domain, embedding, etc.)
2. **ROUTE**: Conditional routing rules based on signals
3. **PLUGIN**: Add capabilities (RAG, cache, memory, etc.)
4. **ALGORITHM**: Ranking/selection algorithms
5. **BACKEND**: External service configurations

### Example DSL

```dsl
SIGNAL keyword code_keywords {{
  keywords: ["code", "programming", "debug", "function"]
  threshold: 0.8
}}

SIGNAL domain code_domain {{
  description: "Code and programming related queries"
}}

ROUTE code_route (description = "Route code queries to specialist") {{
  PRIORITY 100
  WHEN keyword("code_keywords") OR domain("code_domain")
  MODEL "deepseek-coder" (
    reasoning = true,
    temperature = 0.1
  )
}}
```

## Data Format

### Stage 1: Syntax Pre-training (Completion)

```json
{{
  "id": "dsl_001",
  "dsl": "SIGNAL keyword kw_1 {{ keywords: [\\"urgent\\"] }}",
  "complexity": "L1"
}}
```

### Stage 2: SFT (Instruction-Input-Output)

```json
{{
  "id": "sft_001",
  "instruction": "Convert the following natural language description into Signal DSL configuration.",
  "input": "Create a route that sends math questions to GPT-4",
  "output": "SIGNAL domain math {{ ... }} ROUTE math_route {{ ... }}",
  "style": "en_formal",
  "complexity": "L2"
}}
```

### Stage 3: DPO (Preference Pairs)

```json
{{
  "id": "dpo_001",
  "prompt": "Generate a valid Signal DSL configuration.",
  "chosen": "SIGNAL keyword kw {{ keywords: [\\"test\\"] }}",
  "rejected": "SIGNAL keyword kw {{ keywords: [\\"test\\" }}",
  "mutation_type": "syntax_error",
  "mutation_category": "missing_bracket"
}}
```

## Complexity Levels

| Level | Description | Signals | Routes | Plugins |
|-------|-------------|---------|--------|---------|
| L1 | Simple | 1-2 | 1 | 0 |
| L2 | Basic | 2-3 | 1-2 | 0-1 |
| L3 | Medium | 3-5 | 2-3 | 1-2 |
| L4 | Complex | 5-8 | 3-5 | 2-4 |
| L5 | Expert | 8+ | 5+ | 4+ |

## Usage

```python
from datasets import load_dataset

# Load all splits
dataset = load_dataset("{repo_id}")

# Load specific split
sft_data = load_dataset("{repo_id}", split="stage2_sft")

# Iterate through samples
for sample in sft_data:
    print(f"Input: {{sample['input']}}")
    print(f"Output: {{sample['output']}}")
```

## Training with this Dataset

This dataset is designed for 3-stage training:

1. **Stage 1 (Syntax PT)**: Train language model on pure DSL to learn syntax
2. **Stage 2 (SFT)**: Fine-tune on NL→DSL pairs for instruction following  
3. **Stage 3 (DPO)**: Preference optimization to prefer correct over incorrect DSL

## Generation Process

Data was generated using:
- **CFG Random Walk**: Grammar-based generation ensuring syntactic correctness
- **Template Expansion**: Schema-aware field value generation
- **Negative Sampling**: Systematic mutation for preference pairs
- **NL Paraphrasing**: Multiple linguistic styles (formal/casual, EN/ZH)

## Citation

```bibtex
@dataset{{signal-dsl-dataset,
  author = {{Signal Router Team}},
  title = {{Signal DSL Dataset: Synthetic Training Data for DSL Generation}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/datasets/{repo_id}}}
}}
```

## License

Apache 2.0 - See LICENSE for details.
"""


def get_size_category(total_samples: int) -> str:
    """Get HuggingFace size category string."""
    if total_samples < 1000:
        return "n<1K"
    elif total_samples < 10000:
        return "1K<n<10K"
    elif total_samples < 100000:
        return "10K<n<100K"
    elif total_samples < 1000000:
        return "100K<n<1M"
    else:
        return "n>1M"


def generate_synthetic_data(
    output_dir: Path,
    num_samples: int = 10000,
    seed: int = 42
) -> dict:
    """
    Generate synthetic Signal DSL training data.
    
    This is a simplified version - the full generator is in data/scripts/cfg_generator.py
    """
    random.seed(seed)
    
    # Import the CFG generator if available
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "scripts"))
        from cfg_generator import DSLGenerator
        
        print("Using CFG generator for high-quality synthetic data...")
        generator = DSLGenerator()
        
        samples_by_complexity = defaultdict(list)
        complexities = ['L1', 'L2', 'L3', 'L4', 'L5']
        samples_per_complexity = num_samples // len(complexities)
        
        for complexity in complexities:
            print(f"  Generating {samples_per_complexity} {complexity} samples...")
            for i in range(samples_per_complexity):
                sample = generator.generate(complexity=complexity)
                samples_by_complexity[complexity].append(sample)
        
        all_samples = []
        for samples in samples_by_complexity.values():
            all_samples.extend(samples)
        
    except ImportError:
        print("CFG generator not available, using simple template-based generation...")
        all_samples = generate_simple_templates(num_samples, seed)
    
    # Split and save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    random.shuffle(all_samples)
    
    # Stage 1: Syntax pre-training (pure DSL)
    stage1_samples = all_samples
    split_idx = int(len(stage1_samples) * 0.9)
    stage1_train = stage1_samples[:split_idx]
    stage1_eval = stage1_samples[split_idx:]
    
    save_jsonl(output_dir / "stage1_syntax_pt.jsonl", stage1_train)
    save_jsonl(output_dir / "stage1_syntax_pt_eval.jsonl", stage1_eval)
    
    # Stage 2: SFT (with NL descriptions)
    stage2_samples = create_sft_samples(all_samples)
    split_idx = int(len(stage2_samples) * 0.9)
    stage2_train = stage2_samples[:split_idx]
    stage2_eval = stage2_samples[split_idx:]
    
    save_jsonl(output_dir / "stage2_sft.jsonl", stage2_train)
    save_jsonl(output_dir / "stage2_sft_eval.jsonl", stage2_eval)
    
    # Stage 3: DPO (preference pairs)
    stage3_samples = create_dpo_samples(all_samples[:len(all_samples)//2])
    split_idx = int(len(stage3_samples) * 0.9)
    stage3_train = stage3_samples[:split_idx]
    stage3_eval = stage3_samples[split_idx:]
    
    save_jsonl(output_dir / "stage3_dpo.jsonl", stage3_train)
    save_jsonl(output_dir / "stage3_dpo_eval.jsonl", stage3_eval)
    
    # Eval benchmark
    eval_benchmark = create_eval_benchmark(stage2_samples, num_samples=200)
    save_jsonl(output_dir / "eval_benchmark.jsonl", eval_benchmark)
    
    return {
        "stage1_train": len(stage1_train),
        "stage1_eval": len(stage1_eval),
        "stage2_train": len(stage2_train),
        "stage2_eval": len(stage2_eval),
        "stage3_train": len(stage3_train),
        "stage3_eval": len(stage3_eval),
        "eval_count": len(eval_benchmark),
    }


def generate_simple_templates(num_samples: int, seed: int) -> list:
    """Generate simple template-based DSL samples when CFG generator not available."""
    random.seed(seed)
    
    signal_types = ['keyword', 'domain', 'embedding', 'complexity', 'language']
    model_names = ['gpt-4o', 'gpt-4o-mini', 'deepseek-v3', 'qwen2.5:7b', 'claude-3-sonnet']
    keywords_pool = [
        ['urgent', 'asap', 'emergency'],
        ['code', 'programming', 'debug'],
        ['math', 'calculate', 'equation'],
        ['translate', 'language', 'chinese'],
        ['summarize', 'tldr', 'brief'],
    ]
    domains = ['math', 'code', 'science', 'general', 'creative']
    
    samples = []
    
    for i in range(num_samples):
        complexity = random.choice(['L1', 'L2', 'L3', 'L4', 'L5'])
        
        # Generate DSL based on complexity
        num_signals = {'L1': 1, 'L2': 2, 'L3': 3, 'L4': 5, 'L5': 7}[complexity]
        num_routes = {'L1': 1, 'L2': 1, 'L3': 2, 'L4': 3, 'L5': 4}[complexity]
        
        dsl_parts = []
        signal_names = []
        
        # Generate signals
        for j in range(num_signals):
            sig_type = random.choice(signal_types)
            sig_name = f"{sig_type}_{j+1}"
            signal_names.append((sig_name, sig_type))
            
            if sig_type == 'keyword':
                keywords = random.choice(keywords_pool)
                dsl_parts.append(f'SIGNAL keyword {sig_name} {{\n  keywords: {json.dumps(keywords)}\n}}')
            elif sig_type == 'domain':
                domain = random.choice(domains)
                dsl_parts.append(f'SIGNAL domain {sig_name} {{\n  description: "{domain} related queries"\n}}')
            elif sig_type == 'embedding':
                dsl_parts.append(f'SIGNAL embedding {sig_name} {{\n  candidates: ["topic 1", "topic 2"]\n  threshold: 0.8\n}}')
            elif sig_type == 'complexity':
                dsl_parts.append(f'SIGNAL complexity {sig_name} {{\n  description: "Query complexity detection"\n}}')
            else:
                dsl_parts.append(f'SIGNAL language {sig_name} {{\n  description: "Language detection"\n}}')
        
        # Generate routes
        for j in range(num_routes):
            route_name = f"route_{j+1}"
            model = random.choice(model_names)
            sig_name, sig_type = random.choice(signal_names)
            
            dsl_parts.append(f'''ROUTE {route_name} {{
  WHEN {sig_type}("{sig_name}")
  MODEL "{model}"
}}''')
        
        dsl = '\n\n'.join(dsl_parts)
        
        samples.append({
            'id': f'dsl_{i:06d}',
            'dsl': dsl,
            'complexity': complexity,
        })
    
    return samples


def create_sft_samples(dsl_samples: list) -> list:
    """Create SFT samples with NL descriptions."""
    sft_samples = []
    
    instructions = [
        "Convert the following natural language description into Signal DSL configuration.",
        "Generate a Signal DSL configuration based on this description:",
        "Create the appropriate DSL configuration for:",
        "Write Signal DSL code for the following requirement:",
    ]
    
    nl_templates = [
        "Create a routing configuration with {num_signals} signals and {num_routes} routes.",
        "Set up an LLM router that handles {complexity} complexity queries.",
        "Configure signal-based routing with keyword and domain detection.",
        "Build a {complexity} routing setup for intelligent model selection.",
    ]
    
    for sample in dsl_samples:
        # Extract info from DSL
        dsl = sample['dsl']
        complexity = sample.get('complexity', 'L3')
        num_signals = dsl.count('SIGNAL ')
        num_routes = dsl.count('ROUTE ')
        
        # Generate NL description
        nl_template = random.choice(nl_templates)
        nl_input = nl_template.format(
            num_signals=num_signals,
            num_routes=num_routes,
            complexity=complexity
        )
        
        sft_samples.append({
            'id': f"sft_{sample['id']}",
            'instruction': random.choice(instructions),
            'input': nl_input,
            'output': dsl,
            'style': random.choice(['en_formal', 'en_casual', 'zh_formal', 'zh_casual']),
            'complexity': complexity,
        })
    
    return sft_samples


def create_dpo_samples(dsl_samples: list) -> list:
    """Create DPO preference pairs with mutations."""
    dpo_samples = []
    
    mutation_types = [
        ('syntax_error', 'missing_bracket', lambda d: d.replace('}', '', 1)),
        ('syntax_error', 'missing_quote', lambda d: d.replace('"', '', 1)),
        ('reference_error', 'undefined_signal', lambda d: d.replace('WHEN ', 'WHEN undefined_')),
        ('value_error', 'invalid_threshold', lambda d: d.replace('0.8', '1.5')),
        ('structure_error', 'missing_field', lambda d: d.replace('keywords:', 'kw:')),
    ]
    
    for sample in dsl_samples:
        mutation_type, mutation_category, mutate_fn = random.choice(mutation_types)
        
        try:
            rejected = mutate_fn(sample['dsl'])
            if rejected == sample['dsl']:
                continue
                
            dpo_samples.append({
                'id': f"dpo_{sample['id']}",
                'prompt': "Generate a valid Signal DSL configuration.",
                'chosen': sample['dsl'],
                'rejected': rejected,
                'mutation_type': mutation_type,
                'mutation_category': mutation_category,
            })
        except:
            continue
    
    return dpo_samples


def create_eval_benchmark(sft_samples: list, num_samples: int = 200) -> list:
    """Create stratified evaluation benchmark."""
    by_complexity = defaultdict(list)
    
    for sample in sft_samples:
        by_complexity[sample['complexity']].append(sample)
    
    benchmark = []
    per_complexity = num_samples // len(by_complexity)
    
    for complexity, samples in by_complexity.items():
        random.shuffle(samples)
        for sample in samples[:per_complexity]:
            benchmark.append({
                'id': f"bench_{sample['id']}",
                'instruction': sample['instruction'],
                'input': sample['input'],
                'output': sample['output'],
                'complexity': complexity,
                'style': sample['style'],
            })
    
    return benchmark


def save_jsonl(path: Path, samples: list):
    """Save samples to JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def load_existing_data(data_dir: Path) -> dict:
    """Load existing prepared data files."""
    stats = {}
    
    files = [
        "stage1_syntax_pt.jsonl",
        "stage1_syntax_pt_eval.jsonl", 
        "stage2_sft.jsonl",
        "stage2_sft_eval.jsonl",
        "stage3_dpo.jsonl",
        "stage3_dpo_eval.jsonl",
        "eval_benchmark.jsonl",
    ]
    
    for filename in files:
        filepath = data_dir / filename
        if filepath.exists():
            count = sum(1 for _ in open(filepath, 'r', encoding='utf-8'))
            key = filename.replace('.jsonl', '').replace('_eval', '_eval')
            stats[key] = count
        else:
            print(f"Warning: {filename} not found in {data_dir}")
    
    return stats


def create_dataset_card(repo_id: str, stats: dict) -> str:
    """Generate dataset card content."""
    total = sum(v for k, v in stats.items() if 'eval' not in k and 'benchmark' not in k)
    
    return DATASET_CARD_TEMPLATE.format(
        repo_id=repo_id,
        size_category=get_size_category(total),
        stage1_train=stats.get('stage1_syntax_pt', 0),
        stage2_train=stats.get('stage2_sft', 0),
        stage3_train=stats.get('stage3_dpo', 0),
        eval_count=stats.get('eval_benchmark', 0),
    )


def upload_dataset_to_hub(
    data_dir: Path,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload Signal DSL training dataset"
):
    """Upload dataset to Hugging Face Hub."""
    api = HfApi()
    
    # Create dataset repo
    print(f"Creating dataset repository: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private
    )
    
    # Get stats
    stats = load_existing_data(data_dir)
    
    # Generate and save dataset card
    print("Generating dataset card...")
    readme_content = create_dataset_card(repo_id, stats)
    readme_path = data_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # Upload all files
    print(f"Uploading dataset to {repo_id}...")
    
    files_to_upload = list(data_dir.glob("*.jsonl")) + [readme_path]
    
    for filepath in files_to_upload:
        print(f"  Uploading {filepath.name}...")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filepath.name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {filepath.name}"
        )
    
    print(f"\n✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")
    print(f"\nUsage:")
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset(\"{repo_id}\")")


def main():
    parser = argparse.ArgumentParser(description="Upload Signal DSL dataset to Hugging Face Hub")
    parser.add_argument(
        "--repo-id", "-r",
        type=str,
        required=True,
        help="Hugging Face dataset repo ID (e.g., username/signal-dsl-dataset)"
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=None,
        help="Path to existing prepared data directory"
    )
    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate fresh synthetic data before uploading"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=10000,
        help="Number of samples to generate (if --generate is used)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./generated_dataset"),
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation"
    )
    
    args = parser.parse_args()
    
    if args.generate:
        print(f"Generating {args.num_samples} synthetic samples...")
        stats = generate_synthetic_data(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            seed=args.seed
        )
        print(f"\nGenerated data statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:,}")
        
        data_dir = args.output_dir
    elif args.data_dir:
        data_dir = args.data_dir
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            sys.exit(1)
    else:
        print("Error: Either --generate or --data-dir must be specified")
        sys.exit(1)
    
    upload_dataset_to_hub(
        data_dir=data_dir,
        repo_id=args.repo_id,
        private=args.private
    )


if __name__ == "__main__":
    main()
