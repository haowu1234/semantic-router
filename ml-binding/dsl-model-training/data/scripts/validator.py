#!/usr/bin/env python3
"""
DSL 数据验证管线

对所有生成的 DSL 数据进行验证，确保标签正确：
1. 语法验证 (调用 Go parser)
2. 语义验证 (调用 Go validator)  
3. 编译验证 (调用 Go compiler)

同时构建最终训练数据集。

用法:
    python validator.py --input . --output final/ --parser-bin /path/to/parser
"""

import argparse
import json
import subprocess
import os
import hashlib
from pathlib import Path
from typing import Generator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class ValidationResult:
    """验证结果"""
    syntax_valid: bool
    semantic_valid: bool
    compile_valid: bool
    errors: list[str]
    warnings: list[str]


def validate_dsl_with_go(dsl: str, parser_bin: Path | None) -> ValidationResult:
    """使用 Go 工具验证 DSL
    
    如果 parser_bin 可用，调用它进行验证；
    否则使用简单的正则检查作为 fallback。
    """
    if parser_bin and parser_bin.exists():
        try:
            # 调用 Go parser 进行验证
            result = subprocess.run(
                [str(parser_bin), '--validate', '--json'],
                input=dsl,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = json.loads(result.stdout)
                return ValidationResult(
                    syntax_valid=output.get('syntax_valid', True),
                    semantic_valid=output.get('semantic_valid', True),
                    compile_valid=output.get('compile_valid', True),
                    errors=output.get('errors', []),
                    warnings=output.get('warnings', []),
                )
            else:
                return ValidationResult(
                    syntax_valid=False,
                    semantic_valid=False,
                    compile_valid=False,
                    errors=[result.stderr or 'Parser failed'],
                    warnings=[],
                )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                syntax_valid=False,
                semantic_valid=False,
                compile_valid=False,
                errors=['Validation timeout'],
                warnings=[],
            )
        except json.JSONDecodeError:
            # Parser 可能不支持 JSON 输出，检查返回码
            return ValidationResult(
                syntax_valid=result.returncode == 0,
                semantic_valid=False,
                compile_valid=False,
                errors=[],
                warnings=[],
            )
        except Exception as e:
            return ValidationResult(
                syntax_valid=False,
                semantic_valid=False,
                compile_valid=False,
                errors=[str(e)],
                warnings=[],
            )
    
    # Fallback: 简单的语法检查
    return validate_dsl_simple(dsl)


def validate_dsl_simple(dsl: str) -> ValidationResult:
    """简单的 DSL 语法验证 (无 Go 工具时的 fallback)"""
    import re
    
    errors = []
    warnings = []
    
    # 检查括号匹配
    brace_count = dsl.count('{') - dsl.count('}')
    if brace_count != 0:
        errors.append(f'Brace mismatch: {brace_count} unclosed')
    
    # 检查引号匹配 (简化检查)
    quote_count = dsl.count('"')
    if quote_count % 2 != 0:
        errors.append('Unclosed string literal')
    
    # 检查关键字拼写
    keywords = ['SIGNAL', 'ROUTE', 'PLUGIN', 'BACKEND', 'GLOBAL', 'PRIORITY', 'WHEN', 'MODEL', 'ALGORITHM']
    for kw in keywords:
        # 检查常见拼写错误
        wrong_patterns = [kw + 'S', kw[:-1], kw[:3] + kw[4:]]
        for wrong in wrong_patterns:
            if re.search(rf'\b{wrong}\b', dsl, re.IGNORECASE) and wrong != kw:
                errors.append(f'Possible typo: {wrong} (should be {kw}?)')
    
    # 检查信号引用
    defined_signals = set()
    for match in re.finditer(r'SIGNAL\s+(\w+)\s+(\w+)', dsl):
        sig_type, sig_name = match.groups()
        defined_signals.add((sig_type, sig_name))
    
    for match in re.finditer(r'WHEN\s+.*?(\w+)\("(\w+)"\)', dsl):
        sig_type, sig_name = match.groups()
        if (sig_type, sig_name) not in defined_signals and (sig_type, sig_name.split(':')[0]) not in defined_signals:
            warnings.append(f'Reference to undefined signal: {sig_type}("{sig_name}")')
    
    syntax_valid = len(errors) == 0
    semantic_valid = len([w for w in warnings if 'undefined' in w.lower()]) == 0
    
    return ValidationResult(
        syntax_valid=syntax_valid,
        semantic_valid=semantic_valid,
        compile_valid=syntax_valid and semantic_valid,
        errors=errors,
        warnings=warnings,
    )


def load_samples(input_path: Path) -> Generator[dict, None, None]:
    """递归加载所有 JSONL 文件中的样本"""
    if input_path.is_file():
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    elif input_path.is_dir():
        for jsonl_file in sorted(input_path.rglob('*.jsonl')):
            # 跳过 final/ 目录
            if 'final' in str(jsonl_file):
                continue
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue


def validate_sample(sample: dict, parser_bin: Path | None) -> tuple[dict, ValidationResult]:
    """验证单个样本"""
    dsl = sample.get('dsl', '')
    if not dsl:
        return sample, ValidationResult(
            syntax_valid=False,
            semantic_valid=False,
            compile_valid=False,
            errors=['Empty DSL'],
            warnings=[],
        )
    
    result = validate_dsl_with_go(dsl, parser_bin)
    return sample, result


def build_stage1_data(samples: list[dict]) -> list[dict]:
    """构建 Stage 1 (语法预训练) 数据
    
    纯 DSL 文本，用于 Causal LM 训练。
    """
    return [
        {
            'id': s.get('id', ''),
            'dsl': s.get('dsl', ''),
            'complexity': s.get('complexity', s.get('metadata', {}).get('complexity', 'unknown')),
        }
        for s in samples
        if s.get('dsl') and s.get('valid', True)
    ]


def build_stage2_data(samples: list[dict]) -> list[dict]:
    """构建 Stage 2 (SFT) 数据
    
    NL→DSL 配对数据，Chat 格式。
    """
    sft_data = []
    for s in samples:
        if not s.get('input') or not s.get('output'):
            continue
        
        sft_data.append({
            'id': s.get('id', ''),
            'instruction': s.get('instruction', 'Convert the following natural language description into Signal DSL configuration.'),
            'input': s['input'],
            'output': s['output'],
            'style': s.get('style', 'unknown'),
            'complexity': s.get('complexity', 'unknown'),
        })
    
    return sft_data


def build_stage3_data(positive_samples: list[dict], negative_samples: list[dict]) -> list[dict]:
    """构建 Stage 3 (DPO) 数据
    
    (prompt, chosen, rejected) 三元组。
    """
    dpo_data = []
    
    # 为每个负样本找到对应的正样本
    positive_by_id = {s.get('id', ''): s for s in positive_samples}
    
    for neg in negative_samples:
        orig_id = neg.get('original_id', '')
        orig_dsl = neg.get('original_dsl', '')
        
        if not orig_dsl:
            # 尝试从 positive_samples 找
            pos = positive_by_id.get(orig_id)
            if pos:
                orig_dsl = pos.get('dsl', '')
        
        if not orig_dsl or not neg.get('dsl'):
            continue
        
        # 构建简单的 prompt
        prompt = f"Generate a valid Signal DSL configuration."
        
        dpo_data.append({
            'id': neg.get('id', ''),
            'prompt': prompt,
            'chosen': orig_dsl,
            'rejected': neg.get('dsl', ''),
            'mutation_type': neg.get('mutation_type', ''),
            'mutation_category': neg.get('mutation_category', ''),
        })
    
    return dpo_data


def build_eval_benchmark(samples: list[dict], size: int = 200) -> list[dict]:
    """构建评估基准集"""
    import random
    
    # 按复杂度分层抽样
    by_complexity = {}
    for s in samples:
        c = s.get('complexity', 'unknown')
        by_complexity.setdefault(c, []).append(s)
    
    # 分配比例
    allocation = {'L1': 0.05, 'L2': 0.20, 'L3': 0.40, 'L4': 0.25, 'L5': 0.10}
    
    benchmark = []
    for comp, ratio in allocation.items():
        pool = by_complexity.get(comp, [])
        n = min(int(size * ratio), len(pool))
        if n > 0:
            benchmark.extend(random.sample(pool, n))
    
    return benchmark


def main():
    parser = argparse.ArgumentParser(description='Validate DSL data and build training datasets')
    parser.add_argument('--input', type=Path, required=True,
                        help='Input directory containing all data subdirectories')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output directory for final training data')
    parser.add_argument('--parser-bin', type=Path, default=None,
                        help='Path to Go DSL parser binary')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel validation workers')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip validation, trust existing labels')
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("Loading samples...")
    all_samples = list(load_samples(args.input))
    print(f"Loaded {len(all_samples)} samples")
    
    # 分类样本
    dsl_samples = [s for s in all_samples if s.get('dsl') and not s.get('input')]
    nl_pair_samples = [s for s in all_samples if s.get('input') and s.get('output')]
    negative_samples = [s for s in all_samples if s.get('valid') == False]
    
    print(f"  DSL-only samples: {len(dsl_samples)}")
    print(f"  NL-DSL pairs: {len(nl_pair_samples)}")
    print(f"  Negative samples: {len(negative_samples)}")
    
    # 验证 (如果启用)
    valid_dsl_samples = []
    if not args.skip_validation and dsl_samples:
        print("\nValidating DSL samples...")
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(validate_sample, s, args.parser_bin)
                for s in dsl_samples
            ]
            
            validated = 0
            errors = 0
            for future in as_completed(futures):
                sample, result = future.result()
                
                # 更新验证状态
                if result.syntax_valid:
                    sample['valid'] = True
                    sample['validation'] = {
                        'syntax': True,
                        'semantic': result.semantic_valid,
                        'compile': result.compile_valid,
                        'warnings': result.warnings,
                    }
                    valid_dsl_samples.append(sample)
                else:
                    sample['valid'] = False
                    sample['validation_errors'] = result.errors
                    errors += 1
                
                validated += 1
                if validated % 500 == 0:
                    print(f"  Validated {validated}/{len(dsl_samples)} samples...")
        
        print(f"  Valid: {len(valid_dsl_samples)}, Invalid: {errors}")
    else:
        valid_dsl_samples = [s for s in dsl_samples if s.get('valid', True)]
        print(f"  Skipping validation, using {len(valid_dsl_samples)} samples marked as valid")
    
    # 构建 Stage 1 数据
    print("\nBuilding Stage 1 (Syntax Pretraining) data...")
    stage1_data = build_stage1_data(valid_dsl_samples)
    with open(args.output / 'stage1_syntax_pt.jsonl', 'w', encoding='utf-8') as f:
        for item in stage1_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Stage 1: {len(stage1_data)} samples")
    
    # 构建 Stage 2 数据
    print("\nBuilding Stage 2 (SFT) data...")
    stage2_data = build_stage2_data(nl_pair_samples)
    with open(args.output / 'stage2_sft.jsonl', 'w', encoding='utf-8') as f:
        for item in stage2_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Stage 2: {len(stage2_data)} samples")
    
    # 构建 Stage 3 数据
    print("\nBuilding Stage 3 (DPO) data...")
    stage3_data = build_stage3_data(valid_dsl_samples, negative_samples)
    with open(args.output / 'stage3_dpo.jsonl', 'w', encoding='utf-8') as f:
        for item in stage3_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Stage 3: {len(stage3_data)} samples")
    
    # 构建评估基准
    print("\nBuilding evaluation benchmark...")
    benchmark_data = build_eval_benchmark(nl_pair_samples, size=200)
    with open(args.output / 'eval_benchmark.jsonl', 'w', encoding='utf-8') as f:
        for item in benchmark_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Benchmark: {len(benchmark_data)} samples")
    
    # 总结统计
    stats = {
        'total_input_samples': len(all_samples),
        'dsl_only': len(dsl_samples),
        'nl_pairs': len(nl_pair_samples),
        'negative': len(negative_samples),
        'valid_dsl': len(valid_dsl_samples),
        'stage1_size': len(stage1_data),
        'stage2_size': len(stage2_data),
        'stage3_size': len(stage3_data),
        'benchmark_size': len(benchmark_data),
    }
    
    with open(args.output / 'dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n=== Final Dataset Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nOutput saved to: {args.output}")


if __name__ == '__main__':
    main()
