#!/usr/bin/env python3
"""
负样本生成器

通过变异有效 DSL 生成各种类型的错误样本，用于 DPO (Direct Preference Optimization) 训练。

变异类型:
1. syntax_error: 语法错误 (删除括号、拼错关键字)
2. reference_error: 引用错误 (WHEN 引用未定义信号)
3. constraint_violation: 约束违反 (threshold: 1.5)
4. schema_mismatch: Schema 不匹配 (字段用错类型)

用法:
    python negative_sampler.py --input synthetic/ --output negative/
    python negative_sampler.py --input synthetic/L3_standard.jsonl --output negative/L3_negatives.jsonl
"""

import argparse
import json
import random
import re
import hashlib
from pathlib import Path
from typing import Generator, Callable
from dataclasses import dataclass


# 变异类型定义
@dataclass
class MutationType:
    name: str
    description: str
    mutator: Callable[[str], str]


def mutate_syntax_missing_brace(dsl: str) -> str:
    """删除一个闭合括号"""
    # 找到所有 } 的位置
    brace_positions = [i for i, c in enumerate(dsl) if c == '}']
    if len(brace_positions) > 1:
        # 删除一个非最后的 }
        pos = random.choice(brace_positions[:-1])
        return dsl[:pos] + dsl[pos+1:]
    return dsl


def mutate_syntax_typo_keyword(dsl: str) -> str:
    """拼错关键字"""
    typos = {
        'SIGNAL': ['SIGNALS', 'SINGAL', 'SGNL', 'SINGNAL'],
        'ROUTE': ['ROUTES', 'ROUT', 'ROUTER', 'ROUE'],
        'PLUGIN': ['PLUGINS', 'PLUGN', 'PLGUIN', 'PULGIN'],
        'BACKEND': ['BACKENDS', 'BACKND', 'BCKEND', 'BAKEND'],
        'GLOBAL': ['GLOBALS', 'GLOBL', 'GLOBA', 'GLOABL'],
        'PRIORITY': ['PRIORITIES', 'PRIRITY', 'PRORITY', 'PIROITY'],
        'WHEN': ['WHNE', 'WHEM', 'WHN', 'WEHN'],
        'MODEL': ['MODLE', 'MODL', 'MOEDL', 'MODELL'],
        'ALGORITHM': ['ALGORITH', 'ALGORTHM', 'ALGORITM', 'ALOGRITHM'],
    }
    
    for keyword, wrong_forms in typos.items():
        pattern = rf'\b{keyword}\b'
        if re.search(pattern, dsl):
            wrong = random.choice(wrong_forms)
            return re.sub(pattern, wrong, dsl, count=1)
    return dsl


def mutate_syntax_missing_colon(dsl: str) -> str:
    """删除字段后的冒号"""
    # 匹配 field: value 模式
    pattern = r'(\s+\w+):\s*'
    matches = list(re.finditer(pattern, dsl))
    if matches:
        match = random.choice(matches)
        # 删除冒号
        return dsl[:match.start()+len(match.group(1))] + ' ' + dsl[match.end():]
    return dsl


def mutate_syntax_unclosed_string(dsl: str) -> str:
    """创建未闭合的字符串"""
    # 找到所有字符串
    pattern = r'"[^"]*"'
    matches = list(re.finditer(pattern, dsl))
    if matches:
        match = random.choice(matches)
        # 删除结束引号
        return dsl[:match.end()-1] + dsl[match.end():]
    return dsl


def mutate_reference_undefined_signal(dsl: str) -> str:
    """在 WHEN 中引用未定义的信号"""
    # 找到 WHEN 子句
    when_pattern = r'WHEN\s+(\w+)\("([^"]+)"\)'
    match = re.search(when_pattern, dsl)
    if match:
        sig_type = match.group(1)
        # 生成一个不存在的信号名
        fake_name = f'undefined_{random.randint(100, 999)}'
        return dsl[:match.start()] + f'WHEN {sig_type}("{fake_name}")' + dsl[match.end():]
    return dsl


def mutate_reference_wrong_signal_type(dsl: str) -> str:
    """使用错误的信号类型"""
    signal_types = ['keyword', 'embedding', 'domain', 'jailbreak', 'pii', 'context']
    when_pattern = r'WHEN\s+(\w+)\("([^"]+)"\)'
    match = re.search(when_pattern, dsl)
    if match:
        old_type = match.group(1)
        sig_name = match.group(2)
        # 选择一个不同的类型
        new_types = [t for t in signal_types if t != old_type]
        if new_types:
            new_type = random.choice(new_types)
            return dsl[:match.start()] + f'WHEN {new_type}("{sig_name}")' + dsl[match.end():]
    return dsl


def mutate_reference_undefined_plugin(dsl: str) -> str:
    """引用未定义的插件"""
    # 找到路由中的 PLUGIN 引用
    plugin_pattern = r'(\s+PLUGIN\s+)(\w+)(\s*[^{])'
    matches = list(re.finditer(plugin_pattern, dsl))
    if matches:
        match = random.choice(matches)
        fake_plugin = f'nonexistent_plugin_{random.randint(100, 999)}'
        return dsl[:match.start()] + f'{match.group(1)}{fake_plugin}{match.group(3)}' + dsl[match.end():]
    return dsl


def mutate_constraint_threshold_overflow(dsl: str) -> str:
    """将 threshold 设置为超出范围的值"""
    threshold_pattern = r'(threshold:\s*)([\d.]+)'
    match = re.search(threshold_pattern, dsl)
    if match:
        # 设置为 > 1.0 或 < 0.0
        bad_value = random.choice(['1.5', '2.0', '-0.5', '100'])
        return dsl[:match.start()] + f'{match.group(1)}{bad_value}' + dsl[match.end():]
    return dsl


def mutate_constraint_priority_negative(dsl: str) -> str:
    """将 PRIORITY 设置为负数"""
    priority_pattern = r'(PRIORITY\s+)(\d+)'
    match = re.search(priority_pattern, dsl)
    if match:
        bad_value = random.choice(['-1', '-100', '-999'])
        return dsl[:match.start()] + f'{match.group(1)}{bad_value}' + dsl[match.end():]
    return dsl


def mutate_constraint_port_overflow(dsl: str) -> str:
    """将 port 设置为超出范围的值"""
    port_pattern = r'(port:\s*)(\d+)'
    match = re.search(port_pattern, dsl)
    if match:
        bad_value = random.choice(['0', '70000', '99999', '-1'])
        return dsl[:match.start()] + f'{match.group(1)}{bad_value}' + dsl[match.end():]
    return dsl


def mutate_schema_wrong_field_type(dsl: str) -> str:
    """将字段值类型弄错"""
    # 找到布尔字段，换成字符串
    bool_pattern = r'(enabled:\s*)(true|false)'
    match = re.search(bool_pattern, dsl)
    if match:
        return dsl[:match.start()] + f'{match.group(1)}"yes"' + dsl[match.end():]
    
    # 找到数字字段，换成字符串
    num_pattern = r'(threshold:\s*)([\d.]+)'
    match = re.search(num_pattern, dsl)
    if match:
        return dsl[:match.start()] + f'{match.group(1)}"high"' + dsl[match.end():]
    
    return dsl


def mutate_schema_wrong_field_for_type(dsl: str) -> str:
    """给信号类型添加不属于它的字段"""
    # 找到 SIGNAL 声明
    signal_pattern = r'(SIGNAL\s+(\w+)\s+\w+\s*\{[^}]*)(})'
    match = re.search(signal_pattern, dsl, re.DOTALL)
    if match:
        sig_type = match.group(2)
        # 添加一个不属于该类型的字段
        wrong_fields = {
            'keyword': '\n  mmlu_categories: ["math"]',  # 属于 domain
            'embedding': '\n  keywords: ["test"]',  # 属于 keyword
            'domain': '\n  candidates: ["test"]',  # 属于 embedding
            'jailbreak': '\n  retrieval_limit: 5',  # 属于 memory plugin
            'pii': '\n  operator: "any"',  # 属于 keyword
        }
        wrong_field = wrong_fields.get(sig_type, '\n  unknown_field: true')
        return dsl[:match.end(1)] + wrong_field + match.group(3) + dsl[match.end():]
    return dsl


def mutate_schema_invalid_enum(dsl: str) -> str:
    """使用无效的枚举值"""
    # 找到 method 字段
    method_pattern = r'(method:\s*)"([^"]+)"'
    match = re.search(method_pattern, dsl)
    if match:
        return dsl[:match.start()] + f'{match.group(1)}"invalid_method"' + dsl[match.end():]
    
    # 找到 strategy 字段
    strategy_pattern = r'(strategy:\s*)"([^"]+)"'
    match = re.search(strategy_pattern, dsl)
    if match:
        return dsl[:match.start()] + f'{match.group(1)}"unknown_strategy"' + dsl[match.end():]
    
    return dsl


# 变异类型注册
MUTATION_TYPES: dict[str, list[MutationType]] = {
    'syntax_error': [
        MutationType('missing_brace', 'Missing closing brace', mutate_syntax_missing_brace),
        MutationType('typo_keyword', 'Typo in keyword', mutate_syntax_typo_keyword),
        MutationType('missing_colon', 'Missing colon after field', mutate_syntax_missing_colon),
        MutationType('unclosed_string', 'Unclosed string literal', mutate_syntax_unclosed_string),
    ],
    'reference_error': [
        MutationType('undefined_signal', 'Reference to undefined signal', mutate_reference_undefined_signal),
        MutationType('wrong_signal_type', 'Wrong signal type in reference', mutate_reference_wrong_signal_type),
        MutationType('undefined_plugin', 'Reference to undefined plugin', mutate_reference_undefined_plugin),
    ],
    'constraint_violation': [
        MutationType('threshold_overflow', 'Threshold value out of range', mutate_constraint_threshold_overflow),
        MutationType('priority_negative', 'Negative priority value', mutate_constraint_priority_negative),
        MutationType('port_overflow', 'Port number out of range', mutate_constraint_port_overflow),
    ],
    'schema_mismatch': [
        MutationType('wrong_field_type', 'Wrong field value type', mutate_schema_wrong_field_type),
        MutationType('wrong_field_for_type', 'Field not valid for signal type', mutate_schema_wrong_field_for_type),
        MutationType('invalid_enum', 'Invalid enum value', mutate_schema_invalid_enum),
    ],
}


def generate_negative_sample(sample: dict, mutation_category: str) -> dict | None:
    """为一个样本生成负样本"""
    dsl = sample.get('dsl', '')
    if not dsl:
        return None
    
    mutations = MUTATION_TYPES.get(mutation_category, [])
    if not mutations:
        return None
    
    # 随机选择一个变异
    mutation = random.choice(mutations)
    
    # 应用变异
    mutated_dsl = mutation.mutator(dsl)
    
    # 如果变异没有效果，跳过
    if mutated_dsl == dsl:
        return None
    
    dsl_hash = hashlib.md5(mutated_dsl.encode()).hexdigest()[:8]
    
    return {
        'id': f"neg_{mutation_category}_{mutation.name}_{dsl_hash}",
        'dsl': mutated_dsl,
        'original_id': sample.get('id'),
        'mutation_category': mutation_category,
        'mutation_type': mutation.name,
        'mutation_description': mutation.description,
        'valid': False,
        'original_dsl': dsl,
        'complexity': sample.get('complexity', 'unknown'),
    }


def load_samples(input_path: Path) -> Generator[dict, None, None]:
    """加载样本"""
    if input_path.is_file():
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif input_path.is_dir():
        for jsonl_file in sorted(input_path.glob('*.jsonl')):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description='Generate negative samples for DPO training')
    parser.add_argument('--input', type=Path, required=True,
                        help='Input JSONL file or directory with valid DSL samples')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output directory for negative samples')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='Ratio of negative samples to generate per valid sample')
    parser.add_argument('--categories', type=str, nargs='+',
                        choices=list(MUTATION_TYPES.keys()),
                        default=list(MUTATION_TYPES.keys()),
                        help='Mutation categories to use')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of input samples to process')
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 按类别分文件
    category_files = {
        cat: open(args.output / f'{cat}.jsonl', 'w', encoding='utf-8')
        for cat in args.categories
    }
    
    # 统计
    stats = {cat: 0 for cat in args.categories}
    total_input = 0
    
    try:
        for sample in load_samples(args.input):
            if args.limit and total_input >= args.limit:
                break
            
            if not sample.get('valid', True):
                continue
            
            total_input += 1
            
            # 对每个类别生成负样本
            for category in args.categories:
                # 按比例决定是否生成
                if random.random() > args.ratio:
                    continue
                
                neg_sample = generate_negative_sample(sample, category)
                if neg_sample:
                    category_files[category].write(json.dumps(neg_sample, ensure_ascii=False) + '\n')
                    stats[category] += 1
    
    finally:
        for f in category_files.values():
            f.close()
    
    # 输出统计
    print(f"\n=== Negative Sample Generation Summary ===")
    print(f"Input samples processed: {total_input}")
    print(f"Negative samples by category:")
    for cat, count in stats.items():
        print(f"  {cat}: {count}")
    print(f"Total negative samples: {sum(stats.values())}")
    print(f"Output directory: {args.output}")
    
    # 保存统计
    with open(args.output / 'negative_stats.json', 'w', encoding='utf-8') as f:
        json.dump({
            'input_samples': total_input,
            'by_category': stats,
            'total': sum(stats.values()),
            'ratio': args.ratio,
        }, f, indent=2)


if __name__ == '__main__':
    main()
