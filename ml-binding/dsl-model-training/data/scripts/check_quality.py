#!/usr/bin/env python3
"""快速检查 synthetic/ 数据质量"""

import json
import re
import random
import os
from pathlib import Path

def quick_validate(dsl: str) -> list[str]:
    """简单验证 DSL 语法"""
    errors = []
    
    # 1. 括号匹配
    if dsl.count('{') != dsl.count('}'):
        errors.append('brace_mismatch')
    if dsl.count('(') != dsl.count(')'):
        errors.append('paren_mismatch')
    if dsl.count('"') % 2 != 0:
        errors.append('quote_mismatch')
    
    # 2. 检查信号定义
    defined = set(re.findall(r'SIGNAL\s+\w+\s+(\w+)', dsl))
    
    # 3. 检查 ROUTE 必需字段
    if 'ROUTE' in dsl and 'PRIORITY' not in dsl:
        errors.append('route_missing_priority')
    if 'ROUTE' in dsl and 'MODEL' not in dsl:
        errors.append('route_missing_model')
    
    # 4. 检查 WHEN 引用的信号是否已定义
    sig_types = {'keyword', 'embedding', 'domain', 'fact_check', 'user_feedback',
                 'preference', 'language', 'context', 'complexity', 'modality',
                 'authz', 'jailbreak', 'pii'}
    for match in re.finditer(r'(\w+)\s*\(\s*"(\w+)"\s*\)', dsl):
        sig_type, sig_name = match.groups()
        if sig_type in sig_types and sig_name not in defined:
            errors.append(f'undefined_signal:{sig_name}')
    
    return errors


def main():
    data_dir = Path(__file__).parent.parent
    synthetic_dir = data_dir / 'synthetic'
    
    # 加载样本
    samples = []
    all_file = synthetic_dir / 'synthetic_all.jsonl'
    
    if all_file.exists():
        with open(all_file) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:
        for fname in sorted(synthetic_dir.glob('L*.jsonl')):
            with open(fname) as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
    
    print(f'Total samples: {len(samples)}')
    
    # 抽样 500 个
    random.seed(42)
    sample_size = min(500, len(samples))
    sample_set = random.sample(samples, sample_size)
    
    # 按级别统计
    level_stats = {}
    error_counts = {}
    invalid_examples = []
    
    for s in sample_set:
        dsl = s.get('dsl', '')
        level = s.get('complexity_level', s.get('level', 'unknown'))
        
        if level not in level_stats:
            level_stats[level] = {'valid': 0, 'invalid': 0}
        
        errs = quick_validate(dsl)
        if errs:
            level_stats[level]['invalid'] += 1
            for e in errs:
                error_counts[e] = error_counts.get(e, 0) + 1
            if len(invalid_examples) < 3:
                invalid_examples.append({'id': s.get('id', 'N/A'), 'errors': errs, 'dsl': dsl[:200]})
        else:
            level_stats[level]['valid'] += 1
    
    # 输出结果
    print(f'\n=== 抽样验证结果 (n={sample_size}) ===')
    total_valid = sum(v['valid'] for v in level_stats.values())
    total_invalid = sum(v['invalid'] for v in level_stats.values())
    print(f'✅ Valid: {total_valid} ({total_valid*100/sample_size:.1f}%)')
    print(f'❌ Invalid: {total_invalid} ({total_invalid*100/sample_size:.1f}%)')
    
    print('\n--- 按复杂度级别 ---')
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        total = stats['valid'] + stats['invalid']
        pct = stats['valid'] * 100 / total if total > 0 else 0
        status = '✅' if pct >= 95 else '⚠️' if pct >= 90 else '❌'
        print(f'  {status} {level}: {stats["valid"]}/{total} valid ({pct:.1f}%)')
    
    if error_counts:
        print('\n--- 错误类型分布 ---')
        for e, c in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
            print(f'  {e}: {c}')
    
    if invalid_examples:
        print('\n--- 错误示例 ---')
        for ex in invalid_examples:
            print(f'\nID: {ex["id"]}')
            print(f'Errors: {ex["errors"]}')
            print(f'DSL: {ex["dsl"]}...')


if __name__ == '__main__':
    main()
