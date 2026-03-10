#!/usr/bin/env python3
"""检查 synthetic/ 数据质量

支持两种模式：
- 抽样模式（默认）: 随机抽取部分样本快速检查
- 全量模式（--full）: 检查所有样本

用法:
    python check_quality.py              # 抽样 500 个
    python check_quality.py --full       # 全量检查
    python check_quality.py -n 1000      # 抽样 1000 个
    python check_quality.py --output errors.jsonl  # 输出错误样本
"""

import argparse
import json
import re
import random
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator

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


def validate_sample(sample: dict) -> tuple[dict, list[str]]:
    """验证单个样本，返回 (样本, 错误列表)"""
    dsl = sample.get('dsl', '')
    errs = quick_validate(dsl)
    return sample, errs


def load_samples(synthetic_dir: Path) -> Iterator[dict]:
    """生成器：逐个加载样本，节省内存"""
    all_file = synthetic_dir / 'synthetic_all.jsonl'
    
    if all_file.exists():
        with open(all_file) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        for fname in sorted(synthetic_dir.glob('L*.jsonl')):
            with open(fname) as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def count_samples(synthetic_dir: Path) -> int:
    """快速统计样本总数"""
    all_file = synthetic_dir / 'synthetic_all.jsonl'
    count = 0
    
    if all_file.exists():
        with open(all_file) as f:
            for _ in f:
                count += 1
    else:
        for fname in sorted(synthetic_dir.glob('L*.jsonl')):
            with open(fname) as f:
                for _ in f:
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description='检查 DSL 数据质量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    python check_quality.py                    # 默认抽样 500 个
    python check_quality.py --full             # 全量检查
    python check_quality.py -n 2000            # 抽样 2000 个
    python check_quality.py --full --output errors.jsonl  # 全量检查并输出错误
    python check_quality.py --input /path/to/data  # 指定数据目录
        '''
    )
    parser.add_argument('--full', action='store_true',
                        help='全量检查所有样本（而非抽样）')
    parser.add_argument('-n', '--sample-size', type=int, default=500,
                        help='抽样数量（默认 500，--full 时忽略）')
    parser.add_argument('--input', type=Path, default=None,
                        help='输入数据目录（默认为 ../synthetic）')
    parser.add_argument('--output', type=Path, default=None,
                        help='将错误样本输出到指定 JSONL 文件')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行处理的 worker 数量（全量模式下使用）')
    parser.add_argument('--show-examples', type=int, default=5,
                        help='显示的错误示例数量（默认 5）')
    args = parser.parse_args()
    
    # 确定数据目录
    data_dir = Path(__file__).parent.parent
    synthetic_dir = args.input if args.input else data_dir / 'synthetic'
    
    if not synthetic_dir.exists():
        print(f'❌ 数据目录不存在: {synthetic_dir}')
        sys.exit(1)
    
    # 统计总数
    print(f'📁 数据目录: {synthetic_dir}')
    total_count = count_samples(synthetic_dir)
    print(f'📊 总样本数: {total_count:,}')
    
    if total_count == 0:
        print('❌ 没有找到任何样本')
        sys.exit(1)
    
    # 决定检查模式
    if args.full:
        mode_name = '全量检查'
        check_count = total_count
    else:
        check_count = min(args.sample_size, total_count)
        mode_name = f'抽样检查 (n={check_count})'
    
    print(f'🔍 模式: {mode_name}')
    print('-' * 50)
    
    # 加载样本
    if args.full:
        # 全量模式：直接使用生成器
        samples_to_check = list(load_samples(synthetic_dir))
    else:
        # 抽样模式：先加载再抽样
        all_samples = list(load_samples(synthetic_dir))
        random.seed(42)
        samples_to_check = random.sample(all_samples, check_count)
    
    # 按级别统计
    level_stats = {}
    error_counts = {}
    invalid_samples = []
    
    # 并行验证（全量模式下使用多进程）
    if args.full and len(samples_to_check) > 1000:
        print(f'⏳ 使用 {args.workers} 个 worker 并行验证...')
        processed = 0
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(validate_sample, s): s for s in samples_to_check}
            
            for future in as_completed(futures):
                sample, errs = future.result()
                level = sample.get('complexity_level', sample.get('level', 'unknown'))
                
                if level not in level_stats:
                    level_stats[level] = {'valid': 0, 'invalid': 0}
                
                if errs:
                    level_stats[level]['invalid'] += 1
                    for e in errs:
                        error_counts[e] = error_counts.get(e, 0) + 1
                    invalid_samples.append({
                        'id': sample.get('id', 'N/A'),
                        'level': level,
                        'errors': errs,
                        'dsl': sample.get('dsl', ''),
                    })
                else:
                    level_stats[level]['valid'] += 1
                
                processed += 1
                if processed % 5000 == 0:
                    print(f'  已处理: {processed:,}/{len(samples_to_check):,} ({processed*100/len(samples_to_check):.1f}%)')
    else:
        # 串行验证
        for i, sample in enumerate(samples_to_check):
            dsl = sample.get('dsl', '')
            level = sample.get('complexity_level', sample.get('level', 'unknown'))
            
            if level not in level_stats:
                level_stats[level] = {'valid': 0, 'invalid': 0}
            
            errs = quick_validate(dsl)
            if errs:
                level_stats[level]['invalid'] += 1
                for e in errs:
                    error_counts[e] = error_counts.get(e, 0) + 1
                invalid_samples.append({
                    'id': sample.get('id', 'N/A'),
                    'level': level,
                    'errors': errs,
                    'dsl': dsl,
                })
            else:
                level_stats[level]['valid'] += 1
            
            # 进度显示（全量模式）
            if args.full and (i + 1) % 5000 == 0:
                print(f'  已处理: {i+1:,}/{len(samples_to_check):,} ({(i+1)*100/len(samples_to_check):.1f}%)')
    
    # 输出结果
    checked_count = len(samples_to_check)
    total_valid = sum(v['valid'] for v in level_stats.values())
    total_invalid = sum(v['invalid'] for v in level_stats.values())
    
    print()
    print('=' * 50)
    print(f'📊 验证结果 ({mode_name})')
    print('=' * 50)
    print(f'✅ Valid:   {total_valid:,} ({total_valid*100/checked_count:.2f}%)')
    print(f'❌ Invalid: {total_invalid:,} ({total_invalid*100/checked_count:.2f}%)')
    
    print('\n--- 按复杂度级别 ---')
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        total = stats['valid'] + stats['invalid']
        pct = stats['valid'] * 100 / total if total > 0 else 0
        status = '✅' if pct >= 95 else '⚠️' if pct >= 90 else '❌'
        print(f'  {status} {level}: {stats["valid"]:,}/{total:,} valid ({pct:.1f}%)')
    
    if error_counts:
        print('\n--- 错误类型分布 ---')
        for e, c in sorted(error_counts.items(), key=lambda x: -x[1])[:15]:
            pct = c * 100 / total_invalid if total_invalid > 0 else 0
            print(f'  {e}: {c:,} ({pct:.1f}%)')
    
    # 显示错误示例
    if invalid_samples and args.show_examples > 0:
        print(f'\n--- 错误示例 (前 {min(args.show_examples, len(invalid_samples))} 个) ---')
        for ex in invalid_samples[:args.show_examples]:
            print(f'\nID: {ex["id"]}')
            print(f'Level: {ex["level"]}')
            print(f'Errors: {ex["errors"]}')
            dsl_preview = ex["dsl"][:300] + '...' if len(ex["dsl"]) > 300 else ex["dsl"]
            print(f'DSL:\n{dsl_preview}')
    
    # 输出错误样本到文件
    if args.output and invalid_samples:
        with open(args.output, 'w', encoding='utf-8') as f:
            for sample in invalid_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f'\n📝 错误样本已保存到: {args.output} ({len(invalid_samples):,} 条)')
    
    # 返回退出码
    if total_invalid > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
