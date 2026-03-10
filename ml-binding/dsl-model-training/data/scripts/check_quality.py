#!/usr/bin/env python3
"""检查 synthetic/ 数据质量

支持两种模式：
- 抽样模式（默认）: 随机抽取部分样本快速检查
- 全量模式（--full）: 检查所有样本

支持两种验证器：
- 内置验证器（默认）: 简单的正则检查，速度快
- Go 验证器（--use-go-validator）: 调用 sr-dsl 工具进行完整的三级验证

用法:
    python check_quality.py              # 抽样 500 个（内置验证）
    python check_quality.py --full       # 全量检查（内置验证）
    python check_quality.py --use-go-validator  # 使用 Go 验证器
    python check_quality.py -n 1000      # 抽样 1000 个
    python check_quality.py --output errors.jsonl  # 输出错误样本
"""

import argparse
import json
import re
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Iterator, Optional

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


# ============== Go 验证器支持 ==============

@dataclass
class GoValidationResult:
    """Go 验证器结果"""
    valid: bool
    errors: list[str]      # Level 1: 语法错误
    warnings: list[str]    # Level 2: 引用错误
    constraints: list[str] # Level 3: 约束违反
    
    @property
    def all_issues(self) -> list[str]:
        """返回所有问题"""
        issues = []
        for e in self.errors:
            issues.append(f'error:{e}')
        for w in self.warnings:
            issues.append(f'warning:{w}')
        for c in self.constraints:
            issues.append(f'constraint:{c}')
        return issues


# Go DSL 工具的可能路径
GO_DSL_PATHS = [
    # 项目内编译的路径
    Path(__file__).parent.parent.parent.parent.parent / 'src' / 'semantic-router' / 'bin' / 'sr-dsl',
    # go install 安装的路径
    Path.home() / 'go' / 'bin' / 'sr-dsl',
    # 系统 PATH 中
    'sr-dsl',
]


def find_go_validator() -> Optional[str]:
    """查找 Go DSL 验证器可执行文件"""
    for path in GO_DSL_PATHS:
        if isinstance(path, Path):
            if path.exists():
                return str(path)
        else:
            # 检查 PATH 中是否存在
            if shutil.which(path):
                return path
    return None


def validate_with_go(dsl: str, go_bin: str) -> GoValidationResult:
    """使用 Go 验证器验证 DSL
    
    调用 sr-dsl validate 命令，解析输出结果。
    """
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dsl', delete=False) as f:
        f.write(dsl)
        tmp_path = f.name
    
    try:
        result = subprocess.run(
            [go_bin, 'validate', tmp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # 解析输出
        errors = []
        warnings = []
        constraints = []
        
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith('Summary:') or line == 'No issues found.':
                continue
            
            # 解析诊断格式: 🔴 Error: ... / 🟡 Warning: ... / 🟠 Constraint: ...
            if '🔴' in line or 'Error:' in line:
                msg = line.split('Error:', 1)[-1].strip() if 'Error:' in line else line
                errors.append(msg)
            elif '🟡' in line or 'Warning:' in line:
                msg = line.split('Warning:', 1)[-1].strip() if 'Warning:' in line else line
                warnings.append(msg)
            elif '🟠' in line or 'Constraint:' in line:
                msg = line.split('Constraint:', 1)[-1].strip() if 'Constraint:' in line else line
                constraints.append(msg)
            elif result.returncode != 0:
                # 其他输出在失败时当作错误
                errors.append(line)
        
        # stderr 中的内容也作为错误
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                if line.strip():
                    errors.append(line.strip())
        
        valid = result.returncode == 0 and len(errors) == 0
        
        return GoValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            constraints=constraints
        )
        
    except subprocess.TimeoutExpired:
        return GoValidationResult(
            valid=False,
            errors=['Validation timeout'],
            warnings=[],
            constraints=[]
        )
    except Exception as e:
        return GoValidationResult(
            valid=False,
            errors=[str(e)],
            warnings=[],
            constraints=[]
        )
    finally:
        # 清理临时文件
        Path(tmp_path).unlink(missing_ok=True)


def validate_sample_with_go(args: tuple[dict, str]) -> tuple[dict, list[str]]:
    """使用 Go 验证器验证单个样本（用于并行处理）"""
    sample, go_bin = args
    dsl = sample.get('dsl', '')
    result = validate_with_go(dsl, go_bin)
    return sample, result.all_issues


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
    python check_quality.py --use-go-validator     # 使用 Go 验证器（更严格）
    python check_quality.py --go-bin /path/to/sr-dsl  # 指定 Go 验证器路径
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
    parser.add_argument('--use-go-validator', action='store_true',
                        help='使用 Go 的 sr-dsl 验证器（更严格的三级验证）')
    parser.add_argument('--go-bin', type=Path, default=None,
                        help='指定 Go 验证器 sr-dsl 的路径')
    parser.add_argument('--strict', action='store_true',
                        help='严格模式：warnings 和 constraints 也算作错误')
    args = parser.parse_args()
    
    # 确定数据目录
    data_dir = Path(__file__).parent.parent
    synthetic_dir = args.input if args.input else data_dir / 'synthetic'
    
    if not synthetic_dir.exists():
        print(f'❌ 数据目录不存在: {synthetic_dir}')
        sys.exit(1)
    
    # 检查 Go 验证器
    go_bin = None
    if args.use_go_validator:
        if args.go_bin:
            go_bin = str(args.go_bin)
            if not Path(go_bin).exists() and not shutil.which(go_bin):
                print(f'❌ 指定的 Go 验证器不存在: {go_bin}')
                sys.exit(1)
        else:
            go_bin = find_go_validator()
            if not go_bin:
                print('❌ 未找到 Go 验证器 sr-dsl')
                print('   请先编译 Go 验证器:')
                print('   cd src/semantic-router && go build -o bin/sr-dsl ./cmd/dsl')
                print('   或使用 --go-bin 指定路径')
                sys.exit(1)
        print(f'🔧 使用 Go 验证器: {go_bin}')
    
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
    
    validator_name = 'Go 三级验证' if go_bin else '内置快速验证'
    print(f'🔍 模式: {mode_name} ({validator_name})')
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
    warning_counts = {}
    constraint_counts = {}
    invalid_samples = []
    
    # 选择验证函数
    if go_bin:
        # 使用 Go 验证器（ThreadPoolExecutor，因为是 I/O 密集型）
        print(f'⏳ 使用 {args.workers} 个 worker 并行验证...')
        processed = 0
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(validate_sample_with_go, (s, go_bin)): s 
                for s in samples_to_check
            }
            
            for future in as_completed(futures):
                sample, issues = future.result()
                level = sample.get('complexity', sample.get('complexity_level', sample.get('level', 'unknown')))
                
                if level not in level_stats:
                    level_stats[level] = {'valid': 0, 'invalid': 0, 'warnings': 0, 'constraints': 0}
                
                # 分类统计
                errors = [i for i in issues if i.startswith('error:')]
                warnings = [i for i in issues if i.startswith('warning:')]
                constraints = [i for i in issues if i.startswith('constraint:')]
                
                # 在严格模式下，warnings 和 constraints 也算错误
                has_error = len(errors) > 0
                if args.strict:
                    has_error = has_error or len(warnings) > 0 or len(constraints) > 0
                
                if has_error:
                    level_stats[level]['invalid'] += 1
                    for e in errors:
                        error_counts[e] = error_counts.get(e, 0) + 1
                    for w in warnings:
                        warning_counts[w] = warning_counts.get(w, 0) + 1
                    for c in constraints:
                        constraint_counts[c] = constraint_counts.get(c, 0) + 1
                    invalid_samples.append({
                        'id': sample.get('id', 'N/A'),
                        'level': level,
                        'errors': errors,
                        'warnings': warnings,
                        'constraints': constraints,
                        'dsl': sample.get('dsl', ''),
                    })
                else:
                    level_stats[level]['valid'] += 1
                    level_stats[level]['warnings'] += len(warnings)
                    level_stats[level]['constraints'] += len(constraints)
                
                processed += 1
                if processed % 1000 == 0:
                    print(f'  已处理: {processed:,}/{len(samples_to_check):,} ({processed*100/len(samples_to_check):.1f}%)')
    
    elif args.full and len(samples_to_check) > 1000:
        # 内置验证器 + 并行（ProcessPoolExecutor）
        print(f'⏳ 使用 {args.workers} 个 worker 并行验证...')
        processed = 0
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(validate_sample, s): s for s in samples_to_check}
            
            for future in as_completed(futures):
                sample, errs = future.result()
                level = sample.get('complexity', sample.get('complexity_level', sample.get('level', 'unknown')))
                
                if level not in level_stats:
                    level_stats[level] = {'valid': 0, 'invalid': 0, 'warnings': 0, 'constraints': 0}
                
                if errs:
                    level_stats[level]['invalid'] += 1
                    for e in errs:
                        error_counts[e] = error_counts.get(e, 0) + 1
                    invalid_samples.append({
                        'id': sample.get('id', 'N/A'),
                        'level': level,
                        'errors': errs,
                        'warnings': [],
                        'constraints': [],
                        'dsl': sample.get('dsl', ''),
                    })
                else:
                    level_stats[level]['valid'] += 1
                
                processed += 1
                if processed % 5000 == 0:
                    print(f'  已处理: {processed:,}/{len(samples_to_check):,} ({processed*100/len(samples_to_check):.1f}%)')
    else:
        # 内置验证器 + 串行
        for i, sample in enumerate(samples_to_check):
            dsl = sample.get('dsl', '')
            level = sample.get('complexity', sample.get('complexity_level', sample.get('level', 'unknown')))
            
            if level not in level_stats:
                level_stats[level] = {'valid': 0, 'invalid': 0, 'warnings': 0, 'constraints': 0}
            
            errs = quick_validate(dsl)
            if errs:
                level_stats[level]['invalid'] += 1
                for e in errs:
                    error_counts[e] = error_counts.get(e, 0) + 1
                invalid_samples.append({
                    'id': sample.get('id', 'N/A'),
                    'level': level,
                    'errors': errs,
                    'warnings': [],
                    'constraints': [],
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
    total_warnings = sum(v.get('warnings', 0) for v in level_stats.values())
    total_constraints = sum(v.get('constraints', 0) for v in level_stats.values())
    
    print()
    print('=' * 50)
    print(f'📊 验证结果 ({mode_name})')
    print('=' * 50)
    print(f'✅ Valid:   {total_valid:,} ({total_valid*100/checked_count:.2f}%)')
    print(f'❌ Invalid: {total_invalid:,} ({total_invalid*100/checked_count:.2f}%)')
    
    if go_bin:
        # Go 验证器显示详细分类
        total_errors_count = sum(error_counts.values())
        total_warnings_count = sum(warning_counts.values())
        total_constraints_count = sum(constraint_counts.values())
        print(f'\n--- 问题分类统计 ---')
        print(f'  🔴 Errors:      {total_errors_count:,}')
        print(f'  🟡 Warnings:    {total_warnings_count:,}')
        print(f'  🟠 Constraints: {total_constraints_count:,}')
    
    print('\n--- 按复杂度级别 ---')
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        total = stats['valid'] + stats['invalid']
        pct = stats['valid'] * 100 / total if total > 0 else 0
        status = '✅' if pct >= 95 else '⚠️' if pct >= 90 else '❌'
        print(f'  {status} {level}: {stats["valid"]:,}/{total:,} valid ({pct:.1f}%)')
    
    if error_counts:
        print('\n--- 错误类型分布 (Top 15) ---')
        for e, c in sorted(error_counts.items(), key=lambda x: -x[1])[:15]:
            pct = c * 100 / total_invalid if total_invalid > 0 else 0
            print(f'  {e}: {c:,} ({pct:.1f}%)')
    
    if warning_counts and go_bin:
        print('\n--- 警告类型分布 (Top 10) ---')
        for w, c in sorted(warning_counts.items(), key=lambda x: -x[1])[:10]:
            print(f'  {w}: {c:,}')
    
    if constraint_counts and go_bin:
        print('\n--- 约束违反分布 (Top 10) ---')
        for c, cnt in sorted(constraint_counts.items(), key=lambda x: -x[1])[:10]:
            print(f'  {c}: {cnt:,}')
    
    # 显示错误示例
    if invalid_samples and args.show_examples > 0:
        print(f'\n--- 错误示例 (前 {min(args.show_examples, len(invalid_samples))} 个) ---')
        for ex in invalid_samples[:args.show_examples]:
            print(f'\nID: {ex["id"]}')
            print(f'Level: {ex["level"]}')
            if ex.get('errors'):
                print(f'Errors: {ex["errors"]}')
            if ex.get('warnings'):
                print(f'Warnings: {ex["warnings"]}')
            if ex.get('constraints'):
                print(f'Constraints: {ex["constraints"]}')
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
