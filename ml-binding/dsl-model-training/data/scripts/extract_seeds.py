#!/usr/bin/env python3
"""
DSL 种子数据提取器

从项目代码库中提取高质量的 DSL 示例作为训练种子数据：
1. 从 dsl_test.go 测试文件中提取有效 DSL 片段
2. 从 config/*.yaml 配置文件反编译为 DSL
3. 从 DslGuide.tsx 前端文档中提取示例

用法:
    python extract_seeds.py --repo-root /path/to/semantic-router --output seeds/
"""

import argparse
import json
import re
import subprocess
import hashlib
from pathlib import Path
from typing import Generator


def extract_from_go_tests(test_file: Path) -> Generator[dict, None, None]:
    """从 Go 测试文件中提取 DSL 字符串字面量。
    
    匹配模式:
    1. 反引号多行字符串: `SIGNAL ... }`
    2. 普通字符串中的 DSL 片段
    """
    if not test_file.exists():
        print(f"Warning: Test file not found: {test_file}")
        return
    
    content = test_file.read_text(encoding='utf-8')
    
    # 模式1: 反引号多行字符串 (最常见)
    # 匹配以 SIGNAL/ROUTE/PLUGIN/BACKEND/GLOBAL 开头的反引号字符串
    backtick_pattern = r'`((?:SIGNAL|ROUTE|PLUGIN|BACKEND|GLOBAL)[\s\S]*?)`'
    
    for match in re.finditer(backtick_pattern, content):
        dsl_text = match.group(1).strip()
        if len(dsl_text) > 20:  # 过滤太短的片段
            # 计算唯一ID
            dsl_hash = hashlib.md5(dsl_text.encode()).hexdigest()[:8]
            yield {
                'id': f'test_{dsl_hash}',
                'dsl': dsl_text,
                'source': str(test_file),
                'extraction_method': 'go_test_backtick',
                'valid': True,  # 来自测试文件，假设有效
            }

    # 模式2: 命名常量 (如 fullDSLExample)
    const_pattern = r'const\s+(\w+)\s*=\s*`((?:SIGNAL|ROUTE|PLUGIN|BACKEND|GLOBAL|#)[\s\S]*?)`'
    for match in re.finditer(const_pattern, content):
        const_name = match.group(1)
        dsl_text = match.group(2).strip()
        if len(dsl_text) > 50:
            dsl_hash = hashlib.md5(dsl_text.encode()).hexdigest()[:8]
            yield {
                'id': f'test_const_{const_name}_{dsl_hash}',
                'dsl': dsl_text,
                'source': str(test_file),
                'extraction_method': 'go_test_const',
                'const_name': const_name,
                'valid': True,
            }


def extract_from_tsx_guide(guide_file: Path) -> Generator[dict, None, None]:
    """从前端 DslGuide.tsx 组件中提取 DSL 示例。"""
    if not guide_file.exists():
        print(f"Warning: Guide file not found: {guide_file}")
        return
    
    content = guide_file.read_text(encoding='utf-8')
    
    # 匹配 TypeScript 模板字符串中的 DSL
    # 模式: `SIGNAL ... GLOBAL { ... }`
    template_pattern = r'`((?:# |SIGNAL|ROUTE|PLUGIN|BACKEND|GLOBAL)[\s\S]*?)`'
    
    for match in re.finditer(template_pattern, content):
        dsl_text = match.group(1).strip()
        # 过滤掉太短或不完整的片段
        if len(dsl_text) > 100 and ('SIGNAL' in dsl_text or 'ROUTE' in dsl_text):
            dsl_hash = hashlib.md5(dsl_text.encode()).hexdigest()[:8]
            yield {
                'id': f'guide_{dsl_hash}',
                'dsl': dsl_text,
                'source': str(guide_file),
                'extraction_method': 'tsx_template',
                'valid': True,
            }


def decompile_yaml_configs(config_dir: Path, decompiler_bin: Path | None = None) -> Generator[dict, None, None]:
    """从 YAML 配置文件反编译为 DSL。
    
    如果提供了 decompiler_bin，使用该二进制进行反编译；
    否则跳过此步骤（需要先构建 decompiler）。
    """
    if decompiler_bin and not decompiler_bin.exists():
        print(f"Warning: Decompiler binary not found: {decompiler_bin}")
        print("Skipping YAML decompilation. Build the decompiler first.")
        return
    
    yaml_files = list(config_dir.glob('**/*.yaml'))
    print(f"Found {len(yaml_files)} YAML files in {config_dir}")
    
    for yaml_file in yaml_files:
        # 跳过非路由配置文件
        if 'envoy' in yaml_file.name.lower():
            continue
        
        try:
            if decompiler_bin:
                # 使用 decompiler 二进制
                result = subprocess.run(
                    [str(decompiler_bin), '--input', str(yaml_file), '--format', 'dsl'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    dsl_text = result.stdout.strip()
                    dsl_hash = hashlib.md5(dsl_text.encode()).hexdigest()[:8]
                    yield {
                        'id': f'yaml_{yaml_file.stem}_{dsl_hash}',
                        'dsl': dsl_text,
                        'source': str(yaml_file),
                        'extraction_method': 'yaml_decompile',
                        'valid': True,
                    }
            else:
                # 没有 decompiler，记录文件路径供后续处理
                yield {
                    'id': f'yaml_pending_{yaml_file.stem}',
                    'yaml_path': str(yaml_file),
                    'source': str(yaml_file),
                    'extraction_method': 'yaml_pending',
                    'valid': None,  # 待验证
                }
        except subprocess.TimeoutExpired:
            print(f"Warning: Timeout decompiling {yaml_file}")
        except Exception as e:
            print(f"Warning: Error decompiling {yaml_file}: {e}")


def classify_complexity(dsl_text: str) -> str:
    """根据 DSL 内容推断复杂度级别。"""
    signal_count = len(re.findall(r'\bSIGNAL\s+', dsl_text))
    route_count = len(re.findall(r'\bROUTE\s+', dsl_text))
    plugin_count = len(re.findall(r'\bPLUGIN\s+', dsl_text))
    backend_count = len(re.findall(r'\bBACKEND\s+', dsl_text))
    
    # 条件深度估算 (基于 AND/OR/NOT 数量)
    bool_ops = len(re.findall(r'\b(AND|OR|NOT)\b', dsl_text))
    nested_parens = dsl_text.count('(') - dsl_text.count('("')  # 排除字符串中的括号
    
    total_constructs = signal_count + route_count + plugin_count + backend_count
    condition_depth = min(bool_ops, nested_parens // 2 + 1)
    
    if total_constructs <= 3 and condition_depth == 0:
        return 'L1'
    elif total_constructs <= 6 and condition_depth <= 1:
        return 'L2'
    elif total_constructs <= 10 and condition_depth <= 2:
        return 'L3'
    elif total_constructs <= 15 and condition_depth <= 3:
        return 'L4'
    else:
        return 'L5'


def extract_signal_types(dsl_text: str) -> list[str]:
    """提取 DSL 中使用的信号类型。"""
    pattern = r'\bSIGNAL\s+(\w+)\s+'
    return list(set(re.findall(pattern, dsl_text)))


def extract_plugin_types(dsl_text: str) -> list[str]:
    """提取 DSL 中使用的插件类型。"""
    # PLUGIN <name> <type> { 或 PLUGIN <type> {
    pattern1 = r'\bPLUGIN\s+\w+\s+(\w+)\s*\{'
    pattern2 = r'\bPLUGIN\s+(\w+)\s*\{'
    types = set(re.findall(pattern1, dsl_text))
    types.update(re.findall(pattern2, dsl_text))
    return list(types)


def extract_algorithm_types(dsl_text: str) -> list[str]:
    """提取 DSL 中使用的算法类型。"""
    pattern = r'\bALGORITHM\s+(\w+)\s*\{'
    return list(set(re.findall(pattern, dsl_text)))


def enrich_metadata(sample: dict) -> dict:
    """为样本添加元数据。"""
    dsl_text = sample.get('dsl', '')
    if not dsl_text:
        return sample
    
    sample['metadata'] = {
        'complexity': classify_complexity(dsl_text),
        'signal_types': extract_signal_types(dsl_text),
        'plugin_types': extract_plugin_types(dsl_text),
        'algorithm_types': extract_algorithm_types(dsl_text),
        'num_signals': len(re.findall(r'\bSIGNAL\s+', dsl_text)),
        'num_routes': len(re.findall(r'\bROUTE\s+', dsl_text)),
        'num_plugins': len(re.findall(r'\bPLUGIN\s+', dsl_text)),
        'num_backends': len(re.findall(r'\bBACKEND\s+', dsl_text)),
        'has_global': 'GLOBAL' in dsl_text,
        'char_count': len(dsl_text),
        'line_count': dsl_text.count('\n') + 1,
    }
    return sample


def main():
    parser = argparse.ArgumentParser(description='Extract DSL seed data from project codebase')
    parser.add_argument('--repo-root', type=Path, required=True,
                        help='Root directory of semantic-router repository')
    parser.add_argument('--output', type=Path, default=Path('seeds/'),
                        help='Output directory for extracted seeds')
    parser.add_argument('--decompiler', type=Path, default=None,
                        help='Path to DSL decompiler binary (optional)')
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.output.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    # 1. 从测试文件提取
    test_file = args.repo_root / 'src/semantic-router/pkg/dsl/dsl_test.go'
    print(f"\n=== Extracting from Go tests: {test_file} ===")
    test_samples = list(extract_from_go_tests(test_file))
    print(f"Extracted {len(test_samples)} samples from tests")
    all_samples.extend(test_samples)
    
    # 保存测试来源的样本
    with open(args.output / 'from_tests.jsonl', 'w', encoding='utf-8') as f:
        for sample in test_samples:
            sample = enrich_metadata(sample)
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 2. 从前端指南提取
    guide_file = args.repo_root / 'dashboard/frontend/src/components/DslGuide.tsx'
    print(f"\n=== Extracting from TSX guide: {guide_file} ===")
    guide_samples = list(extract_from_tsx_guide(guide_file))
    print(f"Extracted {len(guide_samples)} samples from guide")
    all_samples.extend(guide_samples)
    
    # 保存指南来源的样本
    with open(args.output / 'from_guide.jsonl', 'w', encoding='utf-8') as f:
        for sample in guide_samples:
            sample = enrich_metadata(sample)
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 3. 从 YAML 配置反编译
    config_dir = args.repo_root / 'config'
    print(f"\n=== Decompiling YAML configs: {config_dir} ===")
    yaml_samples = list(decompile_yaml_configs(config_dir, args.decompiler))
    print(f"Processed {len(yaml_samples)} YAML files")
    all_samples.extend([s for s in yaml_samples if s.get('dsl')])
    
    # 保存 YAML 来源的样本
    with open(args.output / 'from_yaml.jsonl', 'w', encoding='utf-8') as f:
        for sample in yaml_samples:
            if sample.get('dsl'):
                sample = enrich_metadata(sample)
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 4. 去重
    seen_hashes = set()
    unique_samples = []
    for sample in all_samples:
        dsl = sample.get('dsl', '')
        if dsl:
            h = hashlib.md5(dsl.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_samples.append(sample)
    
    # 5. 统计报告
    print(f"\n=== Summary ===")
    print(f"Total samples extracted: {len(all_samples)}")
    print(f"Unique samples (after dedup): {len(unique_samples)}")
    
    # 按复杂度分布
    complexity_dist = {}
    for sample in unique_samples:
        c = sample.get('metadata', {}).get('complexity', 'unknown')
        complexity_dist[c] = complexity_dist.get(c, 0) + 1
    print(f"Complexity distribution: {complexity_dist}")
    
    # 保存统计
    stats = {
        'total_samples': len(all_samples),
        'unique_samples': len(unique_samples),
        'by_source': {
            'from_tests': len(test_samples),
            'from_guide': len(guide_samples),
            'from_yaml': len([s for s in yaml_samples if s.get('dsl')]),
        },
        'by_complexity': complexity_dist,
    }
    with open(args.output / 'extraction_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nOutput saved to: {args.output}")


if __name__ == '__main__':
    main()
