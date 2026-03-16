#!/usr/bin/env python3
"""
负样本生成器

通过变异有效 DSL 生成各种类型的错误样本，用于 DPO (Direct Preference Optimization) 训练。

变异类型:
1. syntax_error: 语法错误 (删除括号、拼错关键字)
2. reference_error: 引用错误 (WHEN 引用未定义信号)
3. constraint_violation: 约束违反 (threshold: 1.5)
4. schema_mismatch: Schema 不匹配 (字段用错类型)
5. bool_expr_error: 布尔表达式错误 (AND/OR/NOT 嵌套组合错误)
6. model_error: MODEL 声明错误 (多模型语法、属性重复)
7. structural_error: 结构错误 (缺失必要块、块顺序错误)
8. semantic_error: 语义/逻辑错误 (死代码、路由冲突、优先级碰撞)
9. global_error: GLOBAL 配置错误 (缺失、重复、无效默认值)
10. backend_error: BACKEND 配置错误 (缺失字段、无效配置)
11. encoding_error: 编码/格式错误 (混合缩进、特殊字符)

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


MODEL_CANDIDATES = [
    "gpt-4o",
    "gpt-4o-mini",
    "deepseek-r1",
    "deepseek-coder",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen2.5:32b",
    "claude-3-sonnet",
]


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


# ============== 布尔表达式错误 ==============

def mutate_bool_paren_mismatch(dsl: str) -> str:
    """括号不匹配: 删除 WHEN 表达式中的一个括号"""
    # 找到 WHEN 子句
    when_pattern = r'(WHEN\s+)(.+?)(\n\s*\n|\n\s*MODEL)'
    match = re.search(when_pattern, dsl, re.DOTALL)
    if match:
        when_expr = match.group(2)
        # 找到括号位置
        open_parens = [i for i, c in enumerate(when_expr) if c == '(']
        close_parens = [i for i, c in enumerate(when_expr) if c == ')']
        
        # 删除一个闭合括号（制造不匹配）
        if close_parens:
            pos = random.choice(close_parens)
            new_expr = when_expr[:pos] + when_expr[pos+1:]
            return dsl[:match.start(2)] + new_expr + dsl[match.end(2):]
        # 或删除一个开括号
        elif open_parens:
            pos = random.choice(open_parens)
            new_expr = when_expr[:pos] + when_expr[pos+1:]
            return dsl[:match.start(2)] + new_expr + dsl[match.end(2):]
    return dsl


def mutate_bool_missing_operand(dsl: str) -> str:
    """操作数缺失: a AND b → a AND"""
    # 找到 AND 或 OR 后面跟着的操作数，删除它
    pattern = r'(\b(?:AND|OR)\s+)(\w+\("[^"]+"\))'
    matches = list(re.finditer(pattern, dsl))
    if matches:
        match = random.choice(matches)
        # 只保留操作符，删除右操作数
        return dsl[:match.start(2)] + dsl[match.end():]
    return dsl


def mutate_bool_double_operator(dsl: str) -> str:
    """重复操作符: a AND b → a AND AND b"""
    ops = ['AND', 'OR']
    for op in ops:
        pattern = rf'(\s+{op}\s+)'
        match = re.search(pattern, dsl)
        if match:
            # 重复操作符
            return dsl[:match.end()] + f'{op} ' + dsl[match.end():]
    return dsl


def mutate_bool_invalid_nesting(dsl: str) -> str:
    """非法嵌套: NOT NOT a, 或 AND OR b"""
    # 找到 NOT 表达式，变成 NOT NOT
    not_pattern = r'(NOT\s+)(\w+\("[^"]+"\))'
    match = re.search(not_pattern, dsl)
    if match:
        return dsl[:match.start()] + f'NOT NOT {match.group(2)}' + dsl[match.end():]
    
    # 找到 AND，变成 AND OR
    and_pattern = r'(\bAND\s+)'
    match = re.search(and_pattern, dsl)
    if match:
        return dsl[:match.start()] + 'AND OR ' + dsl[match.end():]
    
    return dsl


def mutate_bool_wrong_precedence(dsl: str) -> str:
    """优先级混淆: 移除必要的括号导致语义歧义"""
    # 找到 (a OR b) AND c 形式，去掉括号变成 a OR b AND c
    pattern = r'\((\w+\("[^"]+"\)\s+OR\s+\w+\("[^"]+"\))\)\s+AND'
    match = re.search(pattern, dsl)
    if match:
        # 去掉括号
        inner = match.group(1)
        return dsl[:match.start()] + inner + ' AND' + dsl[match.end():]
    return dsl


def mutate_bool_empty_expr(dsl: str) -> str:
    """空表达式: WHEN 后面没有条件"""
    when_pattern = r'(WHEN\s+)(\w+\("[^"]+"\))'
    match = re.search(when_pattern, dsl)
    if match:
        # 删除条件，只留 WHEN
        return dsl[:match.end(1)] + dsl[match.end():]
    return dsl


# ============== MODEL 错误 ==============

def mutate_model_trailing_comma(dsl: str) -> str:
    """尾逗号: MODEL "a", "b", → 多余逗号"""
    # 找到 MODEL 声明的最后一个模型
    model_pattern = r'(MODEL\s+"[^"]+")(\s*\(.*?\))?(\s*\n)'
    match = re.search(model_pattern, dsl, re.DOTALL)
    if match:
        # 在末尾加逗号
        end_part = match.group(2) or ''
        return dsl[:match.end(1)] + end_part + ',' + match.group(3) + dsl[match.end():]
    return dsl


def mutate_model_duplicate_attr(dsl: str) -> str:
    """重复属性: MODEL "m" (reasoning=true, reasoning=false)"""
    # 找到模型属性
    model_attr_pattern = r'(MODEL\s+"[^"]+"\s*\()([^)]+)(\))'
    match = re.search(model_attr_pattern, dsl)
    if match:
        attrs = match.group(2)
        # 找到一个属性并重复它
        attr_match = re.search(r'(\w+\s*=\s*\w+)', attrs)
        if attr_match:
            dup_attr = attr_match.group(1)
            new_attrs = attrs + ', ' + dup_attr
            return dsl[:match.start(2)] + new_attrs + dsl[match.end(2):]
    return dsl


def mutate_model_invalid_attr(dsl: str) -> str:
    """无效属性: MODEL "m" (unknown_attr = true)"""
    model_pattern = r'(MODEL\s+"[^"]+")(\s*\([^)]*\))?'
    match = re.search(model_pattern, dsl)
    if match:
        if match.group(2):
            # 已有属性，添加无效属性
            attrs = match.group(2).rstrip(')')
            return dsl[:match.end(1)] + attrs + ', invalid_attr = "bad"' + ')' + dsl[match.end():]
        else:
            # 没有属性，添加无效属性
            return dsl[:match.end(1)] + ' (invalid_attr = "bad")' + dsl[match.end():]
    return dsl


def mutate_model_empty_name(dsl: str) -> str:
    """空模型名: MODEL "" """
    model_pattern = r'(MODEL\s+)"([^"]+)"'
    match = re.search(model_pattern, dsl)
    if match:
        return dsl[:match.start(2)] + dsl[match.end(2):]
    return dsl


def mutate_model_missing_quotes(dsl: str) -> str:
    """缺少引号: MODEL gpt-4o 而不是 MODEL "gpt-4o" """
    model_pattern = r'(MODEL\s+)"([^"]+)"'
    match = re.search(model_pattern, dsl)
    if match:
        model_name = match.group(2)
        return dsl[:match.end(1)] + model_name + dsl[match.end():]
    return dsl


# ============== 结构错误 ==============

def mutate_struct_route_missing_model(dsl: str) -> str:
    """路由缺少 MODEL 声明"""
    # 找到 ROUTE 块中的 MODEL 行，删除它
    route_pattern = r'(ROUTE\s+\w+[^{]*\{[^}]*?)(MODEL\s+[^\n]+\n)([^}]*\})'
    match = re.search(route_pattern, dsl, re.DOTALL)
    if match:
        return dsl[:match.start(2)] + dsl[match.end(2):]
    return dsl


def mutate_struct_route_missing_priority(dsl: str) -> str:
    """路由缺少 PRIORITY 声明"""
    priority_pattern = r'(\s+PRIORITY\s+\d+\s*\n)'
    match = re.search(priority_pattern, dsl)
    if match:
        return dsl[:match.start()] + '\n' + dsl[match.end():]
    return dsl


def mutate_struct_signal_after_route(dsl: str) -> str:
    """SIGNAL 声明在 ROUTE 之后（顺序错误）"""
    # 找到第一个 SIGNAL 声明
    signal_pattern = r'(SIGNAL\s+\w+\s+\w+\s*\{[^}]*\}\n*)'
    signal_match = re.search(signal_pattern, dsl)
    if not signal_match:
        return dsl
    
    # 找到最后一个 ROUTE 块的结束位置
    route_pattern = r'(ROUTE\s+\w+[^{]*\{[^}]*\}\n*)'
    route_matches = list(re.finditer(route_pattern, dsl, re.DOTALL))
    if not route_matches:
        return dsl
    
    last_route = route_matches[-1]
    signal_text = signal_match.group(1)
    
    # 删除原 SIGNAL，在 ROUTE 后插入
    new_dsl = dsl[:signal_match.start()] + dsl[signal_match.end():]
    # 重新找到 ROUTE 位置（因为删除 SIGNAL 后位置变了）
    route_matches = list(re.finditer(route_pattern, new_dsl, re.DOTALL))
    if route_matches:
        last_route = route_matches[-1]
        new_dsl = new_dsl[:last_route.end()] + '\n' + signal_text + new_dsl[last_route.end():]
    
    return new_dsl


def mutate_struct_duplicate_route_name(dsl: str) -> str:
    """重复的路由名称"""
    route_pattern = r'ROUTE\s+(\w+)'
    matches = list(re.finditer(route_pattern, dsl))
    if len(matches) >= 2:
        # 把第二个路由名改成和第一个一样
        first_name = matches[0].group(1)
        second_match = matches[1]
        return dsl[:second_match.start(1)] + first_name + dsl[second_match.end(1):]
    return dsl


def mutate_struct_algorithm_single_model(dsl: str) -> str:
    """单 MODEL 却配置了 ALGORITHM（语义错误）"""
    # 找到只有单个 MODEL 的 ROUTE，给它加 ALGORITHM
    route_pattern = r'(ROUTE\s+\w+[^{]*\{[^}]*?MODEL\s+"[^"]+"\s*(?:\([^)]*\))?\s*\n)(\s*\})'
    match = re.search(route_pattern, dsl, re.DOTALL)
    if match:
        # 检查是否只有单个 MODEL（没有逗号分隔的多模型）
        route_content = match.group(1)
        if route_content.count('MODEL') == 1 and ',' not in route_content.split('MODEL')[1].split('\n')[0]:
            algo_block = '\n  ALGORITHM confidence {\n    threshold: 0.8\n  }\n'
            return dsl[:match.start(2)] + algo_block + match.group(2) + dsl[match.end():]
    return dsl


def mutate_struct_nested_route(dsl: str) -> str:
    """嵌套的 ROUTE（不允许）"""
    route_pattern = r'(ROUTE\s+\w+[^{]*\{[^}]*?)(\n\s*\})'
    match = re.search(route_pattern, dsl, re.DOTALL)
    if match:
        nested_route = '\n  ROUTE nested_invalid {\n    PRIORITY 99\n    MODEL "test:1b"\n  }'
        return dsl[:match.start(2)] + nested_route + match.group(2) + dsl[match.end():]
    return dsl


# ============== 语义/逻辑错误 (新增) ==============

def mutate_semantic_conflicting_routes(dsl: str) -> str:
    """相同条件但不同模型的路由冲突"""
    # 找到 ROUTE 块
    route_pattern = r'(ROUTE\s+(\w+)[^{]*\{[^}]*WHEN\s+([^\n]+)[^}]*MODEL\s+"([^"]+)"[^}]*\})'
    matches = list(re.finditer(route_pattern, dsl, re.DOTALL))
    if matches:
        match = matches[0]
        route_name = match.group(2)
        when_clause = match.group(3)
        model_name = match.group(4)
        # 添加一个冲突路由（相同 WHEN，不同 MODEL，相同 PRIORITY）
        conflict_route = f'\n\nROUTE {route_name}_conflict {{\n  PRIORITY 100\n  WHEN {when_clause}\n  MODEL "conflicting-model:7b"\n}}'
        return dsl + conflict_route
    return dsl


def mutate_semantic_dead_when(dsl: str) -> str:
    """创建永假的 WHEN 条件 (死代码)"""
    # 找到 WHEN 子句，改成 a AND NOT a
    when_pattern = r'(WHEN\s+)(\w+\("[^"]+"\))'
    match = re.search(when_pattern, dsl)
    if match:
        condition = match.group(2)
        # 创建永假条件: condition AND NOT condition
        dead_condition = f'{condition} AND NOT {condition}'
        return dsl[:match.start(2)] + dead_condition + dsl[match.end():]
    return dsl


def mutate_semantic_priority_collision(dsl: str) -> str:
    """多个路由相同优先级（歧义）"""
    # 找到所有 PRIORITY
    priority_pattern = r'PRIORITY\s+(\d+)'
    matches = list(re.finditer(priority_pattern, dsl))
    if len(matches) >= 2:
        # 把第二个优先级改成和第一个一样
        first_priority = matches[0].group(1)
        second_match = matches[1]
        return dsl[:second_match.start(1)] + first_priority + dsl[second_match.end(1):]
    return dsl


def mutate_semantic_unreachable_route(dsl: str) -> str:
    """创建永远不会被匹配到的路由（被更高优先级完全覆盖）"""
    # 找到 ROUTE 块
    route_pattern = r'(ROUTE\s+\w+[^{]*\{[^}]*PRIORITY\s+\d+[^}]*\})'
    match = re.search(route_pattern, dsl, re.DOTALL)
    if match:
        # 添加一个极低优先级的 catch-all 路由
        unreachable = '\n\nROUTE unreachable_route {\n  PRIORITY 1\n  WHEN keyword("*")  # This will never match after catch-all\n  MODEL "unused:1b"\n}'
        # 同时添加一个高优先级的 catch-all
        catchall = '\n\nROUTE catch_all {\n  PRIORITY 999\n  MODEL "default:7b"  # No WHEN = catch all\n}'
        return dsl + catchall + unreachable
    return dsl


def mutate_semantic_circular_plugin(dsl: str) -> str:
    """插件循环引用或自引用"""
    # 找到 PLUGIN 声明
    plugin_pattern = r'(PLUGIN\s+(\w+)\s*\{[^}]*)(})'
    match = re.search(plugin_pattern, dsl, re.DOTALL)
    if match:
        plugin_name = match.group(2)
        # 添加自引用配置
        self_ref = f'\n  depends_on: "{plugin_name}"  # Circular reference'
        return dsl[:match.end(1)] + self_ref + '\n' + match.group(3) + dsl[match.end():]
    return dsl


def mutate_semantic_tautology_when(dsl: str) -> str:
    """创建永真的 WHEN 条件 (总是匹配，可能不是期望行为)"""
    when_pattern = r'(WHEN\s+)(\w+\("[^"]+"\))'
    match = re.search(when_pattern, dsl)
    if match:
        condition = match.group(2)
        # 创建永真条件: condition OR NOT condition
        tautology = f'{condition} OR NOT {condition}'
        return dsl[:match.start(2)] + tautology + dsl[match.end():]
    return dsl


# ============== GLOBAL 配置错误 (新增) ==============

def mutate_global_duplicate(dsl: str) -> str:
    """重复的 GLOBAL 块"""
    global_pattern = r'(GLOBAL\s*\{[^}]*\})'
    match = re.search(global_pattern, dsl, re.DOTALL)
    if match:
        # 添加重复的 GLOBAL 块
        dup_global = '\n\nGLOBAL {\n  default_timeout: 60\n}'
        return dsl + dup_global
    else:
        # 如果没有 GLOBAL，添加两个
        two_globals = 'GLOBAL {\n  default_model: "gpt-4o"\n}\n\nGLOBAL {\n  default_model: "claude-3"\n}\n\n'
        return two_globals + dsl
    return dsl


def mutate_global_invalid_default_model(dsl: str) -> str:
    """GLOBAL 中引用不存在的默认模型"""
    global_pattern = r'(GLOBAL\s*\{[^}]*)(})'
    match = re.search(global_pattern, dsl, re.DOTALL)
    if match:
        invalid_default = '\n  default_model: "nonexistent-model-xyz:999b"'
        return dsl[:match.end(1)] + invalid_default + '\n' + match.group(2) + dsl[match.end():]
    else:
        # 添加带无效默认模型的 GLOBAL
        invalid_global = 'GLOBAL {\n  default_model: "nonexistent-model-xyz:999b"\n}\n\n'
        return invalid_global + dsl
    return dsl


def mutate_global_conflicting_settings(dsl: str) -> str:
    """GLOBAL 中的冲突配置"""
    global_pattern = r'(GLOBAL\s*\{[^}]*)(})'
    match = re.search(global_pattern, dsl, re.DOTALL)
    if match:
        # 添加冲突设置
        conflicts = '\n  timeout: 30\n  timeout: 60  # Duplicate conflicting timeout'
        return dsl[:match.end(1)] + conflicts + '\n' + match.group(2) + dsl[match.end():]
    return dsl


def mutate_global_invalid_field(dsl: str) -> str:
    """GLOBAL 中使用无效字段"""
    global_pattern = r'(GLOBAL\s*\{[^}]*)(})'
    match = re.search(global_pattern, dsl, re.DOTALL)
    if match:
        invalid_field = '\n  not_a_valid_global_field: "something"'
        return dsl[:match.end(1)] + invalid_field + '\n' + match.group(2) + dsl[match.end():]
    else:
        invalid_global = 'GLOBAL {\n  not_a_valid_global_field: "something"\n}\n\n'
        return invalid_global + dsl
    return dsl


# ============== BACKEND 配置错误 (新增) ==============

def mutate_backend_missing_host(dsl: str) -> str:
    """BACKEND 缺少必要的 host 字段"""
    # 找到 BACKEND 块
    backend_pattern = r'(BACKEND\s+(\w+)\s*\{)([^}]*)(})'
    match = re.search(backend_pattern, dsl, re.DOTALL)
    if match:
        content = match.group(3)
        # 删除 host 行
        new_content = re.sub(r'\s*host:\s*"[^"]*"\s*\n?', '\n', content)
        if new_content != content:
            return dsl[:match.start(3)] + new_content + dsl[match.end(3):]
    else:
        # 添加一个缺少 host 的 BACKEND
        bad_backend = '\n\nBACKEND incomplete_backend {\n  port: 8080\n  # Missing required host field\n}'
        return dsl + bad_backend
    return dsl


def mutate_backend_invalid_port(dsl: str) -> str:
    """BACKEND 端口配置无效"""
    backend_pattern = r'(BACKEND\s+\w+\s*\{[^}]*port:\s*)(\d+)([^}]*\})'
    match = re.search(backend_pattern, dsl, re.DOTALL)
    if match:
        # 设置无效端口
        bad_port = random.choice(['0', '-1', '65536', '99999', '"not_a_number"'])
        return dsl[:match.start(2)] + bad_port + dsl[match.end(2):]
    else:
        # 添加带无效端口的 BACKEND
        bad_backend = '\n\nBACKEND bad_port_backend {\n  host: "localhost"\n  port: 99999\n}'
        return dsl + bad_backend
    return dsl


def mutate_backend_ssl_mismatch(dsl: str) -> str:
    """SSL 配置和端口不匹配 (port: 443 but ssl: false)"""
    backend_pattern = r'(BACKEND\s+\w+\s*\{[^}]*)(})'
    match = re.search(backend_pattern, dsl, re.DOTALL)
    if match:
        content = match.group(1)
        # 检查是否有 port: 443，如果有添加 ssl: false
        if 'port: 443' in content or 'port: 8443' in content:
            mismatch = '\n  ssl: false  # Mismatch: HTTPS port but SSL disabled'
            return dsl[:match.end(1)] + mismatch + '\n' + match.group(2) + dsl[match.end():]
        else:
            # 添加不匹配的配置
            mismatch = '\n  port: 443\n  ssl: false  # Mismatch'
            return dsl[:match.end(1)] + mismatch + '\n' + match.group(2) + dsl[match.end():]
    else:
        bad_backend = '\n\nBACKEND ssl_mismatch {\n  host: "api.example.com"\n  port: 443\n  ssl: false\n}'
        return dsl + bad_backend
    return dsl


def mutate_backend_duplicate_name(dsl: str) -> str:
    """重复的 BACKEND 名称"""
    backend_pattern = r'BACKEND\s+(\w+)'
    matches = list(re.finditer(backend_pattern, dsl))
    if matches:
        first_name = matches[0].group(1)
        # 添加同名 BACKEND
        dup_backend = f'\n\nBACKEND {first_name} {{\n  host: "duplicate.example.com"\n  port: 8080\n}}'
        return dsl + dup_backend
    return dsl


def mutate_backend_invalid_timeout(dsl: str) -> str:
    """BACKEND 超时配置无效"""
    backend_pattern = r'(BACKEND\s+\w+\s*\{[^}]*)(})'
    match = re.search(backend_pattern, dsl, re.DOTALL)
    if match:
        bad_timeout = random.choice([
            '\n  timeout: -1',
            '\n  timeout: "slow"',
            '\n  timeout: 0',
            '\n  connect_timeout: -100',
        ])
        return dsl[:match.end(1)] + bad_timeout + '\n' + match.group(2) + dsl[match.end():]
    else:
        bad_backend = '\n\nBACKEND bad_timeout {\n  host: "api.example.com"\n  port: 8080\n  timeout: -1\n}'
        return dsl + bad_backend
    return dsl


# ============== 编码/格式错误 (新增) ==============

def mutate_encoding_mixed_indent(dsl: str) -> str:
    """混合使用 tab 和 space 缩进"""
    # 找到用空格缩进的行，替换部分为 tab
    lines = dsl.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if line.startswith('  ') and random.random() < 0.3:
            # 将空格替换为 tab
            new_lines.append('\t' + line.lstrip())
        else:
            new_lines.append(line)
    result = '\n'.join(new_lines)
    return result if result != dsl else dsl


def mutate_encoding_chinese_colon(dsl: str) -> str:
    """使用中文冒号代替英文冒号"""
    # 找到 field: value 模式，替换冒号
    colon_pattern = r'(\s+\w+):\s*'
    matches = list(re.finditer(colon_pattern, dsl))
    if matches:
        match = random.choice(matches)
        # 使用中文冒号
        return dsl[:match.end(1)] + '：' + dsl[match.end(1)+1:]
    return dsl


def mutate_encoding_chinese_quotes(dsl: str) -> str:
    """使用中文引号代替英文引号"""
    # 找到字符串，替换引号
    string_pattern = r'"([^"]*)"'
    matches = list(re.finditer(string_pattern, dsl))
    if matches:
        match = random.choice(matches)
        content = match.group(1)
        # 使用中文引号
        return dsl[:match.start()] + f'"{content}"' + dsl[match.end():]
    return dsl


def mutate_encoding_invisible_char(dsl: str) -> str:
    """插入不可见字符 (零宽字符)"""
    # 在关键字中插入零宽空格
    keywords = ['SIGNAL', 'ROUTE', 'PLUGIN', 'BACKEND', 'MODEL', 'WHEN', 'PRIORITY']
    for kw in keywords:
        if kw in dsl:
            # 在关键字中间插入零宽空格 U+200B
            pos = len(kw) // 2
            corrupted = kw[:pos] + '\u200b' + kw[pos:]
            return dsl.replace(kw, corrupted, 1)
    return dsl


def mutate_encoding_trailing_whitespace(dsl: str) -> str:
    """添加行尾空白字符"""
    lines = dsl.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if line.strip() and random.random() < 0.3:
            # 添加随机数量的尾部空格
            trailing = ' ' * random.randint(1, 5)
            new_lines.append(line + trailing)
        else:
            new_lines.append(line)
    result = '\n'.join(new_lines)
    return result if result != dsl else dsl


def mutate_encoding_bom(dsl: str) -> str:
    """在文件开头添加 BOM"""
    # 添加 UTF-8 BOM
    return '\ufeff' + dsl


def _extract_signal_refs(expr: str) -> list[str]:
    """提取 WHEN 表达式中的原子信号引用。"""
    return re.findall(r'\w+\("[^"]+"\)', expr)


def _extract_defined_signal_refs(dsl: str) -> list[str]:
    """提取 DSL 中已定义的信号引用形式。"""
    refs = []
    for match in re.finditer(r'SIGNAL\s+(\w+)\s+(\w+)\s*\{', dsl):
        refs.append(f'{match.group(1)}("{match.group(2)}")')
    return refs


def mutate_intent_wrong_model_binding(dsl: str) -> str:
    """将模型绑定替换为另一个模型。"""
    matches = list(re.finditer(r'(MODEL\s+)"([^"]+)"', dsl))
    if not matches:
        return dsl

    match = random.choice(matches)
    current_model = match.group(2)
    alternatives = [m for m in MODEL_CANDIDATES if m != current_model]
    if not alternatives:
        return dsl

    replacement = random.choice(alternatives)
    return dsl[:match.start(2)] + replacement + dsl[match.end(2):]


def mutate_intent_missing_condition(dsl: str) -> str:
    """删掉或替换部分条件，制造意图缺失。"""
    when_pattern = r'(WHEN\s+)(.+?)(\n\s*(?:MODEL|PLUGIN|ALGORITHM|$))'
    match = re.search(when_pattern, dsl, re.DOTALL)
    if not match:
        return dsl

    original_expr = match.group(2).strip()
    refs = _extract_signal_refs(original_expr)
    if len(refs) >= 2:
        new_expr = refs[0]
    else:
        defined_refs = [ref for ref in _extract_defined_signal_refs(dsl) if ref not in refs]
        if not defined_refs:
            return dsl
        new_expr = random.choice(defined_refs)

    return dsl[:match.start(2)] + new_expr + dsl[match.end(2):]


def mutate_intent_wrong_priority(dsl: str) -> str:
    """交换优先级，或把优先级显著降低。"""
    matches = list(re.finditer(r'(PRIORITY\s+)(-?\d+)', dsl))
    if len(matches) >= 2:
        first, second = matches[0], matches[1]
        first_value = first.group(2)
        second_value = second.group(2)
        mutated = dsl[:first.start(2)] + second_value + dsl[first.end(2):]
        offset = len(second_value) - len(first_value)
        second_start = second.start(2) + offset
        second_end = second.end(2) + offset
        return mutated[:second_start] + first_value + mutated[second_end:]

    if matches:
        match = matches[0]
        lowered = str(max(1, int(match.group(2)) // 10))
        if lowered == match.group(2):
            lowered = "1"
        return dsl[:match.start(2)] + lowered + dsl[match.end(2):]

    return dsl


def mutate_intent_missing_fallback(dsl: str) -> str:
    """删除最低优先级路由或默认模型配置。"""
    route_pattern = r'(ROUTE\s+\w+[^{]*\{[^}]*\}\n*)'
    routes = list(re.finditer(route_pattern, dsl, re.DOTALL))
    if len(routes) >= 2:
        target = routes[-1]
        return dsl[:target.start()] + dsl[target.end():]

    default_model_pattern = r'(\n?\s*default_model:\s*"[^"]+"\s*\n?)'
    match = re.search(default_model_pattern, dsl)
    if match:
        return dsl[:match.start()] + dsl[match.end():]

    return dsl


def mutate_intent_wrong_reasoning_attr(dsl: str) -> str:
    """翻转 reasoning 相关属性。"""
    reasoning_pattern = r'(reasoning\s*=\s*)(true|false)'
    match = re.search(reasoning_pattern, dsl)
    if match:
        flipped = 'false' if match.group(2) == 'true' else 'true'
        return dsl[:match.start(2)] + flipped + dsl[match.end(2):]

    effort_pattern = r'(effort\s*=\s*)"([^"]+)"'
    match = re.search(effort_pattern, dsl)
    if match:
        options = ['low', 'medium', 'high']
        replacement = random.choice([opt for opt in options if opt != match.group(2)])
        return dsl[:match.start(2)] + replacement + dsl[match.end(2):]

    model_pattern = r'(MODEL\s+"[^"]+")'
    match = re.search(model_pattern, dsl)
    if match:
        return dsl[:match.end(1)] + ' (reasoning = false)' + dsl[match.end(1):]

    return dsl


def mutate_near_miss_drop_route(dsl: str) -> str:
    """删除一个非主要路由，制造近似正确但不完整的配置。"""
    route_pattern = r'(ROUTE\s+\w+[^{]*\{[^}]*\}\n*)'
    routes = list(re.finditer(route_pattern, dsl, re.DOTALL))
    if len(routes) < 2:
        return dsl

    target = random.choice(routes[1:])
    return dsl[:target.start()] + dsl[target.end():]


def mutate_near_miss_over_scope(dsl: str) -> str:
    """去掉布尔守卫，使路由匹配范围过宽。"""
    when_pattern = r'(WHEN\s+)(.+?)(\n\s*(?:MODEL|PLUGIN|ALGORITHM|$))'
    match = re.search(when_pattern, dsl, re.DOTALL)
    if not match:
        return dsl

    expr = match.group(2).strip()
    refs = _extract_signal_refs(expr)
    if refs:
        return dsl[:match.start(2)] + refs[0] + dsl[match.end(2):]

    if expr.startswith('NOT '):
        return dsl[:match.start(2)] + expr[4:] + dsl[match.end(2):]

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
    'bool_expr_error': [
        MutationType('paren_mismatch', 'Mismatched parentheses in boolean expression', mutate_bool_paren_mismatch),
        MutationType('missing_operand', 'Missing operand after AND/OR operator', mutate_bool_missing_operand),
        MutationType('double_operator', 'Duplicate boolean operator (AND AND)', mutate_bool_double_operator),
        MutationType('invalid_nesting', 'Invalid nesting (NOT NOT, AND OR)', mutate_bool_invalid_nesting),
        MutationType('wrong_precedence', 'Missing parentheses causing precedence issues', mutate_bool_wrong_precedence),
        MutationType('empty_expr', 'Empty WHEN expression', mutate_bool_empty_expr),
    ],
    'model_error': [
        MutationType('trailing_comma', 'Trailing comma after last model', mutate_model_trailing_comma),
        MutationType('duplicate_attr', 'Duplicate model attribute', mutate_model_duplicate_attr),
        MutationType('invalid_attr', 'Invalid model attribute name', mutate_model_invalid_attr),
        MutationType('empty_name', 'Empty model name string', mutate_model_empty_name),
        MutationType('missing_quotes', 'Model name without quotes', mutate_model_missing_quotes),
    ],
    'structural_error': [
        MutationType('route_missing_model', 'ROUTE without MODEL declaration', mutate_struct_route_missing_model),
        MutationType('route_missing_priority', 'ROUTE without PRIORITY', mutate_struct_route_missing_priority),
        MutationType('signal_after_route', 'SIGNAL declared after ROUTE (wrong order)', mutate_struct_signal_after_route),
        MutationType('duplicate_route_name', 'Duplicate route names', mutate_struct_duplicate_route_name),
        MutationType('algorithm_single_model', 'ALGORITHM with single MODEL (semantic error)', mutate_struct_algorithm_single_model),
        MutationType('nested_route', 'Nested ROUTE declaration (not allowed)', mutate_struct_nested_route),
    ],
    # ============== 新增变异类别 ==============
    'semantic_error': [
        MutationType('conflicting_routes', 'Routes with same condition but different models', mutate_semantic_conflicting_routes),
        MutationType('dead_when', 'WHEN condition that is always false (dead code)', mutate_semantic_dead_when),
        MutationType('priority_collision', 'Multiple routes with identical priority', mutate_semantic_priority_collision),
        MutationType('unreachable_route', 'Route that can never be matched', mutate_semantic_unreachable_route),
        MutationType('circular_plugin', 'Plugin with circular/self reference', mutate_semantic_circular_plugin),
        MutationType('tautology_when', 'WHEN condition that is always true', mutate_semantic_tautology_when),
    ],
    'global_error': [
        MutationType('duplicate_global', 'Multiple GLOBAL blocks', mutate_global_duplicate),
        MutationType('invalid_default_model', 'GLOBAL references non-existent default model', mutate_global_invalid_default_model),
        MutationType('conflicting_settings', 'Conflicting settings in GLOBAL', mutate_global_conflicting_settings),
        MutationType('invalid_field', 'Invalid field in GLOBAL block', mutate_global_invalid_field),
    ],
    'backend_error': [
        MutationType('missing_host', 'BACKEND without required host field', mutate_backend_missing_host),
        MutationType('invalid_port', 'BACKEND with invalid port value', mutate_backend_invalid_port),
        MutationType('ssl_mismatch', 'SSL config mismatched with port (443 but ssl:false)', mutate_backend_ssl_mismatch),
        MutationType('duplicate_name', 'Duplicate BACKEND names', mutate_backend_duplicate_name),
        MutationType('invalid_timeout', 'BACKEND with invalid timeout value', mutate_backend_invalid_timeout),
    ],
    'encoding_error': [
        MutationType('mixed_indent', 'Mixed tabs and spaces indentation', mutate_encoding_mixed_indent),
        MutationType('chinese_colon', 'Chinese colon instead of ASCII colon', mutate_encoding_chinese_colon),
        MutationType('chinese_quotes', 'Chinese quotes instead of ASCII quotes', mutate_encoding_chinese_quotes),
        MutationType('invisible_char', 'Invisible zero-width characters in keywords', mutate_encoding_invisible_char),
        MutationType('trailing_whitespace', 'Trailing whitespace in lines', mutate_encoding_trailing_whitespace),
        MutationType('bom', 'UTF-8 BOM at file start', mutate_encoding_bom),
    ],
    'intent_mismatch': [
        MutationType('wrong_model_binding', 'Route binds to the wrong model for the original intent', mutate_intent_wrong_model_binding),
        MutationType('missing_condition', 'Critical routing condition removed or replaced', mutate_intent_missing_condition),
        MutationType('wrong_priority', 'Route priority changed so intent ordering is wrong', mutate_intent_wrong_priority),
        MutationType('missing_fallback', 'Fallback route or default model removed', mutate_intent_missing_fallback),
        MutationType('wrong_reasoning_attr', 'Reasoning-related model attributes are flipped', mutate_intent_wrong_reasoning_attr),
    ],
    'near_miss': [
        MutationType('drop_route', 'Configuration is almost correct but missing one route', mutate_near_miss_drop_route),
        MutationType('over_scope', 'Route condition broadened so it matches too much traffic', mutate_near_miss_over_scope),
        MutationType('wrong_priority', 'Near miss caused by route priority drift', mutate_intent_wrong_priority),
        MutationType('wrong_reasoning_attr', 'Near miss caused by reasoning attribute mismatch', mutate_intent_wrong_reasoning_attr),
    ],
}


def classify_negative_category(mutation_category: str) -> tuple[str, str]:
    """Map mutation categories to broad error classes and difficulty."""
    if mutation_category == 'intent_mismatch':
        return 'intent_mismatch', 'hard'
    if mutation_category == 'near_miss':
        return 'near_miss', 'hard'
    if mutation_category in {'syntax_error', 'encoding_error'}:
        return 'legality', 'easy'
    return 'semantic', 'medium'


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
    error_class, difficulty = classify_negative_category(mutation_category)
    
    return {
        'id': f"neg_{mutation_category}_{mutation.name}_{dsl_hash}",
        'dsl': mutated_dsl,
        'original_id': sample.get('id'),
        'mutation_category': mutation_category,
        'mutation_type': mutation.name,
        'mutation_description': mutation.description,
        'error_class': error_class,
        'difficulty': difficulty,
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
