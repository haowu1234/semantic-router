#!/usr/bin/env python3
"""
DSL CFG 随机游走生成器

基于 BNF 语法规则随机生成语法正确的 DSL 配置，用于训练数据扩充。

核心特性:
1. 保证每个生成的 DSL 都语法有效
2. 支持按复杂度级别 (L1-L5) 控制生成
3. 信号引用完整性保证 (WHEN 只引用已定义的信号)
4. 字段值符合 Schema 约束

用法:
    python cfg_generator.py --count 10000 --output synthetic/
    python cfg_generator.py --count 1000 --complexity L3 --output synthetic/L3_only.jsonl
"""

import argparse
import json
import random
import hashlib
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field


# ============== 类型定义 ==============

SIGNAL_TYPES = [
    'keyword', 'embedding', 'domain', 'fact_check', 'user_feedback',
    'preference', 'language', 'context', 'complexity', 'modality',
    'authz', 'jailbreak', 'pii'
]

PLUGIN_TYPES = [
    'semantic_cache', 'memory', 'system_prompt', 'header_mutation',
    'hallucination', 'router_replay', 'rag', 'image_gen', 'fast_response'
]

ALGORITHM_TYPES = [
    'confidence', 'ratings', 'remom', 'static', 'elo', 'router_dc',
    'automix', 'hybrid', 'rl_driven', 'gmtrouter', 'latency_aware',
    'knn', 'kmeans', 'svm'
]

BACKEND_TYPES = [
    'vllm_endpoint', 'provider_profile', 'embedding_model',
    'semantic_cache', 'memory', 'response_api', 'vector_store', 'image_gen_backend'
]

MODEL_NAMES = [
    'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo',
    'deepseek-r1', 'deepseek-v3', 'deepseek-coder',
    'qwen2.5:3b', 'qwen2.5:7b', 'qwen2.5:14b', 'qwen2.5:32b', 'qwen2.5:72b',
    'qwen3:8b', 'qwen3:14b', 'qwen3:32b', 'qwen3:70b',
    'claude-3-sonnet', 'claude-3-opus', 'claude-3-haiku',
    'llama3.1:8b', 'llama3.1:70b', 'llama3.2:3b',
    'gemini-2.0-flash', 'gemini-1.5-pro',
    'mistral-large', 'mixtral-8x7b'
]

KEYWORD_EXAMPLES = [
    ['urgent', 'asap', 'emergency', 'critical'],
    ['code', 'programming', 'developer', 'software'],
    ['math', 'calculate', 'compute', 'equation'],
    ['translate', 'language', 'chinese', 'english'],
    ['summarize', 'summary', 'tldr', 'brief'],
    ['explain', 'clarify', 'elaborate', 'detail'],
    ['compare', 'versus', 'difference', 'similarity'],
    ['review', 'check', 'verify', 'validate'],
]

EMBEDDING_CANDIDATES = [
    ['machine learning', 'deep learning', 'neural network', 'AI'],
    ['code optimization', 'performance tuning', 'refactoring'],
    ['data analysis', 'statistics', 'visualization'],
    ['natural language processing', 'text analysis', 'sentiment'],
    ['computer vision', 'image recognition', 'object detection'],
    ['web development', 'frontend', 'backend', 'fullstack'],
]

DOMAIN_DESCRIPTIONS = {
    'math': ('Mathematics and quantitative reasoning', ['math']),
    'physics': ('Physics and physical sciences', ['physics']),
    'chemistry': ('Chemistry and chemical sciences', ['chemistry']),
    'biology': ('Biology and life sciences', ['biology']),
    'computer_science': ('Computer science and programming', ['computer_science']),
    'medicine': ('Medicine and healthcare', ['medicine']),
    'law': ('Legal documents and contracts', ['law']),
    'economics': ('Economics and finance', ['economics']),
    'history': ('History and historical events', ['history']),
    'philosophy': ('Philosophy and ethics', ['philosophy']),
    'general': ('General knowledge and common sense', ['other']),
}


# ============== 信号类型 Schema ==============

@dataclass
class SignalSchema:
    """信号类型的字段 Schema"""
    required: dict = field(default_factory=dict)
    optional: dict = field(default_factory=dict)


SIGNAL_SCHEMAS: dict[str, SignalSchema] = {
    'keyword': SignalSchema(
        required={
            'operator': ['any', 'all'],
            'keywords': 'keyword_array',
        },
        optional={
            'method': ['regex', 'bm25', 'ngram'],
            'case_sensitive': 'bool',
            'fuzzy_match': 'bool',
            'fuzzy_threshold': (1, 5),
        }
    ),
    'embedding': SignalSchema(
        required={
            'threshold': (0.5, 0.95),
            'candidates': 'embedding_array',
        },
        optional={
            'aggregation_method': ['mean', 'max', 'any'],
        }
    ),
    'domain': SignalSchema(
        required={'description': 'domain_desc'},
        optional={'mmlu_categories': 'mmlu_array'}
    ),
    'fact_check': SignalSchema(required={'description': 'string'}),
    'user_feedback': SignalSchema(required={'description': 'string'}),
    'preference': SignalSchema(required={'description': 'string'}),
    'language': SignalSchema(optional={'description': 'string'}),
    'context': SignalSchema(
        required={
            'min_tokens': ['1K', '2K', '4K', '8K'],
            'max_tokens': ['8K', '16K', '32K', '64K', '128K'],
        },
        optional={'description': 'string'}
    ),
    'complexity': SignalSchema(
        required={'threshold': (0.1, 0.9)},
        optional={
            'hard': 'complexity_examples',
            'easy': 'complexity_examples',
            'description': 'string',
        }
    ),
    'modality': SignalSchema(optional={'description': 'string'}),
    'authz': SignalSchema(
        required={
            'role': 'role_name',
            'subjects': 'subjects_array',
        },
        optional={'description': 'string'}
    ),
    'jailbreak': SignalSchema(
        required={'threshold': (0.7, 0.99)},
        optional={
            'method': ['classifier', 'contrastive'],
            'include_history': 'bool',
        }
    ),
    'pii': SignalSchema(
        required={'threshold': (0.7, 0.95)},
        optional={
            'pii_types_allowed': 'pii_types_array',
            'include_history': 'bool',
        }
    ),
}

PLUGIN_SCHEMAS: dict[str, SignalSchema] = {
    'semantic_cache': SignalSchema(
        optional={
            'enabled': 'bool',
            'similarity_threshold': (0.8, 0.99),
        }
    ),
    'memory': SignalSchema(
        optional={
            'enabled': 'bool',
            'retrieval_limit': (3, 20),
            'similarity_threshold': (0.6, 0.9),
            'auto_store': 'bool',
        }
    ),
    'system_prompt': SignalSchema(
        required={'system_prompt': 'system_prompt_text'},
        optional={
            'enabled': 'bool',
            'mode': ['replace', 'insert'],
        }
    ),
    'hallucination': SignalSchema(
        optional={
            'enabled': 'bool',
            'use_nli': 'bool',
            'hallucination_action': ['header', 'body', 'none'],
        }
    ),
    'router_replay': SignalSchema(
        optional={
            'enabled': 'bool',
            'max_records': (1000, 50000),
            'capture_request_body': 'bool',
            'capture_response_body': 'bool',
        }
    ),
    'rag': SignalSchema(
        required={'backend': 'backend_name'},
        optional={
            'enabled': 'bool',
            'top_k': (3, 20),
            'similarity_threshold': (0.5, 0.9),
            'injection_mode': ['system', 'user', 'context'],
            'on_failure': ['skip', 'fail'],
        }
    ),
    'image_gen': SignalSchema(
        required={'backend': 'backend_name'},
        optional={'enabled': 'bool'}
    ),
    'fast_response': SignalSchema(
        required={'message': 'fast_response_message'}
    ),
    'header_mutation': SignalSchema(
        optional={
            'add': 'header_list',
            'delete': 'string_array',
        }
    ),
}

ALGORITHM_SCHEMAS: dict[str, SignalSchema] = {
    'confidence': SignalSchema(
        optional={
            'confidence_method': ['avg_logprob', 'margin', 'hybrid', 'self_verify'],
            'threshold': (-2.0, 0.8),
            'on_error': ['skip', 'fail'],
            'escalation_order': ['size', 'cost', 'automix'],
        }
    ),
    'ratings': SignalSchema(
        optional={
            'max_concurrent': (2, 10),
            'on_error': ['skip', 'fail'],
        }
    ),
    'remom': SignalSchema(
        required={'breadth_schedule': 'breadth_schedule'},
        optional={
            'model_distribution': ['weighted', 'equal', 'first_only'],
            'temperature': (0.5, 1.5),
            'include_reasoning': 'bool',
            'on_error': ['skip', 'fail'],
        }
    ),
    'static': SignalSchema(),
    'elo': SignalSchema(
        optional={
            'initial_rating': (1200, 1800),
            'k_factor': (16, 64),
            'category_weighted': 'bool',
        }
    ),
    'router_dc': SignalSchema(
        optional={
            'temperature': (0.05, 0.2),
            'dimension_size': [384, 768, 1024],
            'min_similarity': (0.2, 0.5),
        }
    ),
    'automix': SignalSchema(
        optional={
            'verification_threshold': (0.5, 0.9),
            'max_escalations': (1, 4),
            'cost_aware_routing': 'bool',
        }
    ),
    'hybrid': SignalSchema(
        optional={
            'elo_weight': (0.1, 0.5),
            'router_dc_weight': (0.1, 0.5),
            'automix_weight': (0.1, 0.4),
            'cost_weight': (0.1, 0.3),
        }
    ),
    'rl_driven': SignalSchema(
        optional={
            'exploration_rate': (0.1, 0.5),
            'use_thompson_sampling': 'bool',
            'enable_personalization': 'bool',
        }
    ),
    'gmtrouter': SignalSchema(
        optional={
            'enable_personalization': 'bool',
            'history_sample_size': (3, 10),
        }
    ),
    'latency_aware': SignalSchema(
        required={
            'tpot_percentile': (10, 50),
            'ttft_percentile': (10, 50),
        }
    ),
    'knn': SignalSchema(),
    'kmeans': SignalSchema(),
    'svm': SignalSchema(),
}


# ============== 复杂度配置 ==============

@dataclass
class ComplexityConfig:
    """复杂度级别配置"""
    min_signals: int
    max_signals: int
    min_routes: int
    max_routes: int
    max_condition_depth: int
    plugin_prob: float
    backend_prob: float
    algorithm_prob: float
    multi_model_prob: float


COMPLEXITY_CONFIGS: dict[str, ComplexityConfig] = {
    'L1': ComplexityConfig(1, 2, 1, 1, 0, 0.1, 0.2, 0.0, 0.0),
    'L2': ComplexityConfig(2, 4, 1, 2, 1, 0.3, 0.3, 0.2, 0.1),
    'L3': ComplexityConfig(3, 6, 2, 3, 2, 0.5, 0.5, 0.4, 0.3),
    'L4': ComplexityConfig(4, 8, 2, 4, 3, 0.7, 0.7, 0.6, 0.5),
    'L5': ComplexityConfig(5, 12, 3, 5, 4, 0.9, 0.9, 0.8, 0.7),
}


# ============== 生成器类 ==============

class DSLGenerator:
    """DSL 配置生成器"""
    
    def __init__(self, complexity: str = 'L3', seed: int | None = None):
        self.config = COMPLEXITY_CONFIGS[complexity]
        self.complexity = complexity
        if seed is not None:
            random.seed(seed)
        
        # 当前配置状态
        self._signals: dict[str, list[str]] = {}  # signal_type -> [names]
        self._plugins: list[str] = []  # plugin template names
        self._backends: dict[str, list[str]] = {}  # backend_type -> [names]
        self._counter = 0
    
    def generate(self) -> str:
        """生成一个完整的 DSL 配置"""
        self._signals = {}
        self._plugins = []
        self._backends = {}
        self._counter += 1
        
        parts = []
        
        # 1. 生成信号
        num_signals = random.randint(self.config.min_signals, self.config.max_signals)
        for _ in range(num_signals):
            parts.append(self._gen_signal())
        
        # 2. 生成插件模板
        if random.random() < self.config.plugin_prob:
            num_plugins = random.randint(1, 2)
            for _ in range(num_plugins):
                parts.append(self._gen_plugin_template())
        
        # 3. 生成路由
        num_routes = random.randint(self.config.min_routes, self.config.max_routes)
        for i in range(num_routes):
            parts.append(self._gen_route(i))
        
        # 4. 生成后端
        if random.random() < self.config.backend_prob:
            parts.append(self._gen_backend())
        
        # 5. 生成全局配置
        if random.random() > 0.3:
            parts.append(self._gen_global())
        
        return '\n\n'.join(parts)
    
    def _gen_signal(self) -> str:
        """生成信号声明"""
        # 选择信号类型（确保多样性）
        available_types = [t for t in SIGNAL_TYPES if t not in self._signals or len(self._signals[t]) < 2]
        if not available_types:
            available_types = SIGNAL_TYPES
        sig_type = random.choice(available_types)
        
        # 生成名称
        name = f'{sig_type}_{random.randint(1, 99)}'
        self._signals.setdefault(sig_type, []).append(name)
        
        # 生成字段
        fields = self._gen_fields(sig_type, SIGNAL_SCHEMAS.get(sig_type, SignalSchema()))
        
        return f'SIGNAL {sig_type} {name} {{\n{fields}\n}}'
    
    def _gen_fields(self, entity_type: str, schema: SignalSchema) -> str:
        """根据 Schema 生成字段"""
        lines = []
        
        # 必填字段
        for field_name, spec in schema.required.items():
            value = self._gen_value(spec, entity_type, field_name)
            lines.append(f'  {field_name}: {value}')
        
        # 可选字段（随机选择一部分）
        for field_name, spec in schema.optional.items():
            if random.random() > 0.5:
                value = self._gen_value(spec, entity_type, field_name)
                lines.append(f'  {field_name}: {value}')
        
        return '\n'.join(lines)
    
    def _gen_value(self, spec: Any, entity_type: str = '', field_name: str = '') -> str:
        """根据 spec 生成值"""
        if isinstance(spec, list):
            # 枚举类型
            return f'"{random.choice(spec)}"'
        
        if isinstance(spec, tuple) and len(spec) == 2:
            # 数值范围
            if isinstance(spec[0], float):
                return f'{random.uniform(spec[0], spec[1]):.2f}'
            else:
                return str(random.randint(spec[0], spec[1]))
        
        if spec == 'bool':
            return random.choice(['true', 'false'])
        
        if spec == 'string':
            return f'"Auto-generated {entity_type} {field_name}"'
        
        if spec == 'keyword_array':
            keywords = random.choice(KEYWORD_EXAMPLES)
            selected = random.sample(keywords, min(len(keywords), random.randint(2, 4)))
            return '[' + ', '.join(f'"{k}"' for k in selected) + ']'
        
        if spec == 'embedding_array':
            candidates = random.choice(EMBEDDING_CANDIDATES)
            selected = random.sample(candidates, min(len(candidates), random.randint(2, 4)))
            return '[' + ', '.join(f'"{c}"' for c in selected) + ']'
        
        if spec == 'domain_desc':
            domain = random.choice(list(DOMAIN_DESCRIPTIONS.keys()))
            return f'"{DOMAIN_DESCRIPTIONS[domain][0]}"'
        
        if spec == 'mmlu_array':
            domain = random.choice(list(DOMAIN_DESCRIPTIONS.keys()))
            cats = DOMAIN_DESCRIPTIONS[domain][1]
            return '[' + ', '.join(f'"{c}"' for c in cats) + ']'
        
        if spec == 'role_name':
            roles = ['admin', 'premium_tier', 'basic_tier', 'developer', 'viewer']
            return f'"{random.choice(roles)}"'
        
        if spec == 'subjects_array':
            kinds = ['User', 'Group', 'ServiceAccount']
            names = ['admin@example.com', 'developers', 'premium-users', 'ci-bot']
            subj = {'kind': random.choice(kinds), 'name': random.choice(names)}
            return f'[{{ kind: "{subj["kind"]}", name: "{subj["name"]}" }}]'
        
        if spec == 'pii_types_array':
            types = ['EMAIL_ADDRESS', 'PHONE_NUMBER', 'CREDIT_CARD', 'IP_ADDRESS']
            selected = random.sample(types, random.randint(0, 2))
            return '[' + ', '.join(f'"{t}"' for t in selected) + ']'
        
        if spec == 'complexity_examples':
            examples = ['hard task example', 'complex problem', 'advanced scenario']
            selected = random.sample(examples, random.randint(1, 2))
            return '{ candidates: [' + ', '.join(f'"{e}"' for e in selected) + '] }'
        
        if spec == 'system_prompt_text':
            prompts = [
                'You are a helpful assistant.',
                'You are an expert in this domain.',
                'Respond concisely and accurately.',
                'You must respond in JSON format.',
                'Think step by step before answering.',
            ]
            return f'"{random.choice(prompts)}"'
        
        if spec == 'fast_response_message':
            messages = [
                'Request blocked for safety reasons.',
                'I cannot help with that request.',
                'This type of content is not allowed.',
                'Please rephrase your question.',
            ]
            return f'"{random.choice(messages)}"'
        
        if spec == 'backend_name':
            names = ['my_vector_store', 'knowledge_base', 'main_backend', 'rag_backend']
            return f'"{random.choice(names)}"'
        
        if spec == 'breadth_schedule':
            schedules = [[8], [8, 2], [16, 4], [32], [4, 2, 1]]
            return str(random.choice(schedules))
        
        if spec == 'header_list':
            return '[{ name: "X-Custom-Header", value: "custom-value" }]'
        
        if spec == 'string_array':
            return '["item1", "item2"]'
        
        return '"default"'
    
    def _gen_plugin_template(self) -> str:
        """生成插件模板声明"""
        plugin_type = random.choice(PLUGIN_TYPES)
        name = f'tpl_{plugin_type}_{random.randint(1, 99)}'
        self._plugins.append(name)
        
        schema = PLUGIN_SCHEMAS.get(plugin_type, SignalSchema())
        fields = self._gen_fields(plugin_type, schema)
        
        return f'PLUGIN {name} {plugin_type} {{\n{fields}\n}}'
    
    def _gen_route(self, index: int) -> str:
        """生成路由声明"""
        name = f'route_{index + 1}'
        priority = random.choice([10, 50, 100, 200, 500, 1000])
        
        lines = [f'ROUTE {name} (description = "{name} route") {{']
        lines.append(f'  PRIORITY {priority}')
        lines.append('')
        
        # WHEN 子句
        if self._signals:
            when_expr = self._gen_bool_expr(0)
            lines.append(f'  WHEN {when_expr}')
            lines.append('')
        
        # MODEL 子句
        num_models = 1
        if random.random() < self.config.multi_model_prob:
            num_models = random.randint(2, 3)
        
        models = random.sample(MODEL_NAMES, min(num_models, len(MODEL_NAMES)))
        model_strs = []
        for i, m in enumerate(models):
            attrs = []
            if random.random() > 0.3:
                attrs.append(f'reasoning = {random.choice(["true", "false"])}')
            if random.random() > 0.5:
                attrs.append(f'effort = "{random.choice(["low", "medium", "high"])}"')
            if i > 0 and random.random() > 0.6:
                attrs.append(f'weight = {random.randint(1, 10)}')
            
            attr_str = f' ({", ".join(attrs)})' if attrs else ''
            model_strs.append(f'"{m}"{attr_str}')
        
        if len(model_strs) == 1:
            lines.append(f'  MODEL {model_strs[0]}')
        else:
            lines.append(f'  MODEL {model_strs[0]},')
            for ms in model_strs[1:-1]:
                lines.append(f'        {ms},')
            lines.append(f'        {model_strs[-1]}')
        lines.append('')
        
        # ALGORITHM 子句
        if random.random() < self.config.algorithm_prob and num_models > 1:
            algo_type = random.choice(ALGORITHM_TYPES)
            schema = ALGORITHM_SCHEMAS.get(algo_type, SignalSchema())
            algo_fields = self._gen_fields(algo_type, schema)
            if algo_fields.strip():
                lines.append(f'  ALGORITHM {algo_type} {{')
                lines.append(algo_fields)
                lines.append('  }')
            else:
                lines.append(f'  ALGORITHM {algo_type} {{}}')
            lines.append('')
        
        # PLUGIN 引用
        if self._plugins and random.random() > 0.5:
            plugin = random.choice(self._plugins)
            lines.append(f'  PLUGIN {plugin}')
        elif random.random() > 0.7:
            # 内联插件
            inline_type = random.choice(['system_prompt', 'semantic_cache', 'fast_response'])
            schema = PLUGIN_SCHEMAS.get(inline_type, SignalSchema())
            plugin_fields = self._gen_fields(inline_type, schema)
            lines.append(f'  PLUGIN {inline_type} {{')
            lines.append(plugin_fields)
            lines.append('  }')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def _gen_bool_expr(self, depth: int) -> str:
        """生成布尔表达式"""
        # 到达最大深度或随机终止
        if depth >= self.config.max_condition_depth or random.random() > 0.6:
            return self._gen_signal_ref()
        
        op = random.choice(['AND', 'OR', 'NOT'])
        
        if op == 'NOT':
            inner = self._gen_bool_expr(depth + 1)
            return f'NOT {inner}' if ' ' not in inner else f'NOT ({inner})'
        
        left = self._gen_bool_expr(depth + 1)
        right = self._gen_bool_expr(depth + 1)
        
        if op == 'OR':
            return f'({left} OR {right})'
        return f'{left} AND {right}'
    
    def _gen_signal_ref(self) -> str:
        """生成信号引用"""
        if not self._signals:
            return 'domain("fallback")'
        
        sig_type = random.choice(list(self._signals.keys()))
        sig_name = random.choice(self._signals[sig_type])
        return f'{sig_type}("{sig_name}")'
    
    def _gen_backend(self) -> str:
        """生成后端声明"""
        backend_type = random.choice(BACKEND_TYPES)
        name = f'{backend_type.split("_")[0]}_{random.randint(1, 99)}'
        self._backends.setdefault(backend_type, []).append(name)
        
        lines = [f'BACKEND {backend_type} {name} {{']
        
        if backend_type == 'vllm_endpoint':
            lines.append(f'  address: "127.0.0.1"')
            lines.append(f'  port: {random.choice([8080, 11434, 8000, 5000])}')
            lines.append(f'  weight: {random.randint(1, 10)}')
            lines.append(f'  type: "{random.choice(["vllm", "ollama", "openai"])}"')
        elif backend_type == 'provider_profile':
            lines.append(f'  type: "{random.choice(["openai", "anthropic", "google"])}"')
            lines.append(f'  base_url: "https://api.example.com/v1"')
        elif backend_type == 'embedding_model':
            lines.append(f'  mmbert_model_path: "models/embedding-model"')
            lines.append(f'  use_cpu: {random.choice(["true", "false"])}')
        elif backend_type == 'semantic_cache':
            lines.append(f'  enabled: true')
            lines.append(f'  backend_type: "{random.choice(["memory", "redis"])}"')
            lines.append(f'  similarity_threshold: {random.uniform(0.8, 0.95):.2f}')
            lines.append(f'  max_entries: {random.choice([1000, 5000, 10000])}')
        elif backend_type == 'vector_store':
            lines.append(f'  type: "{random.choice(["milvus", "pinecone", "qdrant"])}"')
            lines.append(f'  collection: "knowledge_base"')
        else:
            lines.append(f'  enabled: true')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def _gen_global(self) -> str:
        """生成全局配置"""
        lines = ['GLOBAL {']
        lines.append(f'  default_model: "{random.choice(MODEL_NAMES)}"')
        lines.append(f'  strategy: "{random.choice(["priority", "confidence", "random"])}"')
        
        if random.random() > 0.5:
            lines.append(f'  default_reasoning_effort: "{random.choice(["low", "medium", "high"])}"')
        
        if random.random() > 0.6:
            lines.append('')
            lines.append('  prompt_guard: {')
            lines.append(f'    enabled: true')
            lines.append(f'    threshold: {random.uniform(0.6, 0.9):.2f}')
            lines.append('  }')
        
        if random.random() > 0.7:
            lines.append('')
            lines.append('  observability: {')
            lines.append('    metrics: { enabled: true }')
            lines.append('    tracing: {')
            lines.append('      enabled: true')
            lines.append('      provider: "opentelemetry"')
            lines.append('    }')
            lines.append('  }')
        
        lines.append('}')
        return '\n'.join(lines)


def generate_batch(count: int, complexity: str | None = None, seed: int | None = None) -> list[dict]:
    """批量生成 DSL 样本"""
    samples = []
    
    if complexity:
        complexities = [complexity] * count
    else:
        # 按比例分配复杂度 L1:L2:L3:L4:L5 = 10:25:35:20:10
        weights = {'L1': 0.10, 'L2': 0.25, 'L3': 0.35, 'L4': 0.20, 'L5': 0.10}
        complexities = random.choices(
            list(weights.keys()),
            weights=list(weights.values()),
            k=count
        )
    
    for i, comp in enumerate(complexities):
        gen = DSLGenerator(complexity=comp, seed=seed + i if seed else None)
        dsl_text = gen.generate()
        
        dsl_hash = hashlib.md5(dsl_text.encode()).hexdigest()[:8]
        
        samples.append({
            'id': f'syn_{comp}_{i:05d}_{dsl_hash}',
            'dsl': dsl_text,
            'complexity': comp,
            'source': 'cfg_generator',
            'valid': True,  # CFG 生成保证语法正确
            'metadata': {
                'num_signals': len(gen._signals),
                'num_routes': dsl_text.count('ROUTE '),
                'num_plugins': len(gen._plugins),
                'signal_types': list(gen._signals.keys()),
                'char_count': len(dsl_text),
                'line_count': dsl_text.count('\n') + 1,
            }
        })
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic DSL configurations')
    parser.add_argument('--count', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--complexity', type=str, choices=['L1', 'L2', 'L3', 'L4', 'L5'],
                        help='Specific complexity level (default: mixed)')
    parser.add_argument('--output', type=Path, default=Path('synthetic/'),
                        help='Output directory or file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    print(f"Generating {args.count} DSL samples...")
    samples = generate_batch(args.count, args.complexity, args.seed)
    
    # 确保输出目录存在
    if args.output.suffix == '.jsonl':
        output_file = args.output
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        output_file = args.output / 'synthetic_all.jsonl'
    
    # 保存样本
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 如果输出到目录，按复杂度分文件保存
    if args.output.is_dir():
        for comp in ['L1', 'L2', 'L3', 'L4', 'L5']:
            comp_samples = [s for s in samples if s['complexity'] == comp]
            if comp_samples:
                with open(args.output / f'{comp}_{len(comp_samples):05d}.jsonl', 'w', encoding='utf-8') as f:
                    for sample in comp_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 统计
    print(f"\n=== Generation Summary ===")
    print(f"Total samples: {len(samples)}")
    complexity_dist = {}
    for s in samples:
        c = s['complexity']
        complexity_dist[c] = complexity_dist.get(c, 0) + 1
    print(f"By complexity: {complexity_dist}")
    print(f"Output: {output_file}")


if __name__ == '__main__':
    main()
