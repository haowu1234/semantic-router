# Router DSL 专用生成模型：训练与部署设计

## 1. 问题背景与动机

### 1.1 现状分析

当前 NL → DSL 转换管线（见 `nl-to-dsl-conversion.md`）采用 **通用 LLM + Intent IR + 确定性代码生成** 的三阶段架构：

```
NL → [System Prompt ~1500 tokens] → LLM 7B+ → Intent IR → intentIRToDSL() → DSL → Validate → Repair (×3)
```

局限：

| 局限 | 影响 | 根因 |
|:---|:---|:---|
| **LLM 黑盒依赖** | 每次 ~800-2000 tokens system prompt，延迟 1-3s | 通用 LLM 不理解 DSL 语法 |
| **Intent IR 信息损耗** | 复杂嵌套条件(XOR/4层嵌套)转换可能丢失细节 | IR 是中间抽象层 |
| **Few-shot 覆盖不足** | 仅 6 个硬编码示例，无法覆盖 13×14×9×8 种组合 | Prompt 空间有限 |
| **端到端延迟** | P95 ~3s (含修复重试) | 通用模型推理慢 |
| **外部 API 依赖** | 必须联网 | 无法离线运行 |

### 1.2 设计目标

| 目标 | 指标 | 对比现状 |
|:---|:---|:---|
| **语法正确率** | ≥ 98% 通过 `parser.go` | 现约 85% |
| **语义正确率** | ≥ 95% 通过 `validator.go` 三级验证 | 现约 80% |
| **端到端延迟** | P95 < 500ms | 现 P95 ~3s |
| **模型体积** | ≤ 2B 参数 (Q4 ~1GB) | 现依赖 7B+ |
| **离线运行** | 支持 Edge / WebAssembly | 现必须联网 |
| **去掉 Intent IR** | 直接 NL → DSL Text | 现需中间层 |

### 1.3 架构对比

```
现有: NL → System Prompt → LLM 7B+ → Intent IR → intentIRToDSL() → DSL → Validate → Repair (×3)  [~3s]
目标: NL → DSL-Model ≤ 2B → DSL Text → WASM Validate  [< 500ms, 通常无需修复]
```

---

## 2. DSL 语法形式化分析

### 2.1 BNF 语法摘要

从 `pkg/dsl/ast.go` 和 `pkg/dsl/parser.go` 提取：

```bnf
Program         ::= (SignalDecl | RouteDecl | PluginDecl | BackendDecl | GlobalDecl)* EOF

SignalDecl      ::= 'SIGNAL' SIGNAL_TYPE IDENT '{' FieldList '}'
RouteDecl       ::= 'ROUTE' IDENT Options? '{' RouteBody '}'
PluginDecl      ::= 'PLUGIN' IDENT PLUGIN_TYPE '{' FieldList '}'
BackendDecl     ::= 'BACKEND' BACKEND_TYPE IDENT '{' FieldList '}'
GlobalDecl      ::= 'GLOBAL' '{' FieldList '}'

SIGNAL_TYPE     ::= 'keyword' | 'embedding' | 'domain' | 'fact_check' | 'user_feedback'
                   | 'preference' | 'language' | 'context' | 'complexity' | 'modality'
                   | 'authz' | 'jailbreak' | 'pii'
PLUGIN_TYPE     ::= 'semantic_cache' | 'memory' | 'system_prompt' | 'hallucination'
                   | 'router_replay' | 'rag' | 'header_mutation' | 'image_gen' | 'fast_response'
BACKEND_TYPE    ::= 'vllm_endpoint' | 'provider_profile' | 'embedding_model'
                   | 'semantic_cache' | 'memory' | 'response_api' | 'vector_store' | 'image_gen_backend'
ALGORITHM_TYPE  ::= 'confidence' | 'ratings' | 'remom' | 'static' | 'elo' | 'router_dc'
                   | 'automix' | 'hybrid' | 'rl_driven' | 'gmtrouter' | 'latency_aware'
                   | 'knn' | 'kmeans' | 'svm'

RouteBody       ::= PriorityClause? WhenClause? ModelClause+ AlgorithmClause? PluginRefList?
WhenClause      ::= 'WHEN' BoolExpr
BoolExpr        ::= OrExpr
OrExpr          ::= AndExpr ('OR' AndExpr)*
AndExpr         ::= NotExpr ('AND' NotExpr)*
NotExpr         ::= 'NOT'? AtomExpr
AtomExpr        ::= SignalRef | '(' BoolExpr ')'
SignalRef       ::= SIGNAL_TYPE '(' '"' IDENT '"' ')'

FieldList       ::= Field*
Field           ::= IDENT ':' Value
Value           ::= STRING | INT | FLOAT | BOOL | Array | Object
Array           ::= '[' (Value (',' Value)*)? ']'
Object          ::= '{' FieldList '}'
```

### 2.2 复杂度分级

| 级别 | 构造数 | 条件深度 | 示例 |
|:---|:---|:---|:---|
| **L1** | 2-3 | 0 | "Route all to GPT-4o" |
| **L2** | 4-6 | 1 | "Math→DeepSeek, coding→GPT-4o" |
| **L3** | 6-10 | 2 | "3-tier + jailbreak blocking" |
| **L4** | 10-15 | 3+ | "RBAC + modality + LoRA + cascade" |
| **L5** | 15+ | 4+ | 完整生产级配置 |

---

## 3. 高质量 DSL 语料库构建

### 3.1 数据来源矩阵

| 来源 | 方法 | 数量 | 质量 |
|:---|:---|:---|:---|
| **A: 现有代码库** | `dsl_test.go` 提取 + `config/*.yaml` 反编译 + 测试文件 | ~250 | 金标准 |
| **B: CFG 合成生成** | 基于 BNF 随机游走 + 模板填充 + 变异模糊测试 | ~10,000 | 语法保证 |
| **C: LLM NL 描述** | 对 A+B 每个 DSL 用 GPT-4o 生成 3 英/2 中/1 模糊描述 | ~60,000 | 需人工抽检 |
| **D: 负样本** | 语法错误 / 引用错误 / 约束违反 / Schema 不匹配 | ~5,000 | DPO 训练用 |
| **总计** | | **~75,000 (NL, DSL, label) 三元组** | |

### 3.2 种子数据提取

#### 从 dsl_test.go 提取有效 DSL

```python
import re

def extract_dsl_from_tests(test_file: str) -> list[dict]:
    with open(test_file) as f:
        content = f.read()
    pattern = r'`((?:SIGNAL|ROUTE|PLUGIN|BACKEND|GLOBAL)[\s\S]*?)`'
    matches = re.findall(pattern, content)
    return [{'dsl': m.strip(), 'source': 'dsl_test.go', 'valid': True} for m in matches if len(m) > 10]
```

#### 从 YAML 配置反编译

```python
import subprocess, glob

def decompile_yamls(config_dir: str, decompiler_bin: str) -> list[dict]:
    samples = []
    for path in glob.glob(f'{config_dir}/**/*.yaml', recursive=True):
        result = subprocess.run([decompiler_bin, '--input', path, '--format', 'dsl'],
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            samples.append({'dsl': result.stdout.strip(), 'source': path, 'valid': True})
    return samples
```

### 3.3 CFG 随机游走生成器

核心思想：基于 §2.1 BNF 语法进行随机游走，保证**每个生成的 DSL 都语法有效**。

```python
import random

SIGNAL_TYPES = ['keyword', 'embedding', 'domain', 'fact_check', 'user_feedback',
    'preference', 'language', 'context', 'complexity', 'modality', 'authz', 'jailbreak', 'pii']
PLUGIN_TYPES = ['semantic_cache', 'memory', 'system_prompt', 'hallucination',
    'router_replay', 'rag', 'header_mutation', 'image_gen', 'fast_response']
ALGORITHM_TYPES = ['confidence', 'ratings', 'remom', 'static', 'elo', 'router_dc',
    'automix', 'hybrid', 'rl_driven', 'gmtrouter', 'latency_aware', 'knn', 'kmeans', 'svm']
BACKEND_TYPES = ['vllm_endpoint', 'provider_profile', 'embedding_model',
    'semantic_cache', 'memory', 'response_api', 'vector_store', 'image_gen_backend']
MODEL_NAMES = ['gpt-4o', 'gpt-4o-mini', 'deepseek-r1', 'qwen2.5:3b', 'qwen3:70b', 'claude-3-sonnet']

# 各信号类型的字段 schema (与 nlSchemaRegistry.ts 同步)
SIGNAL_SCHEMAS = {
    'keyword': {'required': {'operator': ['any','all'], 'keywords': 'str_arr'},
                'optional': {'method': ['regex','bm25','ngram'], 'case_sensitive': 'bool'}},
    'embedding': {'required': {'threshold': (0,1,'f'), 'candidates': 'str_arr'},
                  'optional': {'aggregation_method': ['mean','max','any']}},
    'domain': {'required': {'description': 'str'}, 'optional': {'mmlu_categories': 'str_arr'}},
    'jailbreak': {'required': {'threshold': (0,1,'f')},
                  'optional': {'method': ['classifier','contrastive']}},
    'pii': {'required': {'threshold': (0,1,'f')}, 'optional': {'pii_types_allowed': 'str_arr'}},
    'context': {'required': {'min_tokens': ['1K','4K','8K'], 'max_tokens': ['16K','32K','128K']}, 'optional': {}},
    'complexity': {'required': {'threshold': (0,1,'f')}, 'optional': {'description': 'str'}},
    # ... 其余类型类似
}

class DSLGenerator:
    def __init__(self, max_signals=5, max_routes=3, max_depth=2):
        self.max_signals, self.max_routes, self.max_depth = max_signals, max_routes, max_depth
        self._signals = {}  # type -> [names]

    def generate(self) -> str:
        self._signals = {}
        parts = []
        for _ in range(random.randint(1, self.max_signals)):
            parts.append(self._signal())
        if random.random() > 0.4:
            parts.append(self._plugin_template())
        for _ in range(random.randint(1, self.max_routes)):
            parts.append(self._route())
        if random.random() > 0.3:
            parts.append(self._backend())
        if random.random() > 0.4:
            parts.append(self._global())
        return '\n\n'.join(parts)

    def _signal(self) -> str:
        t = random.choice(SIGNAL_TYPES)
        n = f'{t}_{random.randint(1,99)}'
        self._signals.setdefault(t, []).append(n)
        fields = self._gen_fields(t)
        return f'SIGNAL {t} {n} {{\n{fields}\n}}'

    def _route(self) -> str:
        name = f'route_{random.randint(1,999)}'
        lines = [f'ROUTE {name} (description = "{name}") {{',
                 f'  PRIORITY {random.choice([10,50,100,200,500])}']
        if self._signals:
            lines.append(f'  WHEN {self._bool_expr(0)}')
        models = random.sample(MODEL_NAMES, random.randint(1,2))
        model_strs = [f'"{m}" (reasoning = {random.choice(["true","false"])})' for m in models]
        lines.append(f'  MODEL {", ".join(model_strs)}')
        if random.random() > 0.6:
            algo = random.choice(ALGORITHM_TYPES)
            lines.extend([f'  ALGORITHM {algo} {{', f'    on_error: "skip"', f'  }}'])
        lines.append('}')
        return '\n'.join(lines)

    def _bool_expr(self, depth) -> str:
        if depth >= self.max_depth or random.random() > 0.6:
            return self._signal_ref()
        op = random.choice(['AND', 'OR', 'NOT'])
        if op == 'NOT':
            inner = self._bool_expr(depth+1)
            return f'NOT ({inner})' if ' ' in inner else f'NOT {inner}'
        return f'{self._bool_expr(depth+1)} {op} {self._bool_expr(depth+1)}'

    def _signal_ref(self) -> str:
        t = random.choice(list(self._signals.keys())) if self._signals else 'domain'
        n = random.choice(self._signals.get(t, ['fallback']))
        return f'{t}("{n}")'

    def _gen_fields(self, sig_type) -> str:
        schema = SIGNAL_SCHEMAS.get(sig_type, {'required': {'description': 'str'}, 'optional': {}})
        lines = []
        for f, spec in schema['required'].items():
            lines.append(f'  {f}: {self._val(spec)}')
        for f, spec in schema['optional'].items():
            if random.random() > 0.5:
                lines.append(f'  {f}: {self._val(spec)}')
        return '\n'.join(lines)

    def _val(self, spec):
        if isinstance(spec, list): return f'"{random.choice(spec)}"'
        if spec == 'str': return '"auto-generated description"'
        if spec == 'str_arr': return '["item1", "item2"]'
        if spec == 'bool': return random.choice(['true', 'false'])
        if isinstance(spec, tuple): return f'{random.uniform(spec[0], spec[1]):.2f}'
        return '"default"'

    def _plugin_template(self) -> str:
        t = random.choice(PLUGIN_TYPES)
        return f'PLUGIN tpl_{t} {t} {{\n  enabled: true\n}}'

    def _backend(self) -> str:
        t = random.choice(BACKEND_TYPES)
        return f'BACKEND {t} be_{random.randint(1,99)} {{\n  address: "127.0.0.1"\n  port: 8080\n}}'

    def _global(self) -> str:
        return f'GLOBAL {{\n  default_model: "{random.choice(MODEL_NAMES)}"\n  strategy: "priority"\n}}'
```

### 3.4 NL 描述自动化生成

对每个有效 DSL，用 GPT-4o 生成 6 种风格的 NL 描述：

```python
PROMPT = """Given this Signal DSL config, generate 6 NL descriptions (JSON):
- en_formal, en_casual, en_technical: English styles
- zh_formal, zh_casual: Chinese styles
- ambiguous: intentionally underspecified

DSL:
```
{dsl}
```
Output JSON only."""

def generate_nl(client, dsl: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0.8,
        messages=[{"role": "user", "content": PROMPT.format(dsl=dsl)}],
        response_format={"type": "json_object"})
    return json.loads(resp.choices[0].message.content)
```

### 3.5 负样本生成

通过变异有效 DSL 生成错误样本 (用于 DPO 训练)：

| 变异类型 | 方法 | 用途 |
|:---|:---|:---|
| `syntax_error` | 删除 `}` / 改 `SIGNAL→SIGNALS` | 教模型避免语法错误 |
| `reference_error` | WHEN 引用未定义信号 | 教模型保持引用完整性 |
| `constraint_violation` | `threshold: 1.5` | 教模型遵守值域约束 |
| `schema_mismatch` | 给 keyword 类型添加 `mmlu_categories` | 教模型区分字段归属 |

### 3.6 数据验证管线

所有生成数据必须通过 Go 编译器验证，确保标签正确：

```python
def validate_batch(samples: list, parser_bin: str) -> list:
    validated = []
    for s in samples:
        result = subprocess.run([parser_bin, '--validate', '--json'],
                               input=s['dsl'], capture_output=True, text=True, timeout=5)
        actual_valid = json.loads(result.stdout).get('valid', False)
        if actual_valid == s.get('valid', True):
            validated.append(s)
    return validated
```

### 3.7 最终数据集格式

```jsonc
// train.jsonl — 每行一个样本
{
  "id": "syn_L3_00042_en_casual",
  "instruction": "Convert the following natural language description into Signal DSL configuration.",
  "input": "I need a 3-tier routing setup...",
  "output": "SIGNAL domain math {\n  description: \"Math\"\n  ...\n}\n\nROUTE math_route ...",
  "complexity": "L3",
  "style": "en_casual",
  "valid": true,
  "metadata": {"num_signals": 3, "num_routes": 3, "signal_types": ["domain","jailbreak"], "condition_depth": 1}
}
```

---

## 4. DSL-Aware Tokenizer 设计

### 4.1 问题

通用 Tokenizer 将 DSL 关键字碎片化：

```
"SIGNAL" → ["SIGN", "AL"]           # 2 tokens — 语义割裂
"embedding" → ["embed", "ding"]      # 2 tokens
"similarity_threshold" → 4 tokens    # 浪费上下文
```

### 4.2 方案：在基座 Tokenizer 上添加 ~120 个 DSL 专用 Token

```python
# 5 层 Token 设计
DSL_SPECIAL_TOKENS = (
    # Layer 1: Keywords (16)
    ['SIGNAL', 'ROUTE', 'PLUGIN', 'BACKEND', 'GLOBAL',
     'PRIORITY', 'WHEN', 'MODEL', 'ALGORITHM', 'AND', 'OR', 'NOT']
    # Layer 2: Type Names (44) — 每个类型名单 token
    + [f'signal_{t}' for t in SIGNAL_TYPES]
    + [f'plugin_{t}' for t in PLUGIN_TYPES]
    + [f'algo_{t}' for t in ALGORITHM_TYPES]
    + [f'backend_{t}' for t in BACKEND_TYPES]
    # Layer 3: Common Field Names (30)
    + ['threshold', 'similarity_threshold', 'description', 'enabled', 'candidates',
       'keywords', 'operator', 'method', 'confidence_method', 'on_error',
       'system_prompt', 'reasoning', 'effort', 'weight', 'address', 'port',
       'default_model', 'strategy', 'mmlu_categories', 'aggregation_method']
    # Layer 4: Common Model Names (16)
    + ['gpt-4o', 'gpt-4o-mini', 'deepseek-r1', 'qwen2.5:3b', 'qwen3:70b',
       'claude-3-sonnet', 'llama3.1:8b', 'gemini-2.0-flash']
    # Layer 5: Structural Patterns (8)
    + [' {\n', '\n}\n', '  ', '    ', '(description = "', '(reasoning = ']
)

# 构建方法
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
tokenizer.add_tokens(DSL_SPECIAL_TOKENS, special_tokens=False)
# 验证: "SIGNAL" 现在是 1 token (之前是 2)
```

### 4.3 效率预估

| 指标 | 通用 BPE | DSL-Enhanced | 提升 |
|:---|:---|:---|:---|
| "SIGNAL domain math" | 5 tokens | 3 tokens | -40% |
| 典型 L3 (~800 chars) | ~200 tokens | ~120 tokens | -40% |
| 上下文利用率 (2K) | ~4 配置 | ~7 配置 | +75% |

---

## 5. 基座模型选择

### 5.1 候选评估

| 模型 | 参数量 | 上下文 | 代码特化 | Q4 体积 | 推荐 |
|:---|:---|:---|:---|:---|:---|
| **Qwen2.5-Coder-1.5B** | 1.5B | 128K | ✅ 强 | ~900MB | ⭐⭐⭐⭐⭐ |
| StarCoder2-3B | 3B | 16K | ✅ | ~1.8GB | ⭐⭐⭐⭐ |
| CodeGemma-2B | 2B | 8K | ✅ FIM | ~1.2GB | ⭐⭐⭐⭐ |
| DeepSeek-Coder-1.3B | 1.3B | 16K | ✅ | ~800MB | ⭐⭐⭐⭐ |

### 5.2 推荐: Qwen2.5-Coder-1.5B

- **128K 上下文**：支持 L5 级长配置
- **中英双语**：匹配 `nlSchemaRegistry.ts` 中的中文触发词
- **Q4 ~900MB**：可在 Apple M 系列 / 消费级 GPU 运行
- **HumanEval 表现**：1.5B 参数接近某些 7B 通用模型

---

## 6. 微调策略

### 6.1 QLoRA 架构

```
Base Model: Qwen2.5-Coder-1.5B (frozen, 4-bit NF4)
├── Embedding Layer ← resize for +120 DSL tokens
├── Attention (Q,K,V,O) ← LoRA rank=64, alpha=128
├── MLP (gate,up,down) ← LoRA rank=32, alpha=64
└── LM Head ← unfrozen (for new tokens)

Trainable: ~20M params (1.3% of total), ~6GB VRAM
```

### 6.2 超参数

```python
# LoRA Config
lora_config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    rank_pattern={"gate_proj":32, "up_proj":32, "down_proj":32},
)

# Training Args
args = TrainingArguments(
    num_train_epochs=5, per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
    bf16=True, gradient_checkpointing=True, optim="paged_adamw_8bit",
    eval_steps=200, save_steps=500, metric_for_best_model="eval_syntax_accuracy",
)
```

### 6.3 四阶段训练策略

```
Stage 1: DSL Syntax Pretraining (Epoch 1-2)
├── Data: 全部合成 DSL (Source B) — 纯 DSL 文本，无 NL
├── Task: Causal LM (next token prediction)
├── Goal: 内化 DSL 语法结构、关键字分布、嵌套模式
└── LR: 2e-4 → cosine decay

Stage 2: NL → DSL Supervised Fine-Tuning (Epoch 3-4)
├── Data: NL→DSL 对 (Source A+C) — chat format
├── Task: Instruction-following (仅 DSL 输出部分计算 loss)
├── Goal: 建立 NL→DSL 映射
├── LR: 1e-4 (降低，防遗忘)
└── 采样: L1:L2:L3:L4:L5 = 10:25:35:20:10

Stage 3: DPO Preference Alignment (Epoch 5)
├── Data: (NL, DSL_chosen, DSL_rejected) triples
│   chosen = 通过验证的 DSL, rejected = 负样本 (Source D)
├── Goal: 强化语法正确性和引用完整性偏好
└── β = 0.1

Stage 4: RLHF with Compiler Feedback (Optional, Research)
├── Reward: WASM 编译器 → 1.0 (pass) / 0.5 (warning) / 0.0 (error)
├── Method: PPO or GRPO
└── Risk: 需 KL 散度约束防过拟合
```

### 6.4 训练数据格式

```python
# Qwen2.5 Chat Template
TEMPLATE = """<|im_start|>system
You are a Signal DSL expert. Convert natural language into valid Signal DSL.<|im_end|>
<|im_start|>user
{nl_input}<|im_end|>
<|im_start|>assistant
{dsl_output}<|im_end|>"""

# Loss masking: 仅在 <|im_start|>assistant 之后的 token 上计算 loss
```

---

## 7. Constrained Decoding (受约束解码)

利用 DSL BNF 语法在推理阶段约束 token 生成，从根本上消除语法错误：

```python
class DSLGrammarConstraint:
    """基于状态机的解码约束器"""

    def __init__(self, tokenizer):
        self.state = 'PROGRAM_START'
        self.depth = 0
        # 预计算 token 分组
        self.toplevel_ids = tokens_for(['SIGNAL','ROUTE','PLUGIN','BACKEND','GLOBAL'])
        self.signal_type_ids = tokens_for(SIGNAL_TYPES)
        self.bool_op_ids = tokens_for(['AND','OR','NOT'])

    def get_allowed_tokens(self) -> set[int]:
        if self.state == 'PROGRAM_START':
            return self.toplevel_ids | {eos_id}
        elif self.state == 'SIGNAL_TYPE':
            return self.signal_type_ids
        elif self.state == 'WHEN_EXPR':
            return self.signal_type_ids | self.bool_op_ids | tokens_for(['(', ')'])
        # ... 其余状态
        return all_tokens

    def update_state(self, token_id: int):
        text = decode(token_id)
        if text == 'SIGNAL': self.state = 'SIGNAL_TYPE'
        elif text == '{': self.depth += 1; self.state = 'FIELD_NAME'
        elif text == '}':
            self.depth -= 1
            if self.depth == 0: self.state = 'PROGRAM_START'
        elif text == 'WHEN': self.state = 'WHEN_EXPR'
```

> 实际部署推荐使用 [Outlines](https://github.com/outlines-dev/outlines) 或 [Guidance](https://github.com/guidance-ai/guidance) 库，它们原生支持 BNF/正则受约束生成 + vLLM 集成。

---

## 8. 评估体系

### 8.1 五级评估指标

| 级别 | 指标 | 目标 | 方法 |
|:---|:---|:---|:---|
| **L1 语法** | `syntax_accuracy` | ≥ 98% | `parser.go` 自动 |
| **L1 语法** | `bracket_match_rate` | ≥ 99% | 自动 |
| **L2 语义** | `validation_pass_rate` | ≥ 95% | `validator.go` 三级 |
| **L2 语义** | `reference_integrity` | ≥ 97% | 无未定义引用 |
| **L2 语义** | `type_accuracy` | ≥ 95% | NL 语义→正确类型 |
| **L3 功能** | `compilation_success` | ≥ 93% | `compiler.go` |
| **L3 功能** | `decompile_roundtrip` | ≥ 90% | DSL→Compile→Decompile→比对 |
| **L4 忠实度** | `intent_coverage` | ≥ 90% | LLM 辅助 / 人工 |
| **L5 性能** | `latency_p95` | < 500ms | 端到端计时 |
| **L5 性能** | `throughput` | ≥ 100 tok/s | A10 GPU |

### 8.2 评估基准集

```
DSL-Bench v1/
├── syntax_only/        # 50 样本 — 纯语法正确性
├── nl_to_dsl/          # 200 样本 — NL→DSL 端到端
│   ├── L1_10.jsonl     # 各复杂度分组
│   ├── L2_40.jsonl
│   ├── L3_80.jsonl
│   ├── L4_50.jsonl
│   └── L5_20.jsonl
├── edge_cases/         # 30 样本 — 边界情况
│   ├── xor_logic.jsonl
│   ├── deep_nesting.jsonl
│   └── ambiguous_nl.jsonl
├── chinese_nl/         # 50 样本 — 中文 NL 输入
├── modify_mode/        # 30 样本 — 增量修改
└── adversarial/        # 20 样本 — 对抗样本
```

### 8.3 自动化评估脚本

```python
class DSLEvaluator:
    def __init__(self, parser_bin, compiler_bin):
        self.parser_bin, self.compiler_bin = parser_bin, compiler_bin

    def evaluate(self, generated_dsl: str) -> dict:
        # L1: 语法
        parse = subprocess.run([self.parser_bin, '--parse', '--json'],
                              input=generated_dsl, capture_output=True, text=True)
        syntax_ok = json.loads(parse.stdout).get('success', False)
        if not syntax_ok:
            return {'syntax': False, 'semantic': False, 'compile': False}

        # L2: 语义
        validate = subprocess.run([self.parser_bin, '--validate', '--json'],
                                 input=generated_dsl, capture_output=True, text=True)
        val_result = json.loads(validate.stdout)
        semantic_ok = len(val_result.get('errors', [])) == 0

        # L3: 编译
        compile = subprocess.run([self.compiler_bin, '--compile', '--json'],
                                input=generated_dsl, capture_output=True, text=True)
        compile_ok = json.loads(compile.stdout).get('success', False)

        return {'syntax': syntax_ok, 'semantic': semantic_ok, 'compile': compile_ok,
                'warnings': val_result.get('warnings', []), 'errors': val_result.get('errors', [])}

    def batch_report(self, test_set: list) -> dict:
        results = [self.evaluate(s['generated']) for s in test_set]
        n = len(results)
        return {
            'syntax_accuracy': sum(r['syntax'] for r in results) / n,
            'validation_pass_rate': sum(r['semantic'] for r in results) / n,
            'compilation_success': sum(r['compile'] for r in results) / n,
        }
```

---

## 9. 推理部署

### 9.1 部署拓扑

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Deployment Options                              │
│                                                                      │
│  Option A: vLLM Server (推荐生产环境)                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  vLLM + QLoRA merged model (Q4_K_M GGUF)                      │  │
│  │  • GPU: Single A10 (24GB) or L4 (24GB)                        │  │
│  │  • Throughput: ~200 tok/s                                      │  │
│  │  • Latency: P95 ~300ms (L3 config)                            │  │
│  │  • Concurrent: ~50 QPS with continuous batching                │  │
│  │  • Constrained decoding via Outlines LogitsProcessor           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Option B: llama.cpp / Ollama (轻量部署)                              │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  GGUF Q4_K_M (~900MB)                                          │  │
│  │  • Hardware: Apple M2+ (8GB RAM) / Consumer GPU                │  │
│  │  • Throughput: ~50-80 tok/s                                    │  │
│  │  • Latency: P95 ~500ms                                        │  │
│  │  • Use: Dev/staging, 本地离线                                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Option C: WebAssembly (浏览器端, 研究方向)                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  ONNX quantized (INT8) via WebNN / WASM                        │  │
│  │  • Model size: ~500MB (INT4)                                   │  │
│  │  • Throughput: ~10-20 tok/s                                    │  │
│  │  • Use: 完全离线, 无服务器, 与现有 WASM 验证器同运行时              │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 输入输出格式规范

#### 输入格式

```json
{
  "model": "dsl-coder-v1",
  "messages": [
    {
      "role": "system",
      "content": "You are a Signal DSL expert. Convert natural language into valid Signal DSL."
    },
    {
      "role": "user",
      "content": "Route math questions to deepseek-r1 with reasoning enabled, and block jailbreak attempts with a warning message. Use qwen2.5:3b as the default model."
    }
  ],
  "temperature": 0.1,
  "max_tokens": 2048,
  "stop": ["<|im_end|>"]
}
```

#### 输出格式

```json
{
  "id": "dsl-gen-001",
  "object": "chat.completion",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "SIGNAL domain math {\n  description: \"Mathematics and quantitative reasoning\"\n  mmlu_categories: [\"math\"]\n}\n\nSIGNAL jailbreak jailbreak_detect {\n  method: \"classifier\"\n  threshold: 0.9\n}\n\nPLUGIN block_jailbreak fast_response {\n  message: \"Request blocked for safety reasons.\"\n  enabled: true\n}\n\nROUTE safety_route (description = \"Block jailbreak attempts\") {\n  PRIORITY 1000\n  WHEN jailbreak(\"jailbreak_detect\")\n  MODEL \"qwen2.5:3b\"\n  PLUGIN block_jailbreak\n}\n\nROUTE math_route (description = \"Route math to reasoning model\") {\n  PRIORITY 100\n  WHEN domain(\"math\")\n  MODEL \"deepseek-r1\" (reasoning = true, effort = \"high\")\n}\n\nGLOBAL {\n  default_model: \"qwen2.5:3b\"\n  strategy: \"priority\"\n}"
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 42, "completion_tokens": 156, "total_tokens": 198}
}
```

### 9.3 性能优化

| 技术 | 效果 | 适用场景 |
|:---|:---|:---|
| **KV Cache** | 减少重复计算 | 所有场景 |
| **Continuous Batching** | 提升吞吐 ~3x | vLLM 多用户 |
| **Speculative Decoding** | 降低延迟 ~2x | DSL 结构可预测 |
| **Prefix Caching** | System prompt 不重复编码 | 所有场景 |
| **Constrained Decoding** | 消除语法错误 + 减少无效 token | 所有场景 |
| **INT4 Quantization (AWQ/GPTQ)** | 体积 -75%, 速度 +30% | 边缘设备 |
| **Flash Attention 2** | 内存 -50%, 速度 +20% | GPU 推理 |

### 9.4 Speculative Decoding 策略

DSL 具有高度可预测的结构模式，特别适合 speculative decoding：

```
Draft model: 更小的 DSL 模型 (350M) 或 n-gram 模型
Target model: DSL-Coder 1.5B

可预测的模式:
- "SIGNAL" 后必跟类型名 → draft 准确率 ~95%
- "{" 后必跟 "\n  " → draft 准确率 ~99%
- "WHEN" 后必跟信号引用 → draft 准确率 ~90%
- "GLOBAL {\n  default_model:" → 固定前缀 draft ~100%
```

---

## 10. 与现有系统集成

### 10.1 混合推理管线

将 DSL-Model 作为 **快速路径**，现有 LLM + Intent IR 作为 **兜底路径**：

```
                    ┌──────────────────────┐
                    │     NL Input         │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   DSL-Model (≤2B)    │ ← 快速路径 (< 500ms)
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   WASM Validate      │
                    └──────────┬───────────┘
                          ┌────┴────┐
                     Pass │         │ Fail
                          ▼         ▼
                     ┌────────┐ ┌────────────────────┐
                     │ Output │ │ Fallback: LLM 7B+  │ ← 现有管线
                     └────────┘ │ Intent IR → DSL     │
                                │ Repair Loop (×3)    │
                                └────────────────────┘
```

### 10.2 Dashboard 集成点

```typescript
// dashboard/frontend/src/lib/nlPipeline.ts — 修改 processNLInput()
async function processNLInput(input: string, options: NLOptions): Promise<NLResult> {
  // 新增: 尝试 DSL-Model 快速路径
  if (options.useDSLModel && dslModelAvailable()) {
    const dslText = await callDSLModel(input);       // < 500ms
    const validation = await wasmValidate(dslText);   // < 5ms
    if (validation.errors.length === 0) {
      return { dsl: dslText, method: 'dsl-model', latency: Date.now() - start };
    }
    // Fallback 到 Stage 1+2 修复
    console.warn('DSL-Model output invalid, falling back to Intent IR pipeline');
  }

  // 现有管线 (unchanged)
  return existingPipeline(input, options);
}
```

### 10.3 后端 API 路由

```go
// dashboard/backend/handlers/nl_generate.go — 新增端点
// POST /api/v1/nl/generate-dsl
func (h *NLHandler) GenerateDSL(w http.ResponseWriter, r *http.Request) {
    var req struct {
        Input string `json:"input"`
        Model string `json:"model"` // "dsl-coder-v1" or "gpt-4o"
    }
    // 如果指定 dsl-coder-v1, 路由到本地 vLLM/Ollama 实例
    // 否则走现有 LLM 代理逻辑
}
```

---

## 11. 实施路线图

### Phase 1: 数据工程 (Week 1-2)

| 任务 | 输出 | 工时 |
|:---|:---|:---|
| 实现 `extract_dsl_from_tests.py` | ~120 种子 DSL | 0.5d |
| 实现反编译器 CLI + `yaml_to_dsl_decompile.py` | ~60 种子 DSL | 1d |
| 实现 `dsl_generator.py` CFG 生成器 | ~10,000 合成 DSL | 2d |
| 实现 `generate_nl_descriptions.py` | ~60,000 NL-DSL 对 | 1d (+ GPT-4o API 费 ~$50) |
| 实现 `generate_negative_samples.py` | ~5,000 负样本 | 1d |
| 数据验证 + 质量抽检 | 清洗后数据集 | 1d |

### Phase 2: 模型训练 (Week 2-3)

| 任务 | 输出 | 工时 |
|:---|:---|:---|
| Tokenizer 扩展 + 验证 | DSL-enhanced tokenizer | 0.5d |
| Stage 1: DSL 语法预训练 | 语法准确率 baseline | 1d (A100 ~8h) |
| Stage 2: NL→DSL SFT | 端到端模型 v0.1 | 1d (A100 ~12h) |
| Stage 3: DPO 对齐 | 模型 v0.2 | 0.5d (A100 ~4h) |
| 评估基准建立 + 测试 | DSL-Bench v1 报告 | 1d |

### Phase 3: 推理部署 (Week 3-4)

| 任务 | 输出 | 工时 |
|:---|:---|:---|
| 模型量化 (GGUF Q4_K_M) | ~900MB 模型文件 | 0.5d |
| vLLM 部署 + Constrained Decoding | 生产级 API | 1d |
| Ollama 集成 (本地开发) | Ollama Modelfile | 0.5d |
| Dashboard 集成 (快速路径 + fallback) | 前后端联调 | 1.5d |

### Phase 4: 迭代优化 (Week 4+)

| 任务 | 输出 | 工时 |
|:---|:---|:---|
| 收集真实用户 NL→DSL 对 | 增量训练数据 | 持续 |
| 模型 A/B 测试 | 性能报告 | 1d |
| Speculative decoding 实验 | 延迟优化 | 1d |
| Stage 4: RLHF 实验 (可选) | 模型 v1.0 | 2d |

**总计：约 4-5 周，单人可完成 Phase 1-3，Phase 4 为持续迭代。**

---

## 12. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|:---|:---|:---|:---|
| 1.5B 模型容量不足，复杂配置 (L4/L5) 准确率低 | 中 | 高 | Fallback 到 7B+ LLM; 控制复杂度上限 |
| 合成数据分布偏移，真实用户 NL 表述与训练分布不同 | 高 | 中 | 持续收集真实数据; 在线学习 |
| Constrained decoding 增加推理延迟 | 低 | 中 | 仅约束关键状态; 使用 Outlines 高效实现 |
| DSL 语法演进 (新增 Signal/Plugin 类型) | 中 | 中 | 保持 Tokenizer 和生成器与 nlSchemaRegistry.ts 同步 |
| 过拟合编译器 reward (Stage 4 RLHF) | 中 | 中 | KL 散度约束; 人工评估兜底 |
