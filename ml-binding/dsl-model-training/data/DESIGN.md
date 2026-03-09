# DSL Model Training - 数据集设计文档

## 1. 概述

本文档详细描述 Router DSL 专用生成模型的训练数据集构建方案，包括数据来源、生成策略、质量保证和最终数据格式。

## 2. 数据来源矩阵

| 来源 | 方法 | 预估数量 | 质量等级 | 用途 |
|:---|:---|:---|:---|:---|
| **A: 现有代码库** | 测试文件提取 + YAML反编译 | ~250 | ⭐⭐⭐⭐⭐ 金标准 | 种子数据 |
| **B: CFG 合成** | BNF 随机游走 + Schema约束 | ~10,000 | ⭐⭐⭐⭐ 语法保证 | Stage 1 预训练 |
| **C: LLM NL 描述** | GPT-4o 6种风格描述 | ~60,000 | ⭐⭐⭐ 需抽检 | Stage 2 SFT |
| **D: 负样本** | 变异生成 | ~5,000 | ⭐⭐⭐⭐ DPO | Stage 3 DPO |

**总计: ~75,000 (NL, DSL, label) 三元组**

## 3. 数据生成流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据生成管线                                       │
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │  种子提取    │ → │  CFG 生成    │ → │  NL 描述     │ → │  负样本      │  │
│  │              │   │              │   │              │   │              │  │
│  │ dsl_test.go  │   │ 随机游走     │   │ GPT-4o       │   │ 语法错误     │  │
│  │ config/*.yaml│   │ Schema约束   │   │ 6种风格      │   │ 引用错误     │  │
│  │ DslGuide.tsx │   │ 复杂度分级   │   │ 中英双语     │   │ 约束违反     │  │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Go Validator 验证管线                          │   │
│  │  • 语法检查 (parser.go)                                              │   │
│  │  • 语义检查 (validator.go - 引用完整性)                               │   │
│  │  • 编译检查 (compiler.go)                                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        最终训练数据集                                  │   │
│  │                                                                      │   │
│  │  stage1_syntax_pt.jsonl   - Stage 1: 纯 DSL 语法预训练               │   │
│  │  stage2_sft.jsonl         - Stage 2: NL→DSL SFT                      │   │
│  │  stage3_dpo.jsonl         - Stage 3: DPO 偏好对齐                    │   │
│  │  eval_benchmark.jsonl     - 评估基准集                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 4. 复杂度分级标准

| 级别 | 构造数 | 条件深度 | 典型场景 | 比例 |
|:---|:---|:---|:---|:---|
| **L1** | 2-3 | 0 | "Route all to GPT-4o" | 10% |
| **L2** | 4-6 | 1 | "Math→DeepSeek, coding→GPT-4o" | 25% |
| **L3** | 6-10 | 2 | "3-tier cascade + jailbreak blocking" | 35% |
| **L4** | 10-15 | 3+ | "RBAC + modality + LoRA + cascade" | 20% |
| **L5** | 15+ | 4+ | 完整生产级配置 | 10% |

### 复杂度判定规则

```python
def classify_complexity(dsl: str) -> str:
    signal_count = count_signals(dsl)
    route_count = count_routes(dsl)
    plugin_count = count_plugins(dsl)
    backend_count = count_backends(dsl)
    
    total_constructs = signal_count + route_count + plugin_count + backend_count
    condition_depth = estimate_condition_depth(dsl)  # 基于 AND/OR/NOT 嵌套
    
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
```

## 5. NL 描述风格说明

| 风格 | 语言 | 特点 | 示例 |
|:---|:---|:---|:---|
| `en_formal` | 英文 | 技术文档风格，精确完整 | "Configure a router with domain-based routing using DeepSeek R1 for mathematics queries with reasoning enabled at high effort." |
| `en_casual` | 英文 | 对话风格，简洁自然 | "Set up routing so math questions go to DeepSeek with full reasoning." |
| `en_technical` | 英文 | 工程师风格，使用DSL术语 | "Create SIGNAL domain for math with MMLU categories, ROUTE with MODEL deepseek-r1 (reasoning=true, effort=high)." |
| `zh_formal` | 中文 | 正式技术风格 | "配置一个路由器，将数学领域的查询路由到DeepSeek R1模型，启用高强度推理。" |
| `zh_casual` | 中文 | 口语化风格 | "把数学问题发给DeepSeek，让它仔细推理。" |
| `ambiguous` | 混合 | 故意欠指定 | "Route math to a good reasoning model." |

## 6. 负样本变异策略

### 6.1 语法错误 (`syntax_error`)

| 变异类型 | 描述 | 示例 |
|:---|:---|:---|
| `missing_brace` | 删除闭合括号 | `SIGNAL domain math { description: "test"` |
| `typo_keyword` | 拼错关键字 | `SINGAL domain math { ... }` |
| `missing_colon` | 删除字段冒号 | `description "test"` |
| `unclosed_string` | 未闭合字符串 | `description: "test` |

### 6.2 引用错误 (`reference_error`)

| 变异类型 | 描述 | 示例 |
|:---|:---|:---|
| `undefined_signal` | 引用未定义信号 | `WHEN domain("undefined_123")` |
| `wrong_signal_type` | 错误信号类型 | `WHEN keyword("math")` (math 是 domain) |
| `undefined_plugin` | 引用未定义插件 | `PLUGIN nonexistent_plugin` |

### 6.3 约束违反 (`constraint_violation`)

| 变异类型 | 描述 | 示例 |
|:---|:---|:---|
| `threshold_overflow` | 阈值超范围 | `threshold: 1.5` (应为 0-1) |
| `priority_negative` | 负优先级 | `PRIORITY -100` |
| `port_overflow` | 端口超范围 | `port: 70000` |

### 6.4 Schema 不匹配 (`schema_mismatch`)

| 变异类型 | 描述 | 示例 |
|:---|:---|:---|
| `wrong_field_type` | 字段类型错误 | `enabled: "yes"` (应为 bool) |
| `wrong_field_for_type` | 字段不属于该类型 | keyword 信号添加 `mmlu_categories` |
| `invalid_enum` | 无效枚举值 | `method: "invalid_method"` |

## 7. 数据格式规范

### 7.1 Stage 1: 语法预训练

```jsonc
// stage1_syntax_pt.jsonl - 每行一个样本
{
  "id": "syn_L3_00042",
  "dsl": "SIGNAL domain math {\n  description: \"Mathematics\"\n  mmlu_categories: [\"math\"]\n}\n\nROUTE math_route {\n  PRIORITY 100\n  WHEN domain(\"math\")\n  MODEL \"deepseek-r1\" (reasoning = true)\n}",
  "complexity": "L3"
}
```

### 7.2 Stage 2: SFT (NL→DSL)

```jsonc
// stage2_sft.jsonl - Chat 格式
{
  "id": "syn_L3_00042_en_casual",
  "instruction": "Convert the following natural language description into Signal DSL configuration.",
  "input": "I need a setup where math questions get routed to DeepSeek R1 with reasoning enabled.",
  "output": "SIGNAL domain math {\n  description: \"Mathematics\"\n  mmlu_categories: [\"math\"]\n}\n\nROUTE math_route {\n  PRIORITY 100\n  WHEN domain(\"math\")\n  MODEL \"deepseek-r1\" (reasoning = true)\n}",
  "style": "en_casual",
  "complexity": "L3"
}
```

### 7.3 Stage 3: DPO 偏好数据

```jsonc
// stage3_dpo.jsonl - (prompt, chosen, rejected) 三元组
{
  "id": "neg_syntax_error_typo_abc123",
  "prompt": "Generate a valid Signal DSL configuration.",
  "chosen": "SIGNAL domain math {\n  description: \"Mathematics\"\n}\n\nROUTE math_route {\n  PRIORITY 100\n  WHEN domain(\"math\")\n  MODEL \"deepseek-r1\"\n}",
  "rejected": "SINGAL domain math {\n  description: \"Mathematics\"\n}\n\nROUTE math_route {\n  PRIORITY 100\n  WHEN domain(\"math\")\n  MODEL \"deepseek-r1\"\n}",
  "mutation_type": "typo_keyword",
  "mutation_category": "syntax_error"
}
```

### 7.4 评估基准集

```jsonc
// eval_benchmark.jsonl - 带有完整元数据
{
  "id": "bench_L3_001",
  "instruction": "Convert the following natural language description into Signal DSL configuration.",
  "input": "Create a 3-tier routing setup with math going to DeepSeek, coding to GPT-4o, and everything else to Qwen.",
  "output": "...(expected DSL)...",
  "complexity": "L3",
  "style": "en_casual",
  "expected_signal_types": ["domain"],
  "expected_route_count": 3
}
```

## 8. 质量保证

### 8.1 自动化验证

所有生成数据通过 Go 编译器验证:

```bash
# 验证单个 DSL
echo "$DSL" | go run ./cmd/dsl-parser --validate --json

# 输出格式
{
  "syntax_valid": true,
  "semantic_valid": true,
  "compile_valid": true,
  "errors": [],
  "warnings": ["SIGNAL jailbreak not referenced in any WHEN clause"]
}
```

### 8.2 人工抽检

| 阶段 | 抽检比例 | 关注点 |
|:---|:---|:---|
| NL 描述 | 1% (~600 样本) | 描述准确性、语言自然度 |
| 负样本 | 2% (~100 样本) | 变异有效性、错误类型正确 |
| 最终数据 | 0.5% (~375 样本) | 整体质量、标签正确性 |

### 8.3 分布检查

```python
# 确保各维度分布合理
assert abs(count_by_complexity['L3'] / total - 0.35) < 0.05
assert abs(count_by_style['zh_formal'] / total - 1/6) < 0.05
assert count_signal_type_coverage >= 0.9  # 90%+ 信号类型覆盖
```

## 9. 使用指南

### 9.1 快速开始

```bash
cd ml-binding/dsl-model-training/data

# 完整流程 (需要 OpenAI API key)
export OPENAI_API_KEY="sk-..."
make all

# 或者使用本地 NL 生成 (无需 API)
make all NL_API=local

# 快速测试 (小数据量)
make quick-test
```

### 9.2 分步执行

```bash
# 1. 提取种子数据
make seeds

# 2. 生成合成 DSL (可调整数量)
make synthetic SYNTHETIC_COUNT=5000

# 3. 生成 NL 描述
make nl-pairs NL_API=openai NL_MODEL=gpt-4o
# 或 dry-run 模式
make nl-pairs-dry-run

# 4. 生成负样本
make negative NEGATIVE_RATIO=0.5

# 5. 验证并构建最终数据集
make validate

# 6. 查看统计
make stats
```

### 9.3 输出文件

```
final/
├── stage1_syntax_pt.jsonl    # Stage 1: ~10,000 样本
├── stage2_sft.jsonl          # Stage 2: ~60,000 样本
├── stage3_dpo.jsonl          # Stage 3: ~5,000 样本
├── eval_benchmark.jsonl      # 评估集: ~200 样本
└── dataset_stats.json        # 统计报告
```

## 10. 后续扩展

### 10.1 增量数据收集

```python
# 从生产日志收集真实 NL 输入
def collect_production_data(logs_path: Path) -> list[dict]:
    # 解析用户输入日志
    # 过滤成功转换的案例
    # 提取 (NL, DSL) 对
    pass
```

### 10.2 主动学习

```python
# 识别模型不确定的样本，请求人工标注
def active_learning_query(model, unlabeled_pool, k: int) -> list[dict]:
    uncertainties = []
    for sample in unlabeled_pool:
        outputs = model.generate(sample, num_return_sequences=5, temperature=0.8)
        # 计算输出多样性作为不确定性代理
        diversity = calculate_diversity(outputs)
        uncertainties.append((sample, diversity))
    
    # 返回 top-k 最不确定的样本
    return [s for s, _ in sorted(uncertainties, key=lambda x: -x[1])[:k]]
```

### 10.3 数据增强

- **同义替换**: 替换 NL 描述中的同义词
- **回译**: 英→中→英 / 中→英→中
- **Paraphrase**: 使用 LLM 改写 NL 描述
- **DSL 等价变换**: 重排字段顺序、调整格式化
