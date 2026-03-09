# DSL Model Training - Data Engineering

本目录包含 Router DSL 专用生成模型的训练数据构建工具和数据集。

## 目录结构

```
data/
├── README.md                      # 本文档
├── scripts/                       # 数据生成脚本
│   ├── extract_seeds.py          # 从测试文件/YAML提取种子DSL
│   ├── cfg_generator.py          # CFG随机游走生成器
│   ├── nl_generator.py           # NL描述自动生成
│   ├── negative_sampler.py       # 负样本生成
│   ├── validator.py              # 数据验证管线
│   └── split_dataset.py          # 数据集划分
├── schemas/                       # DSL Schema 定义
│   ├── signal_schemas.json       # 各信号类型字段Schema
│   ├── plugin_schemas.json       # 插件类型字段Schema
│   └── algorithm_schemas.json    # 算法类型字段Schema
├── seeds/                         # 种子数据
│   ├── from_tests.jsonl          # 从dsl_test.go提取
│   ├── from_yaml.jsonl           # 从config/*.yaml反编译
│   └── from_guide.jsonl          # 从DslGuide.tsx提取
├── synthetic/                     # 合成数据
│   ├── L1_simple.jsonl           # 复杂度L1
│   ├── L2_basic.jsonl            # 复杂度L2
│   ├── L3_standard.jsonl         # 复杂度L3
│   ├── L4_advanced.jsonl         # 复杂度L4
│   └── L5_production.jsonl       # 复杂度L5
├── nl_pairs/                      # NL-DSL配对数据
│   ├── en_formal.jsonl           # 英文正式
│   ├── en_casual.jsonl           # 英文口语
│   ├── en_technical.jsonl        # 英文技术
│   ├── zh_formal.jsonl           # 中文正式
│   ├── zh_casual.jsonl           # 中文口语
│   └── ambiguous.jsonl           # 模糊描述
├── negative/                      # 负样本 (DPO用)
│   ├── syntax_errors.jsonl       # 语法错误
│   ├── reference_errors.jsonl    # 引用错误
│   ├── constraint_violations.jsonl # 约束违反
│   └── schema_mismatches.jsonl   # Schema不匹配
├── final/                         # 最终训练数据
│   ├── stage1_syntax_pt.jsonl    # Stage1: 纯DSL语法预训练
│   ├── stage2_sft.jsonl          # Stage2: NL→DSL SFT
│   ├── stage3_dpo.jsonl          # Stage3: DPO偏好数据
│   └── eval_benchmark.jsonl      # 评估基准集
└── stats/                         # 数据统计
    └── dataset_report.json       # 数据集统计报告
```

## 数据量目标

| 数据来源 | 方法 | 目标数量 | 质量等级 |
|:---|:---|:---|:---|
| **种子数据** | 测试文件+YAML反编译 | ~250 | ⭐⭐⭐⭐⭐ 金标准 |
| **CFG合成** | BNF随机游走 | ~10,000 | ⭐⭐⭐⭐ 语法保证 |
| **NL配对** | GPT-4o生成6种风格 | ~60,000 | ⭐⭐⭐ 需抽检 |
| **负样本** | 变异生成 | ~5,000 | ⭐⭐⭐⭐ DPO用 |
| **总计** | | **~75,000** | |

## 复杂度分级

| 级别 | 构造数 | 条件深度 | 示例场景 |
|:---|:---|:---|:---|
| **L1** | 2-3 | 0 | "Route all to GPT-4o" |
| **L2** | 4-6 | 1 | "Math→DeepSeek, coding→GPT-4o" |
| **L3** | 6-10 | 2 | "3-tier + jailbreak blocking" |
| **L4** | 10-15 | 3+ | "RBAC + modality + LoRA + cascade" |
| **L5** | 15+ | 4+ | 完整生产级配置 |

## 使用方法

```bash
# 1. 提取种子数据
python scripts/extract_seeds.py --output seeds/

# 2. 生成合成DSL
python scripts/cfg_generator.py --count 10000 --output synthetic/

# 3. 生成NL描述
python scripts/nl_generator.py --input synthetic/ --output nl_pairs/

# 4. 生成负样本
python scripts/negative_sampler.py --input synthetic/ --output negative/

# 5. 验证并构建最终数据集
python scripts/validator.py --input . --output final/

# 6. 生成统计报告
python scripts/split_dataset.py --input final/ --output stats/
```
