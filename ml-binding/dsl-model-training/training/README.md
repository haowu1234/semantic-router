# DSL Model Training Framework

三阶段训练框架，用于训练 Router DSL 专用生成模型。

## 训练阶段

| Stage | 名称 | 目标 | 数据 |
|-------|------|------|------|
| **Stage 1** | Syntax Pre-training | 学习 DSL 语法结构 | `stage1_syntax_pt.jsonl` |
| **Stage 2** | SFT (NL→DSL) | 学习自然语言到 DSL 的映射 | `stage2_sft.jsonl` |
| **Stage 3** | DPO | 偏好对齐，避免常见错误 | `stage3_dpo.jsonl` |

## 目录结构

```
training/
├── README.md                 # 本文档
├── requirements.txt          # Python 依赖
├── configs/                  # 训练配置
│   ├── base.yaml            # 基础配置
│   ├── stage1_pt.yaml       # Stage 1 配置
│   ├── stage2_sft.yaml      # Stage 2 配置
│   └── stage3_dpo.yaml      # Stage 3 配置
├── src/
│   ├── __init__.py
│   ├── data/                # 数据加载
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset 类
│   │   └── collator.py      # Data Collator
│   ├── models/              # 模型相关
│   │   ├── __init__.py
│   │   └── dsl_model.py     # 模型封装
│   ├── trainers/            # 训练器
│   │   ├── __init__.py
│   │   ├── base_trainer.py  # 基础训练器
│   │   ├── sft_trainer.py   # SFT 训练器
│   │   └── dpo_trainer.py   # DPO 训练器
│   ├── evaluation/          # 评估
│   │   ├── __init__.py
│   │   ├── metrics.py       # 评估指标
│   │   └── evaluator.py     # 评估器
│   └── utils/               # 工具
│       ├── __init__.py
│       ├── logger.py        # 日志
│       └── config.py        # 配置加载
├── scripts/
│   ├── prepare_data.py      # 数据预处理
│   ├── train_stage1.py      # Stage 1 训练
│   ├── train_stage2.py      # Stage 2 训练
│   ├── train_stage3.py      # Stage 3 训练
│   ├── train_all.py         # 完整训练流程
│   └── evaluate.py          # 评估脚本
├── checkpoints/             # 模型检查点
└── logs/                    # 训练日志
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
python scripts/prepare_data.py \
    --data-dir ../data \
    --output-dir ./prepared_data

# 3. 完整训练 (3 阶段)
python scripts/train_all.py \
    --config configs/base.yaml \
    --data-dir ./prepared_data \
    --output-dir ./checkpoints

# 或分阶段训练
python scripts/train_stage1.py --config configs/stage1_pt.yaml
python scripts/train_stage2.py --config configs/stage2_sft.yaml
python scripts/train_stage3.py --config configs/stage3_dpo.yaml

# 4. 评估
python scripts/evaluate.py \
    --model ./checkpoints/final \
    --eval-data ./prepared_data/eval_benchmark.jsonl
```

## 配置说明

### 基础模型选择

推荐使用以下基础模型：

| 模型 | 参数量 | 显存需求 | 推荐场景 |
|------|--------|---------|---------|
| Qwen2.5-1.5B | 1.5B | ~8GB | 快速实验 |
| Qwen2.5-3B | 3B | ~12GB | 平衡选择 |
| Qwen2.5-7B | 7B | ~24GB | 生产推荐 |
| DeepSeek-Coder-1.3B | 1.3B | ~6GB | 代码专长 |

### 训练参数

```yaml
# configs/base.yaml
model:
  name: "Qwen/Qwen2.5-3B"
  max_length: 2048
  
training:
  per_device_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  num_epochs: 3
  warmup_ratio: 0.1
  
lora:
  enabled: true
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

## 评估指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **Syntax Accuracy** | 语法正确率 | > 95% |
| **Semantic Accuracy** | 语义正确率 | > 90% |
| **Compile Success** | 编译成功率 | > 90% |
| **BLEU** | 文本相似度 | > 0.7 |
| **Exact Match** | 完全匹配率 | > 60% |
