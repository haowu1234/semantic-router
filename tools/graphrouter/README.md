# GraphRouter 训练工具

基于图神经网络 (GNN) 的模型选择算法训练工具，用于 semantic-router 的 GraphSelector。

## 概述

GraphRouter 将路由问题建模为 Query-LLM 二分图，通过 GNN 学习 Query 与 LLM 之间的关系，为每个查询选择最优的大语言模型。

### 核心优势

- **不依赖客户端反馈**: 离线训练完成后，推理时无需实时反馈
- **关系建模能力强**: GNN 能捕捉 Query-LLM 之间的复杂关系
- **归纳式推理**: 对新 Query 可直接推理，无需重新训练

## 快速开始

### 1. 安装依赖

```bash
cd tools/graphrouter
pip install -r requirements.txt
```

### 2. 准备数据

#### 方法 A: 生成模拟数据（测试用）

```bash
python data_prepare.py mock --output ./data --num-queries 1000 --num-llms 5
```

#### 方法 B: 从 LLMRouter 项目转换

```bash
python data_prepare.py llmrouter --input /path/to/LLMRouter/data --output ./data
```

#### 方法 C: 从 CSV 转换

```bash
python data_prepare.py csv --input routing_data.csv --output ./data
python data_prepare.py embed --input ./data/routing_data.jsonl --output ./data/query_embeddings.pt
```

### 3. 训练模型

```bash
python train.py \
    --config data/config.yaml \
    --output ../../config/models/graph_model.json \
    --checkpoint models/graphrouter.pt
```

### 4. 评测模型

```bash
python evaluate.py \
    --model ../../config/models/graph_model.json \
    --test-data ./data/routing_data.jsonl \
    --embeddings ./data/query_embeddings.pt
```

## 数据格式

### routing_data.jsonl

路由训练数据，每行一个 JSON 对象：

```jsonl
{"query": "什么是量子计算？", "model_name": "gpt-4", "performance": 0.92, "embedding_id": 0}
{"query": "什么是量子计算？", "model_name": "claude-3", "performance": 0.88, "embedding_id": 0}
{"query": "写一个快排算法", "model_name": "gpt-4", "performance": 0.95, "embedding_id": 1}
```

### query_embeddings.pt

Query Embedding 张量，PyTorch 格式：

```python
# Shape: [num_unique_queries, embedding_dim]
embeddings = torch.load("query_embeddings.pt")  # e.g., [10000, 768]
```

### llm_candidates.json

LLM 配置：

```json
{
  "gpt-4": {
    "model": "gpt-4-turbo",
    "embedding": [0.1, 0.2, ...],
    "cost_per_1k_tokens": 0.03
  },
  "claude-3": {
    "model": "claude-3-opus",
    "embedding": [0.15, 0.25, ...],
    "cost_per_1k_tokens": 0.015
  }
}
```

## 配置说明

完整配置示例见 `configs/default.yaml`：

```yaml
data_path:
  routing_data_train: 'data/routing_data.jsonl'
  query_embedding_data: 'data/query_embeddings.pt'
  llm_data: 'data/llm_candidates.json'

hparam:
  hidden_dim: 64           # 隐藏层维度
  learning_rate: 0.001     # 学习率
  weight_decay: 0.0001     # L2 正则化
  train_epoch: 100         # 训练轮数
  batch_size: 4            # 批大小
  train_mask_rate: 0.3     # 边掩码率
  val_split_ratio: 0.2     # 验证集比例
  temperature: 1.0         # Softmax 温度
```

## 导出格式

训练完成后导出的 `graph_model.json` 供 Go 推理使用：

```json
{
  "version": "1.0",
  "model_names": ["gpt-4", "claude-3", "qwen-2"],
  "hidden_dim": 64,
  "query_dim": 768,
  "query_projection": [[...], ...],
  "llm_representations": [[...], ...],
  "temperature": 1.0,
  "metadata": {
    "train_samples": 10000,
    "train_accuracy": 0.85,
    "export_date": "2026-01-16T10:30:00"
  }
}
```

## Go 推理逻辑

```go
// 1. 投影 Query 到隐空间
query_hidden = query_embedding @ query_projection

// 2. 计算与各 LLM 的余弦相似度
for i, llm_rep := range llm_representations {
    scores[i] = cosine_similarity(query_hidden, llm_rep)
}

// 3. Softmax 选择
probs = softmax(scores / temperature)
selected = model_names[argmax(probs)]
```

## 模块说明

| 文件 | 说明 |
|------|------|
| `model.py` | GNN 模型定义 (FeatureAlign, EncoderDecoderNet, GNNPredictor) |
| `data.py` | 数据加载和预处理 |
| `trainer.py` | 训练器 |
| `export.py` | 导出为 Go JSON 格式 |
| `train.py` | 训练入口脚本 |
| `evaluate.py` | 评测脚本 |
| `data_prepare.py` | 数据准备工具 |

## 参考

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [LLMRouter](https://github.com/vllm-project/llm-router)
