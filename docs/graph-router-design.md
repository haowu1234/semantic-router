# GraphRouter 设计方案

> 基于图神经网络的智能模型选择算法，为 semantic-router 实现高精度路由决策

## 目录

- [1. 概述](#1-概述)
- [2. 算法原理](#2-算法原理)
- [3. 架构设计](#3-架构设计)
- [4. 数据流程](#4-数据流程)
- [5. Go 实现方案](#5-go-实现方案)
- [6. Python 训练工具](#6-python-训练工具)
- [7. 配置说明](#7-配置说明)
- [8. 评测方案](#8-评测方案)
- [9. 部署指南](#9-部署指南)
- [10. 路线图](#10-路线图)

---

## 1. 概述

### 1.1 背景

GraphRouter 是一种基于图神经网络（GNN）的模型选择算法，源自 [LLMRouter](https://github.com/vllm-project/llm-router) 项目。它通过学习 Query 与 LLM 之间的关系图，为每个查询选择最优的大语言模型。

### 1.2 核心优势

| 特性 | 说明 |
|------|------|
| **不依赖客户端反馈** | 离线训练完成后，推理时无需实时反馈 |
| **关系建模能力强** | GNN 能捕捉 Query-LLM 之间的复杂关系 |
| **归纳式推理** | 对新 Query 可直接推理，无需重新训练 |
| **可解释性** | 通过边权重可理解模型选择逻辑 |

### 1.3 适用场景

- 多模型部署环境，需要智能分流
- 有历史 Query-Model-Performance 数据
- 对选择精度要求高，可接受离线训练成本

---

## 2. 算法原理

### 2.1 图结构定义

GraphRouter 将路由问题建模为**异构二分图**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Query-LLM 二分图                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query Nodes (Q)              Edge (Performance)    LLM Nodes (L)
│  ┌───┐                           ┌───┐              ┌───┐      │
│  │Q1 │─────────────0.85──────────│   │──────────────│L1 │ GPT-4│
│  └───┘                           │   │              └───┘      │
│  ┌───┐                           │ E │              ┌───┐      │
│  │Q2 │─────────────0.72──────────│   │──────────────│L2 │Claude│
│  └───┘                           │ d │              └───┘      │
│  ┌───┐                           │ g │              ┌───┐      │
│  │Q3 │─────────────0.91──────────│ e │──────────────│L3 │Qwen  │
│  └───┘                           │   │              └───┘      │
│  ┌───┐                           │   │              ┌───┐      │
│  │...│                           │   │              │...│      │
│  └───┘                           └───┘              └───┘      │
│                                                                 │
│  特征: Query Embedding          权重: Performance Score         │
│        (768 维向量)                  (0.0 - 1.0)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 节点与边定义

| 元素 | 描述 | 特征维度 |
|------|------|---------|
| **Query Node** | 用户查询 | Embedding 向量 (768d) |
| **LLM Node** | 候选模型 | 模型能力向量 (768d) |
| **Edge** | Query→LLM 连接 | Performance Score (1d) |

### 2.3 GNN 消息传递

GraphRouter 使用 2 层 `GeneralConv` 进行消息传递：

```
1. 特征对齐层 (Feature Alignment)
   Query Features [768] → Linear → [64]
   LLM Features   [768] → Linear → [64]
   
2. 第一层 GNN 卷积
   x₁ = LeakyReLU(BN(Conv1(x₀, edges, edge_weights)))
   
3. 第二层 GNN 卷积
   x₂ = BN(Conv2(x₁, edges, edge_weights))

4. 边预测 (Edge Prediction)
   score = Sigmoid(mean(x₀[query] * x₂[llm]))
```

### 2.4 训练目标

**损失函数**: Binary Cross Entropy (BCE)

```python
# 标签: 对于每个 Query，最佳 LLM 的边标为 1，其他为 0
label = one_hot(argmax(performance_matrix, axis=1))

# 损失计算
loss = BCE(predicted_edges, label)
```

### 2.5 推理流程

```
新 Query → Embedding → 添加到图中 → GNN 消息传递 → 预测边分数 → 选择最高分 LLM
                                          ↑
                              历史训练数据构成的图作为上下文
```

---

## 3. 架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GraphRouter 系统架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Training Pipeline (Python)                   │   │
│  │                                                                      │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │   │
│  │  │  数据准备  │───▶│ 图构建   │───▶│ GNN训练  │───▶│ 权重导出  │      │   │
│  │  │          │    │          │    │          │    │          │      │   │
│  │  │- JSONL   │    │- 节点特征│    │- PyTorch │    │- JSON    │      │   │
│  │  │- Embed   │    │- 边索引  │    │- PyG     │    │- 预计算   │      │   │
│  │  │- LLM Info│    │- 边权重  │    │          │    │          │      │   │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                        ┌──────────────────────┐                            │
│                        │   model_weights.json  │                            │
│                        │   (预计算 LLM 表示)    │                            │
│                        └──────────────────────┘                            │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Inference Pipeline (Go)                      │   │
│  │                                                                      │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │   │
│  │  │ Query    │───▶│Embedding │───▶│ 投影计算  │───▶│ 模型选择  │      │   │
│  │  │          │    │(Candle)  │    │          │    │          │      │   │
│  │  │用户请求   │    │Qwen3/BERT│    │矩阵乘法   │    │Softmax   │      │   │
│  │  │          │    │          │    │相似度     │    │Top-1     │      │   │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Go 推理架构（轻量级方案）

由于 Go 没有成熟的 GNN 库，采用**预计算 + 相似度匹配**的方案：

```
┌─────────────────────────────────────────────────────────────────┐
│                    GraphSelector (Go)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  预加载数据:                                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • LLM Representations [num_llms, hidden_dim]               │ │
│  │   - 通过 Python 训练时预计算的 LLM 聚合表示                   │ │
│  │                                                             │ │
│  │ • Query Projection Matrix [query_dim, hidden_dim]          │ │
│  │   - 将 Query Embedding 投影到与 LLM 相同的隐空间             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  推理流程:                                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  query_embedding ──▶ query_hidden = matmul(query, W_proj)  │ │
│  │                                                             │ │
│  │  for each llm_rep in llm_representations:                  │ │
│  │      scores[i] = cosine_similarity(query_hidden, llm_rep)  │ │
│  │                                                             │ │
│  │  probs = softmax(scores / temperature)                     │ │
│  │  selected = model_names[argmax(probs)]                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 模块结构

```
pkg/selection/
├── selector.go           # Selector 接口定义 (已存在)
├── factory.go            # Selector 工厂函数 (已存在)
├── knn.go                # KNNSelector 实现 (已存在)
├── static.go             # StaticSelector 实现 (已存在)
├── graph.go              # GraphSelector 核心实现 ⭐ 新增
├── graph_model.go        # 模型权重和数据结构 ⭐ 新增
├── graph_math.go         # 矩阵运算工具函数 ⭐ 新增
└── graph_test.go         # 单元测试 ⭐ 新增

tools/graphrouter/
├── train.py              # 训练脚本 ⭐ 新增
├── export.py             # 权重导出到 JSON ⭐ 新增
├── evaluate.py           # 评测脚本 ⭐ 新增
├── data_prepare.py       # 数据准备工具 ⭐ 新增
├── requirements.txt      # Python 依赖 ⭐ 新增
└── README.md             # 使用说明 ⭐ 新增
```

---

## 4. 数据流程

### 4.1 训练数据格式

**routing_data.jsonl** - 路由训练数据:

```jsonl
{"query": "什么是量子计算？", "model_name": "gpt-4", "performance": 0.92, "embedding_id": 0}
{"query": "什么是量子计算？", "model_name": "claude-3", "performance": 0.88, "embedding_id": 0}
{"query": "什么是量子计算？", "model_name": "qwen-2", "performance": 0.85, "embedding_id": 0}
{"query": "写一个快排算法", "model_name": "gpt-4", "performance": 0.95, "embedding_id": 1}
```

**query_embeddings.pt** - Query Embedding 张量:

```python
# Shape: [num_unique_queries, embedding_dim]
embeddings = torch.load("query_embeddings.pt")  # [10000, 768]
```

**llm_candidates.json** - 模型配置:

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

### 4.2 数据获取方案

#### 方案 A：使用公开 Benchmark（冷启动推荐）

```bash
# 使用 LLMRouter 数据生成
cd LLMRouter && pip install -e .
llmrouter generate --step 1 --task mmlu,gsm8k,humaneval  # 准备数据
llmrouter generate --step 2 --task mmlu,gsm8k,humaneval  # 生成 embedding
llmrouter generate --step 3 --task mmlu,gsm8k,humaneval  # 调用 API
llmrouter generate --step 4 --task mmlu,gsm8k,humaneval  # 合并
```

#### 方案 B：从线上流量采集

```yaml
# 流量采集配置
data_collection:
  sample_rate: 0.1
  scoring:
    method: "llm_judge"
    judge_model: "gpt-4"
```

#### 方案 C：转换已有 KNN 数据

```python
# 复用 KNNSelector 训练数据
with open("knn_model.json") as f:
    knn_data = json.load(f)
# 转换为 GraphRouter 格式...
```

### 4.3 推理时导出格式

**graph_model.json** - Go 推理使用:

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
    "train_date": "2026-01-16"
  }
}
```

---

## 5. Go 实现方案

### 5.1 配置结构

```go
// GraphConfig 图路由器配置
type GraphConfig struct {
    // ModelPath 预训练模型 JSON 路径
    ModelPath string `yaml:"model_path" json:"model_path"`
    
    // Temperature Softmax 温度参数，越高选择越随机
    Temperature float32 `yaml:"temperature" json:"temperature"`
    
    // MinConfidence 最小置信度阈值
    MinConfidence float32 `yaml:"min_confidence" json:"min_confidence"`
    
    // FallbackModel 置信度不足时的回退模型
    FallbackModel string `yaml:"fallback_model" json:"fallback_model"`
}
```

### 5.2 核心实现

```go
// GraphSelector 基于图神经网络的模型选择器
type GraphSelector struct {
    mu     sync.RWMutex
    config GraphConfig

    // 预加载的模型数据
    modelNames          []string      // 模型名称列表
    queryProjection     [][]float32   // Query 投影矩阵 [query_dim, hidden_dim]
    llmRepresentations  [][]float32   // LLM 表示 [num_llms, hidden_dim]
    
    hiddenDim int
    queryDim  int
}

// NewGraphSelector 创建图选择器
func NewGraphSelector(config GraphConfig) *GraphSelector {
    if config.Temperature <= 0 {
        config.Temperature = 1.0
    }
    return &GraphSelector{
        config: config,
    }
}

// LoadFromFile 从 JSON 文件加载模型
func (s *GraphSelector) LoadFromFile(modelPath string) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    data, err := os.ReadFile(modelPath)
    if err != nil {
        return fmt.Errorf("failed to read model file: %w", err)
    }

    var modelData GraphModelData
    if err := json.Unmarshal(data, &modelData); err != nil {
        return fmt.Errorf("failed to parse model file: %w", err)
    }

    s.modelNames = modelData.ModelNames
    s.queryProjection = modelData.QueryProjection
    s.llmRepresentations = modelData.LLMRepresentations
    s.hiddenDim = modelData.HiddenDim
    s.queryDim = modelData.QueryDim

    logging.Infof("[GraphSelector] Loaded model with %d LLMs, hidden_dim=%d",
        len(s.modelNames), s.hiddenDim)
    return nil
}

// Select 选择最佳模型
func (s *GraphSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()

    if len(selCtx.QueryEmbedding) == 0 {
        return nil, fmt.Errorf("GraphSelector requires query embedding")
    }
    if len(s.llmRepresentations) == 0 {
        return nil, fmt.Errorf("GraphSelector model not loaded")
    }

    // 1. 投影 Query 到隐空间
    queryHidden := s.projectQuery(selCtx.QueryEmbedding)

    // 2. 计算与各 LLM 的相似度
    scores := make([]float32, len(s.llmRepresentations))
    for i, llmRep := range s.llmRepresentations {
        scores[i] = cosineSimilarity(queryHidden, llmRep)
    }

    // 3. Softmax 归一化
    probs := softmax(scores, s.config.Temperature)

    // 4. 选择最高分模型
    bestIdx := argmax(probs)
    
    // 5. 过滤候选列表
    selectedModel := s.modelNames[bestIdx]
    if len(selCtx.Candidates) > 0 && !contains(selCtx.Candidates, selectedModel) {
        // 从候选列表中选最高分的
        bestIdx, selectedModel = s.selectFromCandidates(probs, selCtx.Candidates)
    }

    // 6. 置信度检查
    confidence := probs[bestIdx]
    if confidence < s.config.MinConfidence && s.config.FallbackModel != "" {
        selectedModel = s.config.FallbackModel
        confidence = s.config.MinConfidence
    }

    // 构建分数映射
    scoreMap := make(map[string]float32)
    for i, name := range s.modelNames {
        scoreMap[name] = probs[i]
    }

    return &SelectionResult{
        SelectedModel: selectedModel,
        Confidence:    confidence,
        Scores:        scoreMap,
        SelectorName:  s.Name(),
        Metadata: map[string]interface{}{
            "hidden_dim":  s.hiddenDim,
            "temperature": s.config.Temperature,
        },
    }, nil
}

// projectQuery 将 Query Embedding 投影到隐空间
func (s *GraphSelector) projectQuery(queryEmb []float32) []float32 {
    // queryHidden = queryEmb @ queryProjection
    // [1, query_dim] @ [query_dim, hidden_dim] = [1, hidden_dim]
    result := make([]float32, s.hiddenDim)
    for j := 0; j < s.hiddenDim; j++ {
        var sum float32
        for i := 0; i < len(queryEmb) && i < len(s.queryProjection); i++ {
            sum += queryEmb[i] * s.queryProjection[i][j]
        }
        result[j] = sum
    }
    return result
}

func (s *GraphSelector) Name() string {
    return "graph"
}

func (s *GraphSelector) Update(ctx context.Context, feedback *SelectionFeedback) error {
    // GraphSelector 不支持在线更新，需要重新训练
    return nil
}
```

### 5.3 数学工具函数

```go
// graph_math.go

// softmax 计算 softmax 概率分布
func softmax(scores []float32, temperature float32) []float32 {
    if len(scores) == 0 {
        return nil
    }
    
    // 找最大值（数值稳定性）
    maxScore := scores[0]
    for _, s := range scores[1:] {
        if s > maxScore {
            maxScore = s
        }
    }
    
    // 计算 exp 和总和
    expScores := make([]float32, len(scores))
    var sumExp float32
    for i, s := range scores {
        expScores[i] = float32(math.Exp(float64((s - maxScore) / temperature)))
        sumExp += expScores[i]
    }
    
    // 归一化
    probs := make([]float32, len(scores))
    for i := range expScores {
        probs[i] = expScores[i] / sumExp
    }
    return probs
}

// argmax 返回最大值索引
func argmax(values []float32) int {
    if len(values) == 0 {
        return -1
    }
    maxIdx := 0
    maxVal := values[0]
    for i, v := range values[1:] {
        if v > maxVal {
            maxVal = v
            maxIdx = i + 1
        }
    }
    return maxIdx
}

// cosineSimilarity 计算余弦相似度（复用 knn.go 中的实现）
```

### 5.4 工厂注册

```go
// factory.go 中添加

case "graph":
    logging.Infof("[Selection] Initializing Graph selector")
    selector := NewGraphSelector(GraphConfig{
        ModelPath:     config.Graph.ModelPath,
        Temperature:   config.Graph.Temperature,
        MinConfidence: config.Graph.MinConfidence,
        FallbackModel: config.Graph.FallbackModel,
    })
    if config.Graph.ModelPath != "" {
        if err := selector.LoadFromFile(config.Graph.ModelPath); err != nil {
            return nil, fmt.Errorf("failed to load Graph model: %w", err)
        }
    }
    return selector, nil
```

---

## 6. Python 训练工具

> **注意**: 训练工具完全独立实现，不依赖 LLMRouter 项目，确保两个项目解耦。

### 6.1 目录结构

```
tools/graphrouter/
├── __init__.py
├── train.py              # 训练入口脚本
├── model.py              # GNN 模型定义
├── data.py               # 数据加载和处理
├── trainer.py            # 训练器
├── export.py             # 导出为 Go JSON
├── evaluate.py           # 评测脚本
├── configs/
│   └── default.yaml      # 默认配置
├── requirements.txt      # Python 依赖
└── README.md             # 使用说明
```

### 6.2 GNN 模型定义

```python
# tools/graphrouter/model.py
"""
GraphRouter GNN 模型定义
独立实现，不依赖外部路由器项目
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv
from torch_geometric.data import Data


class FeatureAlign(nn.Module):
    """特征对齐层：将 Query 和 LLM 特征映射到相同维度"""
    
    def __init__(self, query_dim: int, llm_dim: int, hidden_dim: int):
        super().__init__()
        self.query_transform = nn.Linear(query_dim, hidden_dim)
        self.llm_transform = nn.Linear(llm_dim, hidden_dim)
    
    def forward(self, query_features: torch.Tensor, llm_features: torch.Tensor) -> torch.Tensor:
        aligned_query = self.query_transform(query_features)
        aligned_llm = self.llm_transform(llm_features)
        return torch.cat([aligned_query, aligned_llm], dim=0)


class GraphRouterModel(nn.Module):
    """
    GraphRouter GNN 模型
    
    架构:
    1. FeatureAlign: 对齐 Query/LLM 特征到隐空间
    2. GeneralConv x 2: 两层图卷积进行消息传递
    3. EdgePredictor: 预测 Query-LLM 边的分数
    """
    
    def __init__(
        self,
        query_dim: int = 768,
        llm_dim: int = 768,
        hidden_dim: int = 64,
        edge_dim: int = 1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 特征对齐
        self.align = FeatureAlign(query_dim, llm_dim, hidden_dim)
        
        # GNN 卷积层
        self.conv1 = GeneralConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            in_edge_channels=edge_dim
        )
        self.conv2 = GeneralConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            in_edge_channels=edge_dim
        )
        
        # BatchNorm
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # 边权重变换
        self.edge_mlp = nn.Linear(edge_dim, edge_dim)
    
    def forward(
        self,
        query_features: torch.Tensor,
        llm_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
        visible_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query_features: [num_queries, query_dim]
            llm_features: [num_llms, llm_dim]
            edge_index: [2, num_edges] 边索引
            edge_attr: [num_edges, 1] 边权重 (performance)
            edge_mask: [num_edges] 需要预测的边
            visible_mask: [num_edges] 可见的边（用于消息传递）
        
        Returns:
            predicted_scores: [num_masked_edges] 预测分数
        """
        # 可见边的索引和权重
        visible_edge_index = edge_index[:, visible_mask]
        visible_edge_attr = edge_attr[visible_mask]
        
        # 预测边的索引
        predict_edge_index = edge_index[:, edge_mask]
        
        # 边权重变换
        visible_edge_attr = F.leaky_relu(
            self.edge_mlp(visible_edge_attr.view(-1, 1))
        )
        
        # 特征对齐
        x0 = self.align(query_features, llm_features)
        
        # GNN 消息传递
        x1 = F.leaky_relu(self.bn1(
            self.conv1(x0, visible_edge_index, edge_attr=visible_edge_attr)
        ))
        x2 = self.bn2(
            self.conv2(x1, visible_edge_index, edge_attr=visible_edge_attr)
        )
        
        # 边预测：源节点初始特征 * 目标节点 GNN 特征
        src_features = x0[predict_edge_index[0]]
        dst_features = x2[predict_edge_index[1]]
        scores = torch.sigmoid((src_features * dst_features).mean(dim=-1))
        
        return scores


class GraphDataBuilder:
    """构建 PyG Data 对象"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def build(
        self,
        query_embeddings: torch.Tensor,
        llm_embeddings: torch.Tensor,
        performance_matrix: torch.Tensor,
        train_mask: torch.Tensor = None,
        val_mask: torch.Tensor = None
    ) -> Data:
        """
        构建图数据
        
        Args:
            query_embeddings: [num_queries, query_dim]
            llm_embeddings: [num_llms, llm_dim]
            performance_matrix: [num_queries, num_llms]
            train_mask: [num_edges] 训练边掩码
            val_mask: [num_edges] 验证边掩码
        """
        num_queries = query_embeddings.shape[0]
        num_llms = llm_embeddings.shape[0]
        
        # 构建边索引：每个 query 连接所有 llm
        query_indices = []
        llm_indices = []
        for q in range(num_queries):
            for l in range(num_llms):
                query_indices.append(q)
                llm_indices.append(num_queries + l)  # LLM 节点从 num_queries 开始
        
        edge_index = torch.tensor([query_indices, llm_indices], dtype=torch.long)
        
        # 边权重（performance）
        edge_attr = performance_matrix.flatten().view(-1, 1)
        
        # 标签：每个 query 的最佳 LLM 边为 1
        best_llm_per_query = performance_matrix.argmax(dim=1)
        labels = torch.zeros(num_queries * num_llms)
        for q, best_l in enumerate(best_llm_per_query):
            labels[q * num_llms + best_l] = 1.0
        
        # 默认掩码
        num_edges = num_queries * num_llms
        if train_mask is None:
            train_mask = torch.ones(num_edges, dtype=torch.bool)
        if val_mask is None:
            val_mask = torch.zeros(num_edges, dtype=torch.bool)
        
        return Data(
            query_features=query_embeddings.to(self.device),
            llm_features=llm_embeddings.to(self.device),
            edge_index=edge_index.to(self.device),
            edge_attr=edge_attr.float().to(self.device),
            labels=labels.to(self.device),
            train_mask=train_mask.to(self.device),
            val_mask=val_mask.to(self.device),
            num_queries=num_queries,
            num_llms=num_llms
        )
```

### 6.3 数据加载

```python
# tools/graphrouter/data.py
"""数据加载和预处理"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler


class GraphRouterDataset:
    """GraphRouter 数据集"""
    
    def __init__(
        self,
        routing_data_path: str,
        embedding_path: str,
        llm_config_path: str,
        val_ratio: float = 0.2
    ):
        self.val_ratio = val_ratio
        
        # 加载数据
        self.routing_df = self._load_routing_data(routing_data_path)
        self.query_embeddings = self._load_embeddings(embedding_path)
        self.llm_config = self._load_llm_config(llm_config_path)
        
        # 处理数据
        self._process_data()
    
    def _load_routing_data(self, path: str) -> pd.DataFrame:
        """加载路由训练数据 (JSONL)"""
        records = []
        with open(path, 'r') as f:
            for line in f:
                records.append(json.loads(line.strip()))
        return pd.DataFrame(records)
    
    def _load_embeddings(self, path: str) -> torch.Tensor:
        """加载 Query Embedding"""
        if path.endswith('.pt'):
            return torch.load(path)
        elif path.endswith('.npy'):
            return torch.from_numpy(np.load(path))
        else:
            raise ValueError(f"Unsupported embedding format: {path}")
    
    def _load_llm_config(self, path: str) -> Dict:
        """加载 LLM 配置"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _process_data(self):
        """处理数据，构建训练所需的张量"""
        # 获取模型列表
        self.model_names = self.routing_df["model_name"].unique().tolist()
        self.num_llms = len(self.model_names)
        self.model_to_idx = {name: i for i, name in enumerate(self.model_names)}
        
        # 获取唯一 query
        unique_queries = self.routing_df["query"].unique().tolist()
        self.num_queries = len(unique_queries)
        
        # 提取 query embedding
        embedding_ids = []
        for query in unique_queries:
            query_data = self.routing_df[self.routing_df["query"] == query]
            embedding_ids.append(query_data["embedding_id"].iloc[0])
        
        query_emb_list = [self.query_embeddings[i].numpy() for i in embedding_ids]
        self.query_embedding_matrix = np.array(query_emb_list)
        
        # 构建 performance 矩阵
        self.performance_matrix = np.zeros((self.num_queries, self.num_llms))
        for i, query in enumerate(unique_queries):
            query_data = self.routing_df[self.routing_df["query"] == query]
            for _, row in query_data.iterrows():
                model_idx = self.model_to_idx[row["model_name"]]
                self.performance_matrix[i, model_idx] = row["performance"]
        
        # 归一化 query embedding
        scaler = MinMaxScaler()
        self.query_embedding_matrix = scaler.fit_transform(self.query_embedding_matrix)
        self.query_dim = self.query_embedding_matrix.shape[1]
        
        # 处理 LLM embedding
        self._prepare_llm_embeddings()
    
    def _prepare_llm_embeddings(self):
        """准备 LLM Embedding"""
        llm_embeddings = []
        for model_name in self.model_names:
            if model_name in self.llm_config and "embedding" in self.llm_config[model_name]:
                emb = self.llm_config[model_name]["embedding"]
            else:
                # 随机初始化
                emb = np.random.randn(self.query_dim).tolist()
            llm_embeddings.append(emb)
        
        self.llm_embedding_matrix = np.array(llm_embeddings)
        
        # 归一化
        scaler = MinMaxScaler()
        self.llm_embedding_matrix = scaler.fit_transform(self.llm_embedding_matrix)
        self.llm_dim = self.llm_embedding_matrix.shape[1]
    
    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取训练张量"""
        return (
            torch.from_numpy(self.query_embedding_matrix).float(),
            torch.from_numpy(self.llm_embedding_matrix).float(),
            torch.from_numpy(self.performance_matrix).float()
        )
    
    def get_train_val_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成训练/验证掩码"""
        num_edges = self.num_queries * self.num_llms
        
        # 按 query 划分（同一 query 的所有边要么都在训练集，要么都在验证集）
        num_val_queries = int(self.num_queries * self.val_ratio)
        indices = np.random.permutation(self.num_queries)
        val_query_indices = set(indices[:num_val_queries])
        
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        
        for q in range(self.num_queries):
            start = q * self.num_llms
            end = start + self.num_llms
            if q in val_query_indices:
                val_mask[start:end] = True
            else:
                train_mask[start:end] = True
        
        return train_mask, val_mask
```

### 6.4 训练器

```python
# tools/graphrouter/trainer.py
"""GraphRouter 训练器"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Optional
from pathlib import Path

from .model import GraphRouterModel, GraphDataBuilder
from .data import GraphRouterDataset


class GraphRouterTrainer:
    """GraphRouter 训练器"""
    
    def __init__(
        self,
        dataset: GraphRouterDataset,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 4,
        mask_rate: float = 0.3,
        device: str = None
    ):
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.mask_rate = mask_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.model = GraphRouterModel(
            query_dim=dataset.query_dim,
            llm_dim=dataset.llm_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 构建图数据
        self.data_builder = GraphDataBuilder(self.device)
        self._build_data()
    
    def _build_data(self):
        """构建训练数据"""
        query_emb, llm_emb, perf_matrix = self.dataset.get_tensors()
        train_mask, val_mask = self.dataset.get_train_val_masks()
        
        self.train_data = self.data_builder.build(
            query_emb, llm_emb, perf_matrix, train_mask, val_mask
        )
        self.val_data = self.data_builder.build(
            query_emb, llm_emb, perf_matrix, train_mask, val_mask
        )
    
    def train(self, save_path: Optional[str] = None) -> float:
        """
        训练模型
        
        Returns:
            best_val_result: 最佳验证结果
        """
        best_result = -1.0
        best_state = None
        
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = self._train_epoch()
            
            # 验证阶段
            self.model.eval()
            val_result = self._validate()
            
            # 保存最佳模型
            if val_result > best_result:
                best_result = val_result
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_result={val_result:.4f}")
        
        # 恢复最佳模型
        if best_state:
            self.model.load_state_dict(best_state)
        
        # 保存模型
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        print(f"Training completed. Best validation result: {best_result:.4f}")
        return best_result
    
    def _train_epoch(self) -> float:
        """训练一个 epoch"""
        total_loss = 0.0
        data = self.train_data
        
        for _ in range(self.batch_size):
            # 随机掩码部分训练边
            mask = data.train_mask.clone()
            random_mask = torch.rand(mask.size(), device=self.device) < self.mask_rate
            edge_mask = mask & random_mask
            visible_mask = data.train_mask & ~edge_mask
            
            self.optimizer.zero_grad()
            
            scores = self.model(
                data.query_features,
                data.llm_features,
                data.edge_index,
                data.edge_attr,
                edge_mask,
                visible_mask
            )
            
            labels = data.labels[edge_mask]
            loss = self.criterion(scores, labels)
            total_loss += loss.item()
            
            loss.backward()
        
        self.optimizer.step()
        return total_loss / self.batch_size
    
    def _validate(self) -> float:
        """验证模型"""
        data = self.val_data
        
        with torch.no_grad():
            scores = self.model(
                data.query_features,
                data.llm_features,
                data.edge_index,
                data.edge_attr,
                data.val_mask,
                data.train_mask  # 训练边作为上下文
            )
        
        # 计算验证指标：选择的模型的平均 performance
        num_llms = data.num_llms
        scores = scores.view(-1, num_llms)
        val_perf = data.edge_attr[data.val_mask].view(-1, num_llms)
        
        # 选择得分最高的模型
        selected_idx = scores.argmax(dim=1)
        selected_perf = val_perf[torch.arange(len(selected_idx)), selected_idx]
        
        return selected_perf.mean().item()
```

### 6.5 导出工具

```python
# tools/graphrouter/export.py
"""导出模型为 Go 可用的 JSON 格式"""

import json
import torch
import numpy as np
from typing import Dict
from datetime import datetime

from .model import GraphRouterModel
from .data import GraphRouterDataset


def export_for_go(
    model: GraphRouterModel,
    dataset: GraphRouterDataset,
    output_path: str,
    train_accuracy: float,
    temperature: float = 1.0
):
    """
    导出模型为 Go 可读的 JSON 格式
    
    Args:
        model: 训练好的模型
        dataset: 数据集（用于获取元信息）
        output_path: 输出 JSON 路径
        train_accuracy: 训练准确率
        temperature: Softmax 温度参数
    """
    model.eval()
    
    # 提取 Query 投影矩阵
    # weight shape: [hidden_dim, query_dim]，需要转置为 [query_dim, hidden_dim]
    query_projection = model.align.query_transform.weight.detach().cpu().numpy().T
    
    # 计算 LLM 表示
    llm_representations = compute_llm_representations(model, dataset)
    
    # 构建导出数据
    export_data = {
        "version": "1.0",
        "model_names": dataset.model_names,
        "hidden_dim": model.hidden_dim,
        "query_dim": dataset.query_dim,
        "query_projection": query_projection.tolist(),
        "llm_representations": llm_representations.tolist(),
        "temperature": temperature,
        "metadata": {
            "train_samples": dataset.num_queries,
            "train_accuracy": float(train_accuracy),
            "num_llms": dataset.num_llms,
            "export_date": datetime.now().isoformat()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Model exported to {output_path}")
    print(f"  - {dataset.num_llms} LLMs")
    print(f"  - hidden_dim: {model.hidden_dim}")
    print(f"  - query_dim: {dataset.query_dim}")


def compute_llm_representations(
    model: GraphRouterModel,
    dataset: GraphRouterDataset
) -> np.ndarray:
    """
    计算 LLM 聚合表示
    
    将 LLM embedding 通过特征对齐层，得到隐空间中的表示
    """
    with torch.no_grad():
        query_emb = torch.from_numpy(dataset.query_embedding_matrix).float()
        llm_emb = torch.from_numpy(dataset.llm_embedding_matrix).float()
        
        # 对齐特征
        aligned = model.align(query_emb, llm_emb)
        
        # 提取 LLM 部分
        llm_aligned = aligned[dataset.num_queries:]  # [num_llms, hidden_dim]
        llm_representations = llm_aligned.cpu().numpy()
    
    # L2 归一化
    norms = np.linalg.norm(llm_representations, axis=1, keepdims=True)
    llm_representations = llm_representations / (norms + 1e-8)
    
    return llm_representations
```

### 6.6 训练入口脚本

```python
#!/usr/bin/env python3
# tools/graphrouter/train.py
"""
GraphRouter 训练入口
用法: python train.py --config config.yaml --output model.json
"""

import argparse
import yaml
from pathlib import Path

from data import GraphRouterDataset
from trainer import GraphRouterTrainer
from export import export_for_go


def main():
    parser = argparse.ArgumentParser(description="Train GraphRouter")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--output", required=True, help="Output JSON path for Go")
    parser.add_argument("--checkpoint", default=None, help="PyTorch checkpoint path")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config.get("data_path", {})
    hparam = config.get("hparam", {})
    
    # 加载数据
    print("Loading dataset...")
    dataset = GraphRouterDataset(
        routing_data_path=data_cfg["routing_data_train"],
        embedding_path=data_cfg["query_embedding_data"],
        llm_config_path=data_cfg["llm_data"],
        val_ratio=hparam.get("val_split_ratio", 0.2)
    )
    print(f"  - {dataset.num_queries} queries")
    print(f"  - {dataset.num_llms} LLMs: {dataset.model_names}")
    
    # 初始化训练器
    print("Initializing trainer...")
    trainer = GraphRouterTrainer(
        dataset=dataset,
        hidden_dim=hparam.get("hidden_dim", 64),
        learning_rate=hparam.get("learning_rate", 0.001),
        weight_decay=hparam.get("weight_decay", 1e-4),
        epochs=hparam.get("train_epoch", 100),
        batch_size=hparam.get("batch_size", 4),
        mask_rate=hparam.get("train_mask_rate", 0.3)
    )
    
    # 训练
    print("Starting training...")
    best_result = trainer.train(save_path=args.checkpoint)
    
    # 导出为 Go 格式
    print("Exporting model...")
    export_for_go(
        model=trainer.model,
        dataset=dataset,
        output_path=args.output,
        train_accuracy=best_result,
        temperature=hparam.get("temperature", 1.0)
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
```

### 6.7 训练配置模板

```yaml
# tools/graphrouter/configs/default.yaml

# 数据路径
data_path:
  routing_data_train: 'data/routing_data.jsonl'
  query_embedding_data: 'data/query_embeddings.pt'
  llm_data: 'data/llm_candidates.json'

# 超参数
hparam:
  hidden_dim: 64         # 隐藏层维度
  learning_rate: 0.001   # 学习率
  weight_decay: 0.0001   # 权重衰减
  train_epoch: 100       # 训练轮数
  batch_size: 4          # 批大小
  train_mask_rate: 0.3   # 边掩码率
  val_split_ratio: 0.2   # 验证集比例
  temperature: 1.0       # Softmax 温度
```

### 6.8 依赖要求

```
# tools/graphrouter/requirements.txt
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
pyyaml>=6.0
```

### 6.9 使用示例

```bash
# 1. 安装依赖
cd tools/graphrouter
pip install -r requirements.txt

# 2. 准备数据
# - data/routing_data.jsonl
# - data/query_embeddings.pt
# - data/llm_candidates.json

# 3. 训练并导出
python train.py \
    --config configs/default.yaml \
    --output ../../config/models/graph_model.json \
    --checkpoint models/graphrouter.pt

# 4. 验证导出文件
python -c "import json; print(json.load(open('../../config/models/graph_model.json'))['metadata'])"
```

---

## 7. 配置说明

### 7.1 semantic-router 配置

```yaml
# config/config.yaml

model_selection:
  method: "graph"  # 使用 GraphSelector
  
  graph:
    # 预训练模型路径
    model_path: "config/models/graph_model.json"
    
    # Softmax 温度参数
    # - 1.0: 标准 softmax
    # - <1.0: 更确定性（选择更集中）
    # - >1.0: 更随机（选择更分散）
    temperature: 1.0
    
    # 最小置信度阈值
    # 低于此阈值时使用 fallback_model
    min_confidence: 0.3
    
    # 回退模型
    fallback_model: "gpt-4"
```

### 7.2 配置结构扩展

```go
// config/config.go 中添加

type GraphSelectionConfig struct {
    ModelPath     string  `yaml:"model_path" json:"model_path"`
    Temperature   float32 `yaml:"temperature" json:"temperature"`
    MinConfidence float32 `yaml:"min_confidence" json:"min_confidence"`
    FallbackModel string  `yaml:"fallback_model" json:"fallback_model"`
}

// ModelSelectionConfig 中添加
type ModelSelectionConfig struct {
    Method   string                `yaml:"method" json:"method"`
    KNN      KNNConfig             `yaml:"knn" json:"knn,omitempty"`
    Graph    GraphSelectionConfig  `yaml:"graph" json:"graph,omitempty"`  // 新增
    // ...
}
```

---

## 8. 评测方案

### 8.1 离线评测指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **Routing Accuracy** | `correct / total` | 选择了最优模型的比例 |
| **Avg Performance** | `mean(selected_perf)` | 选中模型的平均性能 |
| **Oracle Gap** | `oracle - actual` | 与最优选择的差距 |
| **Top-K Accuracy** | `best_in_top_k / total` | 最优模型在 Top-K 中的比例 |

### 8.2 评测脚本

```python
#!/usr/bin/env python3
"""GraphRouter 评测脚本"""

import json
import numpy as np
from typing import Dict, List


def evaluate_router(
    router,
    test_data: List[Dict],
    performance_matrix: np.ndarray
) -> Dict[str, float]:
    """
    评测路由器性能
    
    Args:
        router: GraphRouter 实例
        test_data: 测试查询列表
        performance_matrix: [num_queries, num_llms] 性能矩阵
    
    Returns:
        评测指标字典
    """
    results = {
        "routing_accuracy": 0.0,
        "avg_performance": 0.0,
        "oracle_gap": 0.0,
        "top_3_accuracy": 0.0,
    }
    
    correct = 0
    total_perf = 0.0
    oracle_perf = 0.0
    top_3_correct = 0
    
    model_names = router.model_names
    
    for i, query in enumerate(test_data):
        # 路由
        result = router.route_single(query)
        selected = result["model_name"]
        selected_idx = model_names.index(selected)
        
        # 获取性能
        perfs = performance_matrix[i]
        best_idx = np.argmax(perfs)
        best_perf = perfs[best_idx]
        selected_perf = perfs[selected_idx]
        
        # 统计
        if selected_idx == best_idx:
            correct += 1
        
        total_perf += selected_perf
        oracle_perf += best_perf
        
        # Top-3 准确率
        top_3_idx = np.argsort(perfs)[-3:]
        if best_idx in top_3_idx:
            top_3_correct += 1
    
    n = len(test_data)
    results["routing_accuracy"] = correct / n
    results["avg_performance"] = total_perf / n
    results["oracle_gap"] = (oracle_perf - total_perf) / n
    results["top_3_accuracy"] = top_3_correct / n
    
    return results


def compare_with_baselines(
    test_data: List[Dict],
    performance_matrix: np.ndarray,
    model_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """对比各种基线方法"""
    
    results = {}
    
    # 1. Random Selector
    random_perf = np.mean(performance_matrix)
    results["random"] = {"avg_performance": float(random_perf)}
    
    # 2. Always Best (Oracle)
    oracle_perf = np.mean(np.max(performance_matrix, axis=1))
    results["oracle"] = {"avg_performance": float(oracle_perf)}
    
    # 3. Always Worst
    worst_perf = np.mean(np.min(performance_matrix, axis=1))
    results["worst"] = {"avg_performance": float(worst_perf)}
    
    # 4. Most Frequent Winner
    best_models = np.argmax(performance_matrix, axis=1)
    most_frequent = np.bincount(best_models).argmax()
    frequent_perf = np.mean(performance_matrix[:, most_frequent])
    results["most_frequent"] = {
        "model": model_names[most_frequent],
        "avg_performance": float(frequent_perf)
    }
    
    return results
```

### 8.3 在线 A/B 测试配置

```yaml
# A/B 测试配置
ab_test:
  name: "graph_router_rollout"
  
  variants:
    - name: "control"
      selector: "knn"
      traffic_percentage: 50
    
    - name: "treatment"
      selector: "graph"
      traffic_percentage: 50
  
  metrics:
    primary:
      - name: "response_quality"
        type: "llm_judge"
        threshold: 0.02  # 显著性阈值
    
    secondary:
      - name: "latency_p50"
      - name: "cost_per_query"
      - name: "routing_latency_ms"
  
  duration_days: 14
  min_samples_per_variant: 5000
```

### 8.4 评测结果示例

```json
{
  "graph_router": {
    "routing_accuracy": 0.82,
    "avg_performance": 0.87,
    "oracle_gap": 0.05,
    "top_3_accuracy": 0.95
  },
  "baselines": {
    "random": {"avg_performance": 0.72},
    "oracle": {"avg_performance": 0.92},
    "most_frequent": {"model": "gpt-4", "avg_performance": 0.78}
  },
  "comparison": {
    "vs_random": "+20.8%",
    "vs_most_frequent": "+11.5%",
    "vs_oracle": "-5.4%"
  }
}
```

---

## 9. 部署指南

### 9.1 完整部署流程

```bash
# 1. 准备训练数据
cd tools/graphrouter
python data_prepare.py \
    --routing-data ../../data/routing_data.jsonl \
    --embeddings ../../data/query_embeddings.pt \
    --llm-config ../../data/llm_candidates.json \
    --output ../../data/prepared/

# 2. 训练模型
python train.py \
    --config configs/graphrouter_train.yaml \
    --output ../../config/models/graph_model.json

# 3. 评测
python evaluate.py \
    --model ../../config/models/graph_model.json \
    --test-data ../../data/test_data.jsonl

# 4. 更新配置
cat >> ../../config/config.yaml << EOF
model_selection:
  method: "graph"
  graph:
    model_path: "config/models/graph_model.json"
    temperature: 1.0
    min_confidence: 0.3
    fallback_model: "gpt-4"
EOF

# 5. 重启服务
cd ../..
make restart
```

### 9.2 模型热更新

```go
// 支持运行时重新加载模型
func (s *GraphSelector) Reload(modelPath string) error {
    newModel := &GraphModelData{}
    data, err := os.ReadFile(modelPath)
    if err != nil {
        return err
    }
    if err := json.Unmarshal(data, newModel); err != nil {
        return err
    }
    
    // 原子替换
    s.mu.Lock()
    s.modelNames = newModel.ModelNames
    s.queryProjection = newModel.QueryProjection
    s.llmRepresentations = newModel.LLMRepresentations
    s.mu.Unlock()
    
    logging.Infof("[GraphSelector] Model reloaded from %s", modelPath)
    return nil
}
```

### 9.3 监控指标

```yaml
# Prometheus 指标
metrics:
  - name: graph_selector_latency_ms
    type: histogram
    labels: [model_selected]
    
  - name: graph_selector_confidence
    type: histogram
    labels: [model_selected]
    
  - name: graph_selector_fallback_total
    type: counter
    labels: [reason]
```

---

## 10. 路线图

### Phase 1: 基础实现 (Week 1-2)

- [ ] Go 端 `GraphSelector` 核心实现
- [ ] 模型加载和推理逻辑
- [ ] 工厂注册和配置支持
- [ ] 单元测试

### Phase 2: 训练工具 (Week 2-3)

- [ ] Python 训练脚本
- [ ] 权重导出工具
- [ ] 数据准备工具
- [ ] 训练文档

### Phase 3: 评测集成 (Week 3-4)

- [ ] 离线评测脚本
- [ ] Benchmark 测试
- [ ] 与 KNN/Static 对比
- [ ] 性能调优

### Phase 4: 生产部署 (Week 4-5)

- [ ] 配置示例
- [ ] 监控指标
- [ ] 热更新支持
- [ ] A/B 测试框架集成

### Phase 5: 优化迭代 (持续)

- [ ] 在线增量更新探索
- [ ] 多任务学习支持
- [ ] 成本感知路由
- [ ] 自动超参调优

---

## 附录

### A. 设计决策

| 决策点 | 选择 | 原因 |
|--------|------|------|
| **训练工具** | 独立 Python 实现 | 与 LLMRouter 解耦，便于独立维护 |
| **Go GNN 实现** | 预计算 + 相似度匹配 | Go 无成熟 GNN 库，轻量高效 |
| **模型交换格式** | JSON | Go 原生支持，易于调试和版本管理 |
| **Embedding** | 复用 candle-binding | 已有 Qwen3/BERT 支持 |
| **在线更新** | 不支持 | GNN 需要完整图，增量更新复杂 |

### B. 算法对比

| 算法 | 训练复杂度 | 推理延迟 | 需要反馈 | 适用场景 |
|------|-----------|---------|----------|---------|
| **GraphRouter** | 中等 | < 0.5ms | ❌ | 关系建模强，精度要求高 |
| KNNRouter | 低 | 1-5ms | ✅ | 在线学习，冷启动 |
| StaticRouter | 无 | < 0.1ms | ❌ | 简单场景，固定策略 |
| EloRouter | 低 | < 0.1ms | ✅ | 对战评分，用户偏好 |

### C. 性能预估

| 操作 | 时间复杂度 | 预估延迟 |
|------|-----------|---------|
| Query 投影 | O(query_dim × hidden_dim) | < 0.1ms |
| LLM 相似度 | O(num_llms × hidden_dim) | < 0.1ms |
| Softmax | O(num_llms) | < 0.01ms |
| **总计** | - | **< 0.5ms** |

### D. 模块对应关系

| 功能 | Python 模块 | Go 模块 |
|------|------------|---------|
| GNN 模型定义 | `tools/graphrouter/model.py` | - |
| 数据加载 | `tools/graphrouter/data.py` | - |
| 训练逻辑 | `tools/graphrouter/trainer.py` | - |
| 模型导出 | `tools/graphrouter/export.py` | - |
| 推理选择 | - | `pkg/selection/graph.go` |
| 配置加载 | - | `pkg/selection/factory.go` |

### E. 参考资料

- [PyTorch Geometric GeneralConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GeneralConv)
- [Graph Neural Networks 综述](https://arxiv.org/abs/1901.00596)
- [Heterogeneous Graph Attention Network](https://arxiv.org/abs/1903.07293)
