"""
GraphRouter 模型导出工具

将训练好的 PyTorch 模型导出为 Go 可读的 JSON 格式。

方案 C（完整 GNN 推理）导出内容:
- model_weights: GNN 模型权重
- training_graph: 训练图数据（用于推理时的 GNN 消息传递）
- model_names: LLM 名称列表
- 元数据: 训练信息

Go 推理流程:
1. 将新 Query 加入训练图
2. 运行 GNN 消息传递（训练边作为上下文）
3. 预测新 Query 对所有 LLM 的分数
"""

import json
import torch
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path

try:
    from .model import EncoderDecoderNet
    from .data import GraphRouterDataset
except ImportError:
    from model import EncoderDecoderNet
    from data import GraphRouterDataset


def export_for_go(
    model: EncoderDecoderNet,
    dataset: GraphRouterDataset,
    output_path: str,
    train_accuracy: float,
    temperature: float = 1.0,
    metadata: Optional[Dict] = None
):
    """
    导出模型为 Go 可读的 JSON 格式（方案 C：完整 GNN 推理）
    
    Go 推理逻辑:
    1. 将新 Query embedding 加入 training_query_embeddings
    2. 构建新 Query 到所有 LLM 的边（权重为 0）
    3. 运行 GNN 消息传递（训练边作为上下文）
    4. 预测新 Query 对所有 LLM 的分数
    5. 选择最高分的 LLM
    
    Args:
        model: 训练好的 EncoderDecoderNet 模型
        dataset: 数据集（用于获取元信息和训练图数据）
        output_path: 输出 JSON 路径
        train_accuracy: 训练准确率
        temperature: Softmax 温度参数
        metadata: 额外的元数据
    """
    model.eval()
    
    # 1. 导出模型权重
    model_weights = export_model_weights(model)
    
    # 2. 导出训练图数据（用于推理时的 GNN 消息传递）
    training_graph = export_training_graph(dataset)
    
    # 3. 构建导出数据
    export_data = {
        "version": "2.0",  # 版本 2.0 表示完整 GNN 推理
        "inference_mode": "full_gnn",  # 推理模式标识
        "model_names": dataset.model_names,
        "hidden_dim": model.hidden_dim,
        "query_dim": dataset.query_dim,
        "llm_dim": dataset.llm_dim,
        "num_llms": dataset.num_llms,
        "num_train_queries": dataset.num_queries,
        
        # 模型权重
        "model_weights": model_weights,
        
        # 训练图数据
        "training_graph": training_graph,
        
        "temperature": temperature,
        "metadata": {
            "train_samples": dataset.num_queries,
            "train_accuracy": float(train_accuracy),
            "num_llms": dataset.num_llms,
            "export_date": datetime.now().isoformat(),
            "inference_mode": "full_gnn",
            **(metadata or {})
        }
    }
    
    # 4. 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False)
    
    print(f"\nModel exported to {output_path}")
    print(f"  - Inference mode: full_gnn (方案 C)")
    print(f"  - {dataset.num_llms} LLMs: {dataset.model_names}")
    print(f"  - {dataset.num_queries} training queries")
    print(f"  - hidden_dim: {model.hidden_dim}")
    print(f"  - query_dim: {dataset.query_dim}")
    print(f"  - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def export_model_weights(model: EncoderDecoderNet) -> Dict[str, Any]:
    """
    导出 GNN 模型权重
    
    包含:
    - FeatureAlign: query_transform, llm_transform
    - GNN Conv 层: conv1, conv2
    - BatchNorm: bn1, bn2
    - Edge MLP: edge_mlp
    """
    model.eval()
    weights = {}
    
    # FeatureAlign 权重
    weights["query_transform_weight"] = model.align.query_transform.weight.detach().cpu().numpy().tolist()
    weights["query_transform_bias"] = model.align.query_transform.bias.detach().cpu().numpy().tolist()
    weights["llm_transform_weight"] = model.align.llm_transform.weight.detach().cpu().numpy().tolist()
    weights["llm_transform_bias"] = model.align.llm_transform.bias.detach().cpu().numpy().tolist()
    
    # Edge MLP 权重
    weights["edge_mlp_weight"] = model.edge_mlp.weight.detach().cpu().numpy().tolist()
    weights["edge_mlp_bias"] = model.edge_mlp.bias.detach().cpu().numpy().tolist()
    
    # GNN Conv 层权重（GeneralConv 内部结构）
    # 注意：GeneralConv 的具体参数需要根据其内部实现提取
    weights["conv1_state"] = {k: v.detach().cpu().numpy().tolist() for k, v in model.conv1.state_dict().items()}
    weights["conv2_state"] = {k: v.detach().cpu().numpy().tolist() for k, v in model.conv2.state_dict().items()}
    
    # BatchNorm 权重
    weights["bn1_weight"] = model.bn1.weight.detach().cpu().numpy().tolist()
    weights["bn1_bias"] = model.bn1.bias.detach().cpu().numpy().tolist()
    weights["bn1_running_mean"] = model.bn1.running_mean.detach().cpu().numpy().tolist()
    weights["bn1_running_var"] = model.bn1.running_var.detach().cpu().numpy().tolist()
    
    weights["bn2_weight"] = model.bn2.weight.detach().cpu().numpy().tolist()
    weights["bn2_bias"] = model.bn2.bias.detach().cpu().numpy().tolist()
    weights["bn2_running_mean"] = model.bn2.running_mean.detach().cpu().numpy().tolist()
    weights["bn2_running_var"] = model.bn2.running_var.detach().cpu().numpy().tolist()
    
    return weights


def export_training_graph(dataset: GraphRouterDataset) -> Dict[str, Any]:
    """
    导出训练图数据（用于推理时的 GNN 消息传递上下文）
    
    包含:
    - query_embeddings: 训练 Query 的 embedding [num_queries, query_dim]
    - llm_embeddings: LLM 的 embedding [num_llms, llm_dim]
    - edge_weights: 训练边的权重（performance）[num_queries * num_llms]
    """
    graph_data = {
        # Query embeddings（已归一化）
        "query_embeddings": dataset.query_embedding_matrix.tolist(),
        
        # LLM embeddings（已归一化）
        "llm_embeddings": dataset.llm_embedding_matrix.tolist(),
        
        # 边权重（performance scores）
        # 形状: [num_queries * num_llms]，按 query 顺序排列
        "edge_weights": dataset.performance_matrix.flatten().tolist(),
    }
    
    return graph_data


def export_from_checkpoint(
    checkpoint_path: str,
    dataset: GraphRouterDataset,
    output_path: str,
    hidden_dim: int = 64,
    temperature: float = 1.0,
    train_accuracy: float = 0.0
):
    """
    从 checkpoint 文件导出模型
    
    Args:
        checkpoint_path: PyTorch checkpoint 路径
        dataset: 数据集
        output_path: 输出 JSON 路径
        hidden_dim: 隐藏层维度
        temperature: Softmax 温度
        train_accuracy: 训练准确率
    """
    # 创建模型
    model = EncoderDecoderNet(
        query_dim=dataset.query_dim,
        llm_dim=dataset.llm_dim,
        hidden_dim=hidden_dim
    )
    
    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    
    # 导出
    export_for_go(
        model=model,
        dataset=dataset,
        output_path=output_path,
        train_accuracy=train_accuracy,
        temperature=temperature,
        metadata={"checkpoint_path": checkpoint_path}
    )


def validate_export(export_path: str):
    """
    验证导出的 JSON 文件（方案 C）
    
    Args:
        export_path: JSON 文件路径
    """
    with open(export_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nValidating export: {export_path}")
    print(f"  Version: {data['version']}")
    print(f"  Inference mode: {data.get('inference_mode', 'simple')}")
    print(f"  Model names: {data['model_names']}")
    print(f"  Hidden dim: {data['hidden_dim']}")
    print(f"  Query dim: {data['query_dim']}")
    print(f"  Num LLMs: {data['num_llms']}")
    print(f"  Num train queries: {data['num_train_queries']}")
    print(f"  Temperature: {data['temperature']}")
    
    # 验证训练图数据
    if 'training_graph' in data:
        tg = data['training_graph']
        query_emb = np.array(tg['query_embeddings'])
        llm_emb = np.array(tg['llm_embeddings'])
        edge_weights = np.array(tg['edge_weights'])
        
        print(f"  Training graph:")
        print(f"    - Query embeddings shape: {query_emb.shape}")
        print(f"    - LLM embeddings shape: {llm_emb.shape}")
        print(f"    - Edge weights shape: {edge_weights.shape}")
        
        # 验证边权重数量
        expected_edges = query_emb.shape[0] * llm_emb.shape[0]
        assert len(edge_weights) == expected_edges, \
            f"Edge weights count mismatch: {len(edge_weights)} vs expected {expected_edges}"
    
    # 验证模型权重
    if 'model_weights' in data:
        mw = data['model_weights']
        print(f"  Model weights:")
        print(f"    - Query transform: {np.array(mw['query_transform_weight']).shape}")
        print(f"    - LLM transform: {np.array(mw['llm_transform_weight']).shape}")
        print(f"    - BatchNorm 1: weight shape {np.array(mw['bn1_weight']).shape}")
        print(f"    - BatchNorm 2: weight shape {np.array(mw['bn2_weight']).shape}")
    
    print(f"  Metadata: {data['metadata']}")
    print("  Validation passed!")
