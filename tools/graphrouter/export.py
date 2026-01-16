"""
GraphRouter 模型导出工具

将训练好的 PyTorch 模型导出为 Go 可读的 JSON 格式。

导出内容:
- model_names: LLM 名称列表
- query_projection: Query 投影矩阵 [query_dim, hidden_dim]
- llm_representations: LLM 隐空间表示 [num_llms, hidden_dim]
- 元数据: 训练信息
"""

import json
import torch
import numpy as np
from typing import Dict, Optional
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
    导出模型为 Go 可读的 JSON 格式
    
    Go 推理逻辑:
    1. query_hidden = query_embedding @ query_projection
    2. scores[i] = cosine_similarity(query_hidden, llm_representations[i])
    3. probs = softmax(scores / temperature)
    4. selected = argmax(probs)
    
    Args:
        model: 训练好的 EncoderDecoderNet 模型
        dataset: 数据集（用于获取元信息和 LLM embedding）
        output_path: 输出 JSON 路径
        train_accuracy: 训练准确率
        temperature: Softmax 温度参数
        metadata: 额外的元数据
    """
    model.eval()
    
    # 1. 提取 Query 投影矩阵
    # PyTorch Linear 的 weight shape: [out_features, in_features]
    # 需要转置为 [in_features, out_features] = [query_dim, hidden_dim]
    query_projection = model.align.query_transform.weight.detach().cpu().numpy().T
    
    # 2. 计算 LLM 表示
    llm_representations = compute_llm_representations(model, dataset)
    
    # 3. 构建导出数据
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
            "export_date": datetime.now().isoformat(),
            **(metadata or {})
        }
    }
    
    # 4. 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nModel exported to {output_path}")
    print(f"  - {dataset.num_llms} LLMs: {dataset.model_names}")
    print(f"  - hidden_dim: {model.hidden_dim}")
    print(f"  - query_dim: {dataset.query_dim}")
    print(f"  - query_projection shape: {query_projection.shape}")
    print(f"  - llm_representations shape: {llm_representations.shape}")


def compute_llm_representations(
    model: EncoderDecoderNet,
    dataset: GraphRouterDataset
) -> np.ndarray:
    """
    计算 LLM 聚合表示
    
    将 LLM embedding 通过特征对齐层，得到隐空间中的表示，
    并进行 L2 归一化以便于后续的余弦相似度计算。
    
    Args:
        model: 训练好的模型
        dataset: 数据集
    
    Returns:
        llm_representations: [num_llms, hidden_dim] L2 归一化后的 LLM 表示
    """
    model.eval()
    
    with torch.no_grad():
        # 获取 embedding
        query_emb = torch.from_numpy(dataset.query_embedding_matrix).float()
        llm_emb = torch.from_numpy(dataset.llm_embedding_matrix).float()
        
        # 特征对齐
        # aligned shape: [num_queries + num_llms, hidden_dim]
        aligned = model.align(query_emb, llm_emb)
        
        # 提取 LLM 部分
        llm_aligned = aligned[dataset.num_queries:]  # [num_llms, hidden_dim]
        llm_representations = llm_aligned.cpu().numpy()
    
    # L2 归一化（用于余弦相似度计算）
    norms = np.linalg.norm(llm_representations, axis=1, keepdims=True)
    llm_representations = llm_representations / (norms + 1e-8)
    
    return llm_representations


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
    state_dict = torch.load(checkpoint_path, map_location='cpu')
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
    验证导出的 JSON 文件
    
    Args:
        export_path: JSON 文件路径
    """
    with open(export_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nValidating export: {export_path}")
    print(f"  Version: {data['version']}")
    print(f"  Model names: {data['model_names']}")
    print(f"  Hidden dim: {data['hidden_dim']}")
    print(f"  Query dim: {data['query_dim']}")
    print(f"  Temperature: {data['temperature']}")
    
    # 验证矩阵维度
    query_proj = np.array(data['query_projection'])
    llm_reps = np.array(data['llm_representations'])
    
    assert query_proj.shape == (data['query_dim'], data['hidden_dim']), \
        f"query_projection shape mismatch: {query_proj.shape}"
    assert llm_reps.shape == (len(data['model_names']), data['hidden_dim']), \
        f"llm_representations shape mismatch: {llm_reps.shape}"
    
    # 验证 LLM 表示是否已归一化
    norms = np.linalg.norm(llm_reps, axis=1)
    print(f"  LLM representation norms: min={norms.min():.4f}, max={norms.max():.4f}")
    
    print(f"  Metadata: {data['metadata']}")
    print("  Validation passed!")
