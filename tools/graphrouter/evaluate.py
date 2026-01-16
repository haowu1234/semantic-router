#!/usr/bin/env python3
"""
GraphRouter 评测脚本

评测训练好的模型在测试集上的性能，并与基线方法对比。

用法:
    python evaluate.py --model model.json --test-data test.jsonl --embeddings embeddings.pt
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any
import sys

sys.path.insert(0, str(Path(__file__).parent))


class GoStylePredictor:
    """
    模拟 Go 推理逻辑的预测器
    
    支持两种模式:
    - 版本 1.0: 简化推理（query_projection + llm_representations）
    - 版本 2.0: 完整 GNN 推理（使用训练图作为上下文）
    """
    
    def __init__(self, model_path: str):
        with open(model_path, 'r') as f:
            self.data = json.load(f)
        
        self.model_names = self.data['model_names']
        self.temperature = self.data['temperature']
        self.hidden_dim = self.data['hidden_dim']
        self.version = self.data.get('version', '1.0')
        self.inference_mode = self.data.get('inference_mode', 'simple')
        
        if self.inference_mode == 'full_gnn' or self.version == '2.0':
            self._init_full_gnn()
        else:
            self._init_simple()
    
    def _init_simple(self):
        """初始化简化推理模式（版本 1.0）"""
        self.query_projection = np.array(self.data['query_projection'])
        self.llm_representations = np.array(self.data['llm_representations'])
    
    def _init_full_gnn(self):
        """初始化完整 GNN 推理模式（版本 2.0）"""
        # 加载模型权重
        weights = self.data['model_weights']
        self.query_transform_weight = np.array(weights['query_transform_weight'])
        self.query_transform_bias = np.array(weights['query_transform_bias'])
        self.llm_transform_weight = np.array(weights['llm_transform_weight'])
        self.llm_transform_bias = np.array(weights['llm_transform_bias'])
        
        # 加载训练图数据
        training_graph = self.data['training_graph']
        self.train_query_embeddings = np.array(training_graph['query_embeddings'])
        self.llm_embeddings = np.array(training_graph['llm_embeddings'])
        self.edge_weights = np.array(training_graph['edge_weights'])
        
        self.num_train_queries = self.train_query_embeddings.shape[0]
        self.num_llms = self.llm_embeddings.shape[0]
        
        # 预计算 LLM 的对齐表示
        self._precompute_llm_representations()
    
    def _precompute_llm_representations(self):
        """预计算 LLM 在隐空间中的表示"""
        # LLM embedding -> 隐空间
        # [num_llms, llm_dim] @ [llm_dim, hidden_dim] + [hidden_dim]
        self.llm_aligned = np.dot(self.llm_embeddings, self.llm_transform_weight.T) + self.llm_transform_bias
        
        # L2 归一化
        norms = np.linalg.norm(self.llm_aligned, axis=1, keepdims=True)
        self.llm_aligned = self.llm_aligned / (norms + 1e-8)
    
    def predict(self, query_embedding: np.ndarray) -> Dict[str, Any]:
        """预测最佳 LLM"""
        if self.inference_mode == 'full_gnn' or self.version == '2.0':
            return self._predict_full_gnn(query_embedding)
        else:
            return self._predict_simple(query_embedding)
    
    def _predict_simple(self, query_embedding: np.ndarray) -> Dict[str, Any]:
        """
        简化推理（版本 1.0）
        
        1. query_hidden = query_embedding @ query_projection
        2. scores[i] = cosine_similarity(query_hidden, llm_representations[i])
        3. probs = softmax(scores / temperature)
        """
        # 投影到隐空间
        query_hidden = query_embedding @ self.query_projection
        
        # L2 归一化
        query_norm = np.linalg.norm(query_hidden)
        if query_norm > 1e-8:
            query_hidden = query_hidden / query_norm
        
        # 计算余弦相似度
        scores = np.dot(self.llm_representations, query_hidden)
        
        return self._scores_to_result(scores)
    
    def _predict_full_gnn(self, query_embedding: np.ndarray) -> Dict[str, Any]:
        """
        完整 GNN 推理（版本 2.0）
        
        简化实现：使用预计算的 LLM 表示 + Query 投影
        完整 GNN 消息传递在 Go 中实现，这里用相似度匹配近似
        
        1. query_hidden = query_embedding @ query_transform
        2. scores[i] = cosine_similarity(query_hidden, llm_aligned[i])
        3. probs = softmax(scores / temperature)
        """
        # Query embedding -> 隐空间
        query_hidden = np.dot(query_embedding, self.query_transform_weight.T) + self.query_transform_bias
        
        # L2 归一化
        query_norm = np.linalg.norm(query_hidden)
        if query_norm > 1e-8:
            query_hidden = query_hidden / query_norm
        
        # 计算与所有 LLM 的相似度
        scores = np.dot(self.llm_aligned, query_hidden)
        
        return self._scores_to_result(scores)
    
    def _scores_to_result(self, scores: np.ndarray) -> Dict[str, Any]:
        """将分数转换为结果"""
        # Softmax
        scores_scaled = scores / self.temperature
        scores_scaled = scores_scaled - scores_scaled.max()  # 数值稳定性
        exp_scores = np.exp(scores_scaled)
        probs = exp_scores / exp_scores.sum()
        
        # 选择最高分
        best_idx = np.argmax(probs)
        
        return {
            'model_name': self.model_names[best_idx],
            'model_idx': int(best_idx),
            'confidence': float(probs[best_idx]),
            'scores': {name: float(probs[i]) for i, name in enumerate(self.model_names)}
        }
    
    def predict_batch(self, query_embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """批量预测"""
        return [self.predict(emb) for emb in query_embeddings]


def evaluate_router(
    predictor: GoStylePredictor,
    test_embeddings: np.ndarray,
    performance_matrix: np.ndarray,
    model_names: List[str]
) -> Dict[str, float]:
    """
    评测路由器性能
    
    Args:
        predictor: GoStylePredictor 实例
        test_embeddings: [num_queries, embedding_dim] 测试查询 embedding
        performance_matrix: [num_queries, num_llms] 性能矩阵
        model_names: LLM 名称列表（与 performance_matrix 对应）
    
    Returns:
        评测指标字典
    """
    # 建立模型名称到索引的映射
    pred_model_to_idx = {name: i for i, name in enumerate(predictor.model_names)}
    test_model_to_idx = {name: i for i, name in enumerate(model_names)}
    
    correct = 0
    total_perf = 0.0
    oracle_perf = 0.0
    top_3_correct = 0
    
    num_queries = len(test_embeddings)
    
    for i in range(num_queries):
        # 预测
        result = predictor.predict(test_embeddings[i])
        selected_name = result['model_name']
        
        # 获取性能
        perfs = performance_matrix[i]
        best_idx = np.argmax(perfs)
        best_perf = perfs[best_idx]
        
        # 如果预测的模型在测试集中存在
        if selected_name in test_model_to_idx:
            selected_idx = test_model_to_idx[selected_name]
            selected_perf = perfs[selected_idx]
        else:
            # 模型不在测试集中，使用 0 性能
            selected_perf = 0.0
            selected_idx = -1
        
        # 统计
        if selected_idx == best_idx:
            correct += 1
        
        total_perf += selected_perf
        oracle_perf += best_perf
        
        # Top-3 准确率
        top_3_idx = np.argsort(perfs)[-3:]
        if best_idx in top_3_idx:
            top_3_correct += 1
    
    return {
        'routing_accuracy': correct / num_queries,
        'avg_performance': total_perf / num_queries,
        'oracle_performance': oracle_perf / num_queries,
        'oracle_gap': (oracle_perf - total_perf) / num_queries,
        'top_3_accuracy': top_3_correct / num_queries,
        'num_samples': num_queries
    }


def compare_with_baselines(
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
    
    # 5. Per-model average performance
    model_avg_perfs = {}
    for i, name in enumerate(model_names):
        model_avg_perfs[name] = float(np.mean(performance_matrix[:, i]))
    results["per_model_avg"] = model_avg_perfs
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphRouter model")
    parser.add_argument(
        "--model", 
        required=True, 
        help="Exported model JSON path"
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Test routing data JSONL path"
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Test query embeddings path (.pt or .npy)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output results JSON path"
    )
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    predictor = GoStylePredictor(args.model)
    print(f"  Version: {predictor.version}")
    print(f"  Inference mode: {predictor.inference_mode}")
    print(f"  Model names: {predictor.model_names}")
    print(f"  Hidden dim: {predictor.hidden_dim}")
    
    print(f"\nLoading test data from {args.test_data}...")
    test_records = []
    with open(args.test_data, 'r') as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))
    
    # 获取唯一 query 和对应的 embedding_id
    unique_queries = []
    embedding_ids = []
    seen_queries = set()
    for record in test_records:
        if record['query'] not in seen_queries:
            seen_queries.add(record['query'])
            unique_queries.append(record['query'])
            embedding_ids.append(record['embedding_id'])
    
    print(f"  Found {len(unique_queries)} unique test queries")
    
    # 加载 embeddings
    print(f"\nLoading embeddings from {args.embeddings}...")
    if args.embeddings.endswith('.pt'):
        all_embeddings = torch.load(args.embeddings, map_location='cpu').numpy()
    else:
        all_embeddings = np.load(args.embeddings)
    
    test_embeddings = np.array([all_embeddings[i] for i in embedding_ids])
    print(f"  Test embeddings shape: {test_embeddings.shape}")
    
    # 构建 performance matrix
    model_names = list(set(r['model_name'] for r in test_records))
    model_to_idx = {name: i for i, name in enumerate(model_names)}
    query_to_idx = {q: i for i, q in enumerate(unique_queries)}
    
    performance_matrix = np.zeros((len(unique_queries), len(model_names)))
    for record in test_records:
        q_idx = query_to_idx[record['query']]
        m_idx = model_to_idx[record['model_name']]
        performance_matrix[q_idx, m_idx] = record['performance']
    
    print(f"  Performance matrix shape: {performance_matrix.shape}")
    
    # 归一化 embeddings
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    test_embeddings = scaler.fit_transform(test_embeddings)
    
    # 评测
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    metrics = evaluate_router(
        predictor=predictor,
        test_embeddings=test_embeddings,
        performance_matrix=performance_matrix,
        model_names=model_names
    )
    
    print(f"\nGraphRouter:")
    print(f"  Routing accuracy: {metrics['routing_accuracy']:.4f}")
    print(f"  Avg performance:  {metrics['avg_performance']:.4f}")
    print(f"  Oracle gap:       {metrics['oracle_gap']:.4f}")
    print(f"  Top-3 accuracy:   {metrics['top_3_accuracy']:.4f}")
    
    # 基线对比
    baselines = compare_with_baselines(performance_matrix, model_names)
    
    print(f"\nBaselines:")
    print(f"  Random:        {baselines['random']['avg_performance']:.4f}")
    print(f"  Oracle:        {baselines['oracle']['avg_performance']:.4f}")
    print(f"  Worst:         {baselines['worst']['avg_performance']:.4f}")
    print(f"  Most frequent: {baselines['most_frequent']['avg_performance']:.4f} ({baselines['most_frequent']['model']})")
    
    # 对比
    print(f"\nComparison:")
    improvement_vs_random = (metrics['avg_performance'] - baselines['random']['avg_performance']) / baselines['random']['avg_performance'] * 100
    improvement_vs_frequent = (metrics['avg_performance'] - baselines['most_frequent']['avg_performance']) / baselines['most_frequent']['avg_performance'] * 100
    gap_to_oracle = (baselines['oracle']['avg_performance'] - metrics['avg_performance']) / baselines['oracle']['avg_performance'] * 100
    
    print(f"  vs Random:        {improvement_vs_random:+.1f}%")
    print(f"  vs Most frequent: {improvement_vs_frequent:+.1f}%")
    print(f"  Gap to Oracle:    -{gap_to_oracle:.1f}%")
    
    # 保存结果
    if args.output:
        results = {
            "graph_router": metrics,
            "baselines": baselines,
            "comparison": {
                "vs_random": f"{improvement_vs_random:+.1f}%",
                "vs_most_frequent": f"{improvement_vs_frequent:+.1f}%",
                "gap_to_oracle": f"-{gap_to_oracle:.1f}%"
            }
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
