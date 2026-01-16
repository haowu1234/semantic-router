"""
GraphRouter 数据加载和预处理

支持以下数据格式:
- routing_data.jsonl: 路由训练数据 (query, model_name, performance, embedding_id)
- query_embeddings.pt/.npy: Query Embedding 张量
- llm_candidates.json: LLM 配置 (model, embedding, cost_per_1k_tokens)
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import MinMaxScaler

try:
    from .model import FormData
except ImportError:
    from model import FormData


class GraphRouterDataset:
    """
    GraphRouter 数据集
    
    负责加载和预处理训练数据，构建图数据结构。
    
    Args:
        routing_data_path: 路由数据路径 (JSONL 格式)
        embedding_path: Query Embedding 路径 (.pt 或 .npy)
        llm_config_path: LLM 配置路径 (JSON)
        val_ratio: 验证集比例
        device: 计算设备
    """
    
    def __init__(
        self,
        routing_data_path: str,
        embedding_path: str,
        llm_config_path: str,
        val_ratio: float = 0.2,
        device: str = None
    ):
        self.val_ratio = val_ratio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载数据
        print(f"Loading routing data from {routing_data_path}...")
        self.routing_df = self._load_routing_data(routing_data_path)
        
        print(f"Loading embeddings from {embedding_path}...")
        self.query_embeddings = self._load_embeddings(embedding_path)
        
        print(f"Loading LLM config from {llm_config_path}...")
        self.llm_config = self._load_llm_config(llm_config_path)
        
        # 处理数据
        print("Processing data...")
        self._process_data()
        
        # 初始化 FormData
        self.form_data = FormData(self.device)
    
    def _load_routing_data(self, path: str) -> pd.DataFrame:
        """加载路由训练数据 (JSONL 格式)"""
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)
    
    def _load_embeddings(self, path: str) -> torch.Tensor:
        """加载 Query Embedding"""
        path = str(path)
        if path.endswith('.pt'):
            return torch.load(path, map_location='cpu')
        elif path.endswith('.npy'):
            return torch.from_numpy(np.load(path))
        elif path.endswith('.npz'):
            data = np.load(path)
            # 假设第一个数组是 embeddings
            key = list(data.keys())[0]
            return torch.from_numpy(data[key])
        else:
            raise ValueError(f"Unsupported embedding format: {path}")
    
    def _load_llm_config(self, path: str) -> Dict:
        """加载 LLM 配置"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _process_data(self):
        """处理数据，构建训练所需的张量"""
        # 获取模型列表
        self.model_names = self.routing_df["model_name"].unique().tolist()
        self.num_llms = len(self.model_names)
        self.model_to_idx = {name: i for i, name in enumerate(self.model_names)}
        print(f"  Found {self.num_llms} LLMs: {self.model_names}")
        
        # 获取唯一 query
        unique_queries = self.routing_df["query"].unique().tolist()
        self.num_queries = len(unique_queries)
        self.unique_queries = unique_queries
        print(f"  Found {self.num_queries} unique queries")
        
        # 提取 query embedding
        embedding_ids = []
        for query in unique_queries:
            query_data = self.routing_df[self.routing_df["query"] == query]
            embedding_id = query_data["embedding_id"].iloc[0]
            embedding_ids.append(embedding_id)
        
        query_emb_list = [self.query_embeddings[i].numpy() for i in embedding_ids]
        self.query_embedding_matrix = np.array(query_emb_list)
        
        # 构建 performance 矩阵
        self.performance_matrix = np.zeros((self.num_queries, self.num_llms))
        for i, query in enumerate(unique_queries):
            query_data = self.routing_df[self.routing_df["query"] == query]
            for _, row in query_data.iterrows():
                model_idx = self.model_to_idx[row["model_name"]]
                self.performance_matrix[i, model_idx] = row["performance"]
        
        # 处理 NaN
        self.performance_matrix = np.nan_to_num(self.performance_matrix, nan=0.0)
        
        # 归一化 query embedding
        scaler = MinMaxScaler()
        self.query_embedding_matrix = scaler.fit_transform(self.query_embedding_matrix)
        self.query_dim = self.query_embedding_matrix.shape[1]
        print(f"  Query embedding dimension: {self.query_dim}")
        
        # 处理 LLM embedding
        self._prepare_llm_embeddings()
    
    def _prepare_llm_embeddings(self):
        """准备 LLM Embedding"""
        llm_embeddings = []
        
        for model_name in self.model_names:
            if model_name in self.llm_config and "embedding" in self.llm_config[model_name]:
                emb = self.llm_config[model_name]["embedding"]
            else:
                # 随机初始化（使用与 query 相同的维度）
                print(f"  Warning: No embedding for {model_name}, using random initialization")
                emb = np.random.randn(self.query_dim).tolist()
            llm_embeddings.append(emb)
        
        self.llm_embedding_matrix = np.array(llm_embeddings)
        
        # 确保维度匹配
        if self.llm_embedding_matrix.shape[1] != self.query_dim:
            print(f"  Warning: LLM embedding dim ({self.llm_embedding_matrix.shape[1]}) != query dim ({self.query_dim})")
            # 使用随机初始化
            self.llm_embedding_matrix = np.random.randn(self.num_llms, self.query_dim)
        
        # 归一化
        scaler = MinMaxScaler()
        self.llm_embedding_matrix = scaler.fit_transform(self.llm_embedding_matrix)
        self.llm_dim = self.llm_embedding_matrix.shape[1]
        print(f"  LLM embedding dimension: {self.llm_dim}")
    
    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取训练张量"""
        return (
            torch.from_numpy(self.query_embedding_matrix).float(),
            torch.from_numpy(self.llm_embedding_matrix).float(),
            torch.from_numpy(self.performance_matrix).float()
        )
    
    def get_train_val_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成训练/验证掩码
        
        按 query 划分：同一 query 的所有边要么都在训练集，要么都在验证集
        """
        num_edges = self.num_queries * self.num_llms
        
        # 按 query 划分
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
    
    def build_graph_data(
        self,
        query_embeddings: np.ndarray = None,
        performance_matrix: np.ndarray = None,
        is_train: bool = True
    ) -> Tuple[Any, Any]:
        """
        构建图数据
        
        Args:
            query_embeddings: Query embedding 矩阵，默认使用数据集中的
            performance_matrix: Performance 矩阵，默认使用数据集中的
            is_train: 是否为训练数据
        
        Returns:
            (train_data, val_data) 如果 is_train=True
            test_data 如果 is_train=False
        """
        if query_embeddings is None:
            query_embeddings = self.query_embedding_matrix
        if performance_matrix is None:
            performance_matrix = self.performance_matrix
        
        num_queries = len(query_embeddings)
        performance_flat = performance_matrix.flatten()
        
        # 构建边索引
        edge_org = [q for q in range(num_queries) for _ in range(self.num_llms)]
        edge_des = list(range(self.num_llms)) * num_queries
        
        # 创建标签：每个 query 的最佳 LLM 边为 1
        best_llm_indices = np.argmax(performance_matrix, axis=1)
        labels = np.eye(self.num_llms)[best_llm_indices].flatten()
        
        # 转换为 tensor
        query_tensor = torch.from_numpy(query_embeddings).float()
        llm_tensor = torch.from_numpy(self.llm_embedding_matrix).float()
        edge_weights = torch.from_numpy(performance_flat).float()
        labels_tensor = torch.from_numpy(labels).float()
        
        if is_train:
            train_mask, val_mask = self.get_train_val_masks()
            test_mask = torch.zeros(num_queries * self.num_llms, dtype=torch.bool)
            
            # 训练数据
            train_data = self.form_data.build(
                query_features=query_tensor,
                llm_features=llm_tensor,
                edge_org=edge_org,
                edge_des=edge_des,
                edge_weights=edge_weights,
                labels=labels_tensor,
                edge_mask=train_mask,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )
            
            # 验证数据（使用相同的图结构，不同的掩码）
            val_data = self.form_data.build(
                query_features=query_tensor,
                llm_features=llm_tensor,
                edge_org=edge_org,
                edge_des=edge_des,
                edge_weights=edge_weights,
                labels=labels_tensor,
                edge_mask=val_mask,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )
            
            return train_data, val_data
        else:
            # 测试数据
            num_edges = num_queries * self.num_llms
            test_mask = torch.ones(num_edges, dtype=torch.bool)
            train_mask = torch.zeros(num_edges, dtype=torch.bool)
            val_mask = torch.zeros(num_edges, dtype=torch.bool)
            
            test_data = self.form_data.build(
                query_features=query_tensor,
                llm_features=llm_tensor,
                edge_org=edge_org,
                edge_des=edge_des,
                edge_weights=edge_weights,
                labels=labels_tensor,
                edge_mask=test_mask,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )
            
            return test_data


def load_dataset(config: Dict) -> GraphRouterDataset:
    """
    从配置字典加载数据集
    
    Args:
        config: 包含 data_path 配置的字典
    
    Returns:
        GraphRouterDataset 实例
    """
    data_cfg = config.get("data_path", {})
    hparam = config.get("hparam", {})
    
    return GraphRouterDataset(
        routing_data_path=data_cfg["routing_data_train"],
        embedding_path=data_cfg["query_embedding_data"],
        llm_config_path=data_cfg["llm_data"],
        val_ratio=hparam.get("val_split_ratio", 0.2)
    )
