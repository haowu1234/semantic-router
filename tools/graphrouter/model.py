"""
GraphRouter GNN 模型定义

基于 PyTorch Geometric 实现的图神经网络模型，用于学习 Query-LLM 关系。

架构:
1. FeatureAlign: 对齐 Query/LLM 特征到相同隐空间
2. EncoderDecoderNet: 使用 GeneralConv 进行两层图卷积
3. GNNPredictor: 封装训练、验证和预测逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv
from torch_geometric.data import Data
from torch.optim import AdamW
from typing import Dict, Optional, Tuple


class FeatureAlign(nn.Module):
    """
    特征对齐层：将 Query 和 LLM 特征映射到相同的隐空间维度。
    
    Args:
        query_dim: Query embedding 维度
        llm_dim: LLM embedding 维度
        hidden_dim: 隐空间维度
    """
    
    def __init__(self, query_dim: int, llm_dim: int, hidden_dim: int):
        super().__init__()
        self.query_transform = nn.Linear(query_dim, hidden_dim)
        self.llm_transform = nn.Linear(llm_dim, hidden_dim)
    
    def forward(
        self, 
        query_features: torch.Tensor, 
        llm_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query_features: [num_queries, query_dim]
            llm_features: [num_llms, llm_dim]
        
        Returns:
            aligned_features: [num_queries + num_llms, hidden_dim]
        """
        aligned_query = self.query_transform(query_features)
        aligned_llm = self.llm_transform(llm_features)
        return torch.cat([aligned_query, aligned_llm], dim=0)


class EncoderDecoderNet(nn.Module):
    """
    编码-解码网络：使用 GNN 进行消息传递和边预测。
    
    架构（与 LLMRouter 原始实现一致）:
    1. FeatureAlign: 特征对齐到隐空间
    2. GeneralConv x 2: 两层图卷积进行消息传递
    3. Edge Prediction: score = sigmoid(mean(x_ini[query] * x_gnn[llm]))
    
    关键点：
    - Query 节点使用初始对齐特征 (x_ini)
    - LLM 节点使用 GNN 更新后的特征 (x_gnn)
    - 这允许 LLM 节点聚合来自多个训练 Query 的信息
    
    Args:
        query_dim: Query embedding 维度
        llm_dim: LLM embedding 维度
        hidden_dim: 隐藏层维度
        edge_dim: 边特征维度 (performance score)
    """
    
    def __init__(
        self, 
        query_dim: int, 
        llm_dim: int, 
        hidden_dim: int = 64,
        edge_dim: int = 1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.query_dim = query_dim
        self.llm_dim = llm_dim
        
        # 特征对齐层
        self.align = FeatureAlign(query_dim, llm_dim, hidden_dim)
        
        # GNN 卷积层
        # 注意：使用 mean 聚合避免值爆炸（LLMRouter 默认用 add，但我们数据量更大）
        self.conv1 = GeneralConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            in_edge_channels=edge_dim,
            aggr='mean'
        )
        self.conv2 = GeneralConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            in_edge_channels=edge_dim,
            aggr='mean'
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
        前向传播 - 与 LLMRouter 原始实现一致
        
        Args:
            query_features: [num_queries, query_dim] Query 特征
            llm_features: [num_llms, llm_dim] LLM 特征
            edge_index: [2, num_edges] 边索引
            edge_attr: [num_edges, edge_dim] 边权重 (performance)
            edge_mask: [num_edges] 需要预测的边 (bool)
            visible_mask: [num_edges] 可见的边，用于 GNN 消息传递 (bool)
        
        Returns:
            scores: [num_masked_edges] 预测的边分数
        """
        # 获取可见边的索引和权重（用于 GNN 消息传递）
        visible_edge_index = edge_index[:, visible_mask]
        visible_edge_attr = edge_attr[visible_mask]
        
        # 获取需要预测的边索引
        predict_edge_index = edge_index[:, edge_mask]
        
        # 边权重变换
        visible_edge_attr = F.leaky_relu(
            self.edge_mlp(visible_edge_attr.reshape(-1, self.edge_dim))
        )
        
        # 特征对齐：x_ini 是初始对齐特征
        x_ini = self.align(query_features, llm_features)
        
        # GNN 消息传递
        x1 = F.leaky_relu(self.bn1(
            self.conv1(x_ini, visible_edge_index, edge_attr=visible_edge_attr)
        ))
        x_gnn = self.bn2(
            self.conv2(x1, visible_edge_index, edge_attr=visible_edge_attr)
        )
        
        # 边预测（LLMRouter 核心逻辑）：
        # - 源节点（Query）使用初始特征 x_ini
        # - 目标节点（LLM）使用 GNN 更新后的特征 x_gnn
        src_features = x_ini[predict_edge_index[0]]  # Query 初始特征
        dst_features = x_gnn[predict_edge_index[1]]  # LLM GNN 特征
        
        # 点积 + Sigmoid
        scores = torch.sigmoid((src_features * dst_features).mean(dim=-1))
        
        return scores
    
    def forward_for_inference(
        self,
        query_features: torch.Tensor,
        llm_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        context_mask: torch.Tensor,
        new_query_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        推理前向传播 - 为新 Query 预测所有 LLM 的分数
        
        与训练的区别：
        - context_mask: 所有历史训练边（作为 GNN 消息传递的上下文）
        - new_query_indices: 需要预测的新 Query 索引
        
        Args:
            query_features: [num_queries, query_dim] 包含训练 + 新 Query
            llm_features: [num_llms, llm_dim] LLM 特征
            edge_index: [2, num_edges] 边索引（包含训练边 + 新 Query 边）
            edge_attr: [num_edges] 边权重（新 Query 边权重设为 0）
            context_mask: [num_edges] 用于 GNN 的上下文边（训练边）
            new_query_indices: [num_new_queries] 新 Query 的节点索引
        
        Returns:
            scores: [num_new_queries, num_llms] 每个新 Query 对所有 LLM 的预测分数
        """
        num_llms = llm_features.shape[0]
        num_queries = query_features.shape[0]
        
        # 获取上下文边（用于 GNN 消息传递）
        context_edge_index = edge_index[:, context_mask]
        context_edge_attr = edge_attr[context_mask]
        
        # 边权重变换
        context_edge_attr = F.leaky_relu(
            self.edge_mlp(context_edge_attr.reshape(-1, self.edge_dim))
        )
        
        # 特征对齐
        x_ini = self.align(query_features, llm_features)
        
        # GNN 消息传递（使用训练边作为上下文）
        x1 = F.leaky_relu(self.bn1(
            self.conv1(x_ini, context_edge_index, edge_attr=context_edge_attr)
        ))
        x_gnn = self.bn2(
            self.conv2(x1, context_edge_index, edge_attr=context_edge_attr)
        )
        
        # 为每个新 Query 计算对所有 LLM 的分数
        num_new_queries = len(new_query_indices)
        scores = torch.zeros(num_new_queries, num_llms, device=query_features.device)
        
        for i, q_idx in enumerate(new_query_indices):
            # Query 使用初始特征
            q_feature = x_ini[q_idx]  # [hidden_dim]
            
            # LLM 使用 GNN 更新后的特征
            llm_start_idx = num_queries  # LLM 节点从 num_queries 开始
            llm_features_gnn = x_gnn[llm_start_idx:llm_start_idx + num_llms]  # [num_llms, hidden_dim]
            
            # 计算分数
            scores[i] = torch.sigmoid((q_feature * llm_features_gnn).mean(dim=-1))
        
        return scores


class FormData:
    """
    数据格式化类：将原始数据转换为 PyG Data 对象。
    
    Args:
        device: 计算设备 (cpu/cuda)
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def build(
        self,
        query_features: torch.Tensor,
        llm_features: torch.Tensor,
        edge_org: list,
        edge_des: list,
        edge_weights: torch.Tensor,
        labels: torch.Tensor,
        edge_mask: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor
    ) -> Data:
        """
        构建 PyG Data 对象
        
        Args:
            query_features: [num_queries, query_dim]
            llm_features: [num_llms, llm_dim]
            edge_org: 源节点索引列表 (query 节点)
            edge_des: 目标节点索引列表 (llm 节点偏移后)
            edge_weights: [num_edges] 边权重
            labels: [num_edges] 标签 (最佳 LLM 为 1)
            edge_mask: [num_edges] 用于训练/预测的掩码
            train_mask: [num_edges] 训练边掩码
            val_mask: [num_edges] 验证边掩码
            test_mask: [num_edges] 测试边掩码
        """
        query_features = query_features.to(self.device)
        llm_features = llm_features.to(self.device)
        
        # 构建边索引：LLM 节点从 num_queries 开始编号
        num_queries = query_features.shape[0]
        des_node = [i + num_queries for i in edge_des]
        edge_index = torch.tensor([edge_org, des_node], dtype=torch.long).to(self.device)
        
        # 边权重
        edge_attr = edge_weights.reshape(-1, 1).float().to(self.device)
        
        return Data(
            query_features=query_features,
            llm_features=llm_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            labels=labels.float().to(self.device),
            edge_mask=edge_mask.to(self.device),
            train_mask=train_mask.to(self.device),
            val_mask=val_mask.to(self.device),
            test_mask=test_mask.to(self.device),
            num_queries=num_queries,
            num_llms=llm_features.shape[0]
        )


class GNNPredictor:
    """
    GNN 预测器：封装模型训练、验证和预测逻辑。
    
    Args:
        query_dim: Query embedding 维度
        llm_dim: LLM embedding 维度
        hidden_dim: 隐藏层维度
        config: 训练配置字典
        device: 计算设备
    """
    
    def __init__(
        self,
        query_dim: int,
        llm_dim: int,
        hidden_dim: int = 64,
        config: Optional[Dict] = None,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.llm_dim = llm_dim
        
        # 初始化模型
        self.model = EncoderDecoderNet(
            query_dim=query_dim,
            llm_dim=llm_dim,
            hidden_dim=hidden_dim,
            edge_dim=1
        ).to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # 损失函数：与 LLMRouter 一致，使用 BCELoss
        self.criterion = nn.BCELoss()
        
        # LLM 数量
        self.num_llms = self.config.get('num_llms', 10)
    
    def train_validate(
        self,
        train_data: Data,
        val_data: Data,
        save_path: Optional[str] = None
    ) -> float:
        """
        训练和验证模型
        
        Args:
            train_data: 训练数据 (PyG Data)
            val_data: 验证数据 (PyG Data)
            save_path: 模型保存路径
        
        Returns:
            best_result: 最佳验证结果
        """
        best_result = -1.0
        best_state = None
        
        train_epochs = self.config.get('train_epoch', 100)
        batch_size = self.config.get('batch_size', 4)
        mask_rate = self.config.get('train_mask_rate', 0.3)
        
        # 尝试使用 tqdm 显示进度条
        try:
            from tqdm import tqdm
            epoch_iter = tqdm(range(train_epochs), desc="Training", unit="epoch")
        except ImportError:
            epoch_iter = range(train_epochs)
            print(f"Install tqdm for progress bar: pip install tqdm")
        
        for epoch in epoch_iter:
            self._last_epoch = epoch  # 用于调试
            # 训练阶段
            self.model.train()
            total_loss = 0.0
            
            for _ in range(batch_size):
                # 与 LLMRouter 完全一致的掩码逻辑
                # mask_train 一开始全是 True（训练边）
                mask = train_data.train_mask.clone().bool()
                
                # 随机选择约 mask_rate 的边（这些边用于 GNN，不预测）
                random_mask = torch.rand(mask.size(), device=self.device) < mask_rate
                
                # LLMRouter 逻辑：如果 mask=True 且 random=True，则设为 False
                # 即：random_mask=True 的边不需要预测
                mask = torch.where(mask & random_mask, torch.tensor(False, device=self.device), mask)
                
                # edge_can_see: 训练边中，不需要预测的边（用于 GNN 消息传递）
                visible_mask = ~mask & train_data.train_mask.bool()
                
                # edge_mask: 需要预测的边
                edge_mask = mask
                
                self.optimizer.zero_grad()
                
                scores = self.model(
                    query_features=train_data.query_features,
                    llm_features=train_data.llm_features,
                    edge_index=train_data.edge_index,
                    edge_attr=train_data.edge_attr,
                    edge_mask=edge_mask,
                    visible_mask=visible_mask
                )
                
                # 使用 BCELoss（与 LLMRouter 一致）
                # labels 是 one-hot：最佳 LLM 的边为 1，其他为 0
                labels = train_data.labels[edge_mask]
                loss = self.criterion(scores, labels)
                
                total_loss += loss.item()
                loss.backward()
            
            self.optimizer.step()
            avg_loss = total_loss / batch_size
            
            # 验证阶段
            self.model.eval()
            val_result = self._validate(val_data, train_data)
            
            # 保存最佳模型
            if val_result > best_result:
                best_result = val_result
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            # 更新进度条或打印日志
            if hasattr(epoch_iter, 'set_postfix'):
                epoch_iter.set_postfix(loss=f"{avg_loss:.4f}", val=f"{val_result:.4f}", best=f"{best_result:.4f}")
            elif epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_result={val_result:.4f}, best={best_result:.4f}")
        
        # 恢复最佳模型
        if best_state:
            self.model.load_state_dict(best_state)
        
        # 保存模型
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        print(f"Training completed. Best validation result: {best_result:.4f}")
        return best_result
    
    def _validate(self, val_data: Data, train_data: Data) -> float:
        """
        验证模型（与 LLMRouter 完全一致）
        
        关键：验证数据和训练数据使用同一个图结构，
        只是通过 mask 区分哪些边用于训练，哪些边用于验证
        """
        with torch.no_grad():
            val_mask = val_data.val_mask.bool()
            # 训练边作为可见边（用于 GNN 消息传递）
            visible_mask = train_data.train_mask.bool()
            
            scores = self.model(
                query_features=val_data.query_features,
                llm_features=val_data.llm_features,
                edge_index=val_data.edge_index,
                edge_attr=val_data.edge_attr,
                edge_mask=val_mask,
                visible_mask=visible_mask
            )
            
            # 计算验证指标
            scores = scores.reshape(-1, self.num_llms)
            val_perf = val_data.edge_attr[val_mask].reshape(-1, self.num_llms)
            
            # 选择得分最高的模型
            selected_idx = scores.argmax(dim=1)
            row_indices = torch.arange(len(selected_idx), device=self.device)
            selected_perf = val_perf[row_indices, selected_idx]
            
            # Debug: 检查选择分布
            unique, counts = torch.unique(selected_idx, return_counts=True)
            if hasattr(self, '_last_epoch') and self._last_epoch % 20 == 0:
                print(f"    LLM selection distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
            
            return selected_perf.mean().item()
    
    def predict(self, data: Data) -> torch.Tensor:
        """
        预测模式：返回每个 query 的最佳 LLM 索引
        
        与 LLMRouter 一致：
        - data 应该是由 _build_inference_graph 构建的推理图
        - train_mask 表示 context 边（用于 GNN）
        - val_mask/test_mask 表示需要预测的边
        
        Args:
            data: 推理数据（包含 context Query + new Query）
        
        Returns:
            max_idx: [num_new_queries] 最佳 LLM 索引
        """
        self.model.eval()
        
        with torch.no_grad():
            # 确定需要预测的边
            if data.test_mask.any():
                predict_mask = data.test_mask.bool()
            else:
                predict_mask = data.val_mask.bool()
            
            # context 边用于 GNN 消息传递
            visible_mask = data.train_mask.bool()
            
            scores = self.model(
                query_features=data.query_features,
                llm_features=data.llm_features,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                edge_mask=predict_mask,
                visible_mask=visible_mask
            )
        
        scores = scores.reshape(-1, self.num_llms)
        return scores.argmax(dim=1)
    
    def load_model(self, path: str):
        """加载模型权重"""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {path}")
