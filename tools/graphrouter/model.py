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
    编码-解码网络：学习 Query-LLM 匹配关系。
    
    简化架构（与 Go 推理一致）:
    1. Query Projection: query_features -> hidden_dim
    2. LLM Representation: 可学习的 LLM 表示
    3. Cosine Similarity: 计算 Query-LLM 相似度
    
    训练和推理使用相同的逻辑，确保一致性。
    
    Args:
        query_dim: Query embedding 维度
        llm_dim: LLM embedding 维度
        hidden_dim: 隐藏层维度
        edge_dim: 边特征维度 (未使用，保留接口兼容性)
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
        
        # Query 投影层（与 Go 推理一致）
        self.align = FeatureAlign(query_dim, llm_dim, hidden_dim)
        
        # 保留 GNN 层用于可选的特征增强（但主要依赖直接投影）
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
        前向传播 - 使用与 Go 推理一致的逻辑
        
        Args:
            query_features: [num_queries, query_dim] Query 特征
            llm_features: [num_llms, llm_dim] LLM 特征
            edge_index: [2, num_edges] 边索引
            edge_attr: [num_edges, edge_dim] 边权重 (performance)
            edge_mask: [num_edges] 需要预测的边 (bool)
            visible_mask: [num_edges] 可见的边 (bool) - 未使用，保留接口
        
        Returns:
            scores: [num_masked_edges] 预测的边分数
        """
        num_queries = query_features.shape[0]
        
        # 特征对齐（与 Go 推理一致）
        x0 = self.align(query_features, llm_features)
        
        # 分离 Query 和 LLM 的隐空间表示
        query_hidden = x0[:num_queries]  # [num_queries, hidden_dim]
        llm_hidden = x0[num_queries:]    # [num_llms, hidden_dim]
        
        # L2 归一化（与 Go cosine similarity 一致）
        query_hidden = F.normalize(query_hidden, p=2, dim=1)
        llm_hidden = F.normalize(llm_hidden, p=2, dim=1)
        
        # 获取需要预测的边
        predict_edge_index = edge_index[:, edge_mask]
        
        # 计算点积分数（归一化后等价于 cosine similarity）
        src_features = query_hidden[predict_edge_index[0]]  # Query features
        # LLM 节点索引需要减去 num_queries 偏移
        llm_indices = predict_edge_index[1] - num_queries
        dst_features = llm_hidden[llm_indices]  # LLM features
        
        # 点积打分
        scores = (src_features * dst_features).sum(dim=-1)
        
        # Sigmoid 映射到 [0, 1]
        scores = torch.sigmoid(scores)
        
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
        
        # 损失函数：使用 MarginRankingLoss 学习排序
        self.criterion = nn.MarginRankingLoss(margin=0.1)
        
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
                # 按 query 为单位随机掩码（确保每个 query 的所有 LLM 边同时被 mask 或不被 mask）
                train_mask = train_data.train_mask.clone().bool()
                num_total_edges = train_mask.size(0)
                num_queries = num_total_edges // self.num_llms
                
                # 为每个 query 生成一个随机值，然后扩展到所有 LLM 边
                query_random = torch.rand(num_queries, device=self.device) < mask_rate
                # 扩展：每个 query 的决定应用到其所有 num_llms 条边
                edge_random_mask = query_random.repeat_interleave(self.num_llms)
                
                edge_mask = train_mask & edge_random_mask
                visible_mask = train_mask & ~edge_random_mask
                
                self.optimizer.zero_grad()
                
                scores = self.model(
                    query_features=train_data.query_features,
                    llm_features=train_data.llm_features,
                    edge_index=train_data.edge_index,
                    edge_attr=train_data.edge_attr,
                    edge_mask=edge_mask,
                    visible_mask=visible_mask
                )
                
                # Reshape scores 和 performance 为 [num_queries_masked, num_llms]
                scores_reshaped = scores.reshape(-1, self.num_llms)
                perf = train_data.edge_attr[edge_mask].reshape(-1, self.num_llms)
                
                # ListNet 风格损失：使用 KL 散度让预测分布接近真实分布
                # 真实分布：基于 performance 的 softmax
                # 预测分布：基于 score 的 softmax
                temperature = 1.0
                target_dist = F.softmax(perf / temperature, dim=1)
                pred_dist = F.log_softmax(scores_reshaped / temperature, dim=1)
                
                # KL 散度损失
                loss = F.kl_div(pred_dist, target_dist, reduction='batchmean')
                
                total_loss += loss.item()
                loss.backward()
            
            self.optimizer.step()
            avg_loss = total_loss / batch_size
            
            # 验证阶段
            self.model.eval()
            val_result = self._validate(val_data, train_data.train_mask)
            
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
    
    def _validate(self, val_data: Data, train_mask: torch.Tensor) -> float:
        """验证模型"""
        with torch.no_grad():
            val_mask = val_data.val_mask.bool()
            visible_mask = train_mask.bool()
            
            scores = self.model(
                query_features=val_data.query_features,
                llm_features=val_data.llm_features,
                edge_index=val_data.edge_index,
                edge_attr=val_data.edge_attr,
                edge_mask=val_mask,
                visible_mask=visible_mask
            )
            
            # 计算验证指标：选择的模型的平均 performance
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
        
        Args:
            data: 预测数据
        
        Returns:
            max_idx: [num_queries] 最佳 LLM 索引
        """
        self.model.eval()
        
        with torch.no_grad():
            test_mask = data.test_mask.bool()
            visible_mask = (data.train_mask | data.val_mask).bool()
            
            scores = self.model(
                query_features=data.query_features,
                llm_features=data.llm_features,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                edge_mask=test_mask,
                visible_mask=visible_mask
            )
        
        scores = scores.reshape(-1, self.num_llms)
        return scores.argmax(dim=1)
    
    def load_model(self, path: str):
        """加载模型权重"""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {path}")
