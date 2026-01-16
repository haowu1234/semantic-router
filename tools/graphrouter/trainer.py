"""
GraphRouter 训练器

封装完整的训练流程，支持配置化训练。
"""

import torch
from typing import Dict, Optional
from pathlib import Path

try:
    from .model import GNNPredictor
    from .data import GraphRouterDataset
except ImportError:
    from model import GNNPredictor
    from data import GraphRouterDataset


class GraphRouterTrainer:
    """
    GraphRouter 训练器
    
    提供完整的训练流程：
    1. 初始化模型和优化器
    2. 构建训练/验证数据
    3. 训练循环
    4. 模型保存
    
    Args:
        dataset: GraphRouterDataset 实例
        hidden_dim: 隐藏层维度
        learning_rate: 学习率
        weight_decay: 权重衰减
        epochs: 训练轮数
        batch_size: 批大小
        mask_rate: 边掩码率
        device: 计算设备
    """
    
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
        
        print(f"Using device: {self.device}")
        
        # 训练配置
        self.config = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'train_epoch': epochs,
            'batch_size': batch_size,
            'train_mask_rate': mask_rate,
            'num_llms': dataset.num_llms,
        }
        
        # 初始化 GNNPredictor
        self.predictor = GNNPredictor(
            query_dim=dataset.query_dim,
            llm_dim=dataset.llm_dim,
            hidden_dim=hidden_dim,
            config=self.config,
            device=self.device
        )
        
        # 构建训练数据
        print("Building graph data...")
        self.train_data, self.val_data = dataset.build_graph_data(is_train=True)
        print(f"  Train edges: {self.train_data.train_mask.sum().item()}")
        print(f"  Val edges: {self.val_data.val_mask.sum().item()}")
    
    def train(self, save_path: Optional[str] = None) -> float:
        """
        训练模型
        
        Args:
            save_path: PyTorch 模型保存路径
        
        Returns:
            best_result: 最佳验证结果
        """
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Mask rate: {self.mask_rate}")
        print()
        
        best_result = self.predictor.train_validate(
            train_data=self.train_data,
            val_data=self.val_data,
            save_path=save_path
        )
        
        return best_result
    
    def evaluate(self, test_data=None) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            test_data: 测试数据，默认使用验证数据
        
        Returns:
            评估指标字典
        """
        if test_data is None:
            test_data = self.val_data
        
        self.predictor.model.eval()
        
        with torch.no_grad():
            num_llms = self.dataset.num_llms
            
            # 确定需要预测的边
            if test_data.test_mask.any():
                eval_mask = test_data.test_mask.bool()
            else:
                eval_mask = test_data.val_mask.bool()
            
            # 训练边用于 GNN
            visible_mask = self.train_data.train_mask.bool()
            
            scores = self.predictor.model(
                query_features=test_data.query_features,
                llm_features=test_data.llm_features,
                edge_index=test_data.edge_index,
                edge_attr=test_data.edge_attr,
                edge_mask=eval_mask,
                visible_mask=visible_mask
            )
            
            # 计算指标
            scores = scores.reshape(-1, num_llms)
            predictions = scores.argmax(dim=1)
            
            # 获取真实性能
            perf = test_data.edge_attr[eval_mask].reshape(-1, num_llms)
            
            # 真实最佳 LLM
            true_best = perf.argmax(dim=1)
            
            # 计算准确率
            correct = (predictions == true_best).sum().item()
            total = len(predictions)
            accuracy = correct / total if total > 0 else 0.0
            
            # 计算选择的 LLM 的平均性能
            row_indices = torch.arange(len(predictions), device=self.device)
            selected_perf = perf[row_indices, predictions]
            avg_perf = selected_perf.mean().item()
            
            # Oracle 性能（总是选最佳）
            oracle_perf = perf.max(dim=1)[0].mean().item()
            
            # 性能差距
            oracle_gap = oracle_perf - avg_perf
            
            # 打印选择分布
            unique, counts = torch.unique(predictions, return_counts=True)
            print(f"  LLM selection distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
        
        return {
            'routing_accuracy': accuracy,
            'avg_performance': avg_perf,
            'oracle_performance': oracle_perf,
            'oracle_gap': oracle_gap,
            'num_samples': total
        }
    
    def get_model(self):
        """获取训练好的模型"""
        return self.predictor.model
    
    def load_model(self, path: str):
        """加载模型"""
        self.predictor.load_model(path)


def create_trainer_from_config(config: Dict) -> GraphRouterTrainer:
    """
    从配置字典创建训练器
    
    Args:
        config: 完整配置字典
    
    Returns:
        GraphRouterTrainer 实例
    """
    try:
        from .data import load_dataset
    except ImportError:
        from data import load_dataset
    
    dataset = load_dataset(config)
    hparam = config.get("hparam", {})
    
    return GraphRouterTrainer(
        dataset=dataset,
        hidden_dim=hparam.get("hidden_dim", 64),
        learning_rate=hparam.get("learning_rate", 0.001),
        weight_decay=hparam.get("weight_decay", 1e-4),
        epochs=hparam.get("train_epoch", 100),
        batch_size=hparam.get("batch_size", 4),
        mask_rate=hparam.get("train_mask_rate", 0.3)
    )
