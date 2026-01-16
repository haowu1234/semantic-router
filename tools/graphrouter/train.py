#!/usr/bin/env python3
"""
GraphRouter 训练入口

用法:
    python train.py --config configs/default.yaml --output model.json

完整示例:
    python train.py \
        --config configs/default.yaml \
        --output ../../config/models/graph_model.json \
        --checkpoint models/graphrouter.pt \
        --temperature 1.0
"""

import argparse
import yaml
import sys
from pathlib import Path

# 添加当前目录到 path（用于直接运行脚本）
sys.path.insert(0, str(Path(__file__).parent))

from data import GraphRouterDataset
from trainer import GraphRouterTrainer
from export import export_for_go, validate_export


def main():
    parser = argparse.ArgumentParser(
        description="Train GraphRouter GNN model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用默认配置训练
  python train.py --config configs/default.yaml --output model.json

  # 指定温度参数
  python train.py --config configs/default.yaml --output model.json --temperature 0.5

  # 保存 PyTorch checkpoint
  python train.py --config configs/default.yaml --output model.json --checkpoint models/gnn.pt
        """
    )
    parser.add_argument(
        "--config", 
        required=True, 
        help="Config YAML path"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output JSON path for Go inference"
    )
    parser.add_argument(
        "--checkpoint", 
        default=None, 
        help="PyTorch checkpoint path (optional)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Softmax temperature (override config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (override config)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate exported model, no training"
    )
    
    args = parser.parse_args()
    
    # 验证模式
    if args.validate_only:
        validate_export(args.output)
        return
    
    # 加载配置
    print(f"Loading config from {args.config}...")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖
    if args.temperature is not None:
        config.setdefault("hparam", {})["temperature"] = args.temperature
    if args.epochs is not None:
        config.setdefault("hparam", {})["train_epoch"] = args.epochs
    
    data_cfg = config.get("data_path", {})
    hparam = config.get("hparam", {})
    
    # 加载数据集
    print("\n" + "=" * 60)
    print("Step 1: Loading dataset")
    print("=" * 60)
    
    dataset = GraphRouterDataset(
        routing_data_path=data_cfg["routing_data_train"],
        embedding_path=data_cfg["query_embedding_data"],
        llm_config_path=data_cfg["llm_data"],
        val_ratio=hparam.get("val_split_ratio", 0.2)
    )
    
    print(f"\nDataset loaded:")
    print(f"  - {dataset.num_queries} queries")
    print(f"  - {dataset.num_llms} LLMs: {dataset.model_names}")
    print(f"  - Query dim: {dataset.query_dim}")
    print(f"  - LLM dim: {dataset.llm_dim}")
    
    # 初始化训练器
    print("\n" + "=" * 60)
    print("Step 2: Initializing trainer")
    print("=" * 60)
    
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
    print("\n" + "=" * 60)
    print("Step 3: Training")
    print("=" * 60)
    
    best_result = trainer.train(save_path=args.checkpoint)
    
    # 评估
    print("\n" + "=" * 60)
    print("Step 4: Evaluation")
    print("=" * 60)
    
    metrics = trainer.evaluate()
    print(f"\nEvaluation metrics:")
    print(f"  - Routing accuracy: {metrics['routing_accuracy']:.4f}")
    print(f"  - Average performance: {metrics['avg_performance']:.4f}")
    print(f"  - Oracle performance: {metrics['oracle_performance']:.4f}")
    print(f"  - Oracle gap: {metrics['oracle_gap']:.4f}")
    
    # 导出为 Go 格式
    print("\n" + "=" * 60)
    print("Step 5: Exporting model for Go")
    print("=" * 60)
    
    temperature = hparam.get("temperature", 1.0)
    if args.temperature is not None:
        temperature = args.temperature
    
    export_for_go(
        model=trainer.get_model(),
        dataset=dataset,
        output_path=args.output,
        train_accuracy=metrics['routing_accuracy'],
        temperature=temperature,
        metadata={
            "avg_performance": metrics['avg_performance'],
            "oracle_gap": metrics['oracle_gap'],
            "config_file": args.config
        }
    )
    
    # 验证导出
    print("\n" + "=" * 60)
    print("Step 6: Validating export")
    print("=" * 60)
    
    validate_export(args.output)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
