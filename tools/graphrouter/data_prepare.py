#!/usr/bin/env python3
"""
GraphRouter 数据准备工具

将各种来源的数据转换为 GraphRouter 训练所需的格式。

支持的数据来源:
1. LLMRouter 项目数据
2. 自定义 CSV/JSON 数据
3. 模拟数据（用于测试）

用法:
    # 从 LLMRouter 数据转换
    python data_prepare.py --source llmrouter --input /path/to/llmrouter/data --output ./data

    # 生成模拟数据
    python data_prepare.py --source mock --output ./data --num-queries 1000 --num-llms 5
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

sys.path.insert(0, str(Path(__file__).parent))

# pandas is optional, only needed for CSV conversion
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def generate_mock_data(
    output_dir: str,
    num_queries: int = 1000,
    num_llms: int = 5,
    embedding_dim: int = 768,
    seed: int = 42
):
    """
    生成模拟数据用于测试
    
    Args:
        output_dir: 输出目录
        num_queries: 查询数量
        num_llms: LLM 数量
        embedding_dim: embedding 维度
        seed: 随机种子
    """
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating mock data with {num_queries} queries and {num_llms} LLMs...")
    
    # 1. 生成 LLM 配置
    model_names = [f"model-{i}" for i in range(num_llms)]
    llm_config = {}
    for i, name in enumerate(model_names):
        llm_config[name] = {
            "model": name,
            "embedding": np.random.randn(embedding_dim).tolist(),
            "cost_per_1k_tokens": 0.01 * (i + 1)
        }
    
    llm_config_path = output_dir / "llm_candidates.json"
    with open(llm_config_path, 'w') as f:
        json.dump(llm_config, f, indent=2)
    print(f"  Saved LLM config to {llm_config_path}")
    
    # 2. 生成 Query Embeddings
    query_embeddings = np.random.randn(num_queries, embedding_dim).astype(np.float32)
    embeddings_path = output_dir / "query_embeddings.pt"
    torch.save(torch.from_numpy(query_embeddings), embeddings_path)
    print(f"  Saved query embeddings to {embeddings_path}")
    
    # 3. 生成路由数据
    # 模拟不同 LLM 在不同类型查询上的性能差异
    routing_records = []
    
    for q_idx in range(num_queries):
        query_text = f"This is test query number {q_idx}"
        
        # 为每个 LLM 生成性能分数
        # 使用一些结构化的随机性，使得某些 LLM 在某些查询上更好
        base_perfs = np.random.beta(2, 2, size=num_llms)  # 0-1 之间
        
        # 添加一些噪声
        noise = np.random.normal(0, 0.1, size=num_llms)
        perfs = np.clip(base_perfs + noise, 0, 1)
        
        for llm_idx, name in enumerate(model_names):
            routing_records.append({
                "query": query_text,
                "model_name": name,
                "performance": float(perfs[llm_idx]),
                "embedding_id": q_idx
            })
    
    routing_path = output_dir / "routing_data.jsonl"
    with open(routing_path, 'w') as f:
        for record in routing_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"  Saved routing data to {routing_path}")
    
    # 4. 生成配置文件
    config = {
        "data_path": {
            "routing_data_train": str(routing_path),
            "query_embedding_data": str(embeddings_path),
            "llm_data": str(llm_config_path)
        },
        "hparam": {
            "hidden_dim": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "train_epoch": 50,
            "batch_size": 4,
            "train_mask_rate": 0.3,
            "val_split_ratio": 0.2,
            "temperature": 1.0
        }
    }
    
    config_path = output_dir / "config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Saved config to {config_path}")
    
    print(f"\nMock data generated successfully!")
    print(f"To train, run:")
    print(f"  python train.py --config {config_path} --output model.json")


def convert_from_llmrouter(
    input_dir: str,
    output_dir: str,
    task_name: Optional[str] = None
):
    """
    从 LLMRouter 项目数据格式转换
    
    Args:
        input_dir: LLMRouter 数据目录
        output_dir: 输出目录
        task_name: 任务名称（如果指定，只转换该任务的数据）
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting data from LLMRouter format...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    
    # 查找数据文件
    routing_files = list(input_dir.glob("**/routing_data*.jsonl")) + \
                   list(input_dir.glob("**/*_routing*.jsonl"))
    embedding_files = list(input_dir.glob("**/query_embeddings*.pt")) + \
                     list(input_dir.glob("**/*_embeddings*.pt"))
    llm_files = list(input_dir.glob("**/llm*.json")) + \
               list(input_dir.glob("**/*_llm*.json"))
    
    if not routing_files:
        print("  Error: No routing data files found!")
        return
    
    print(f"  Found {len(routing_files)} routing files")
    print(f"  Found {len(embedding_files)} embedding files")
    print(f"  Found {len(llm_files)} LLM config files")
    
    # 合并路由数据
    all_records = []
    for rf in routing_files:
        with open(rf, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if task_name is None or record.get('task_name') == task_name:
                        all_records.append(record)
    
    print(f"  Total routing records: {len(all_records)}")
    
    # 保存路由数据
    routing_path = output_dir / "routing_data.jsonl"
    with open(routing_path, 'w') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # 复制 embedding 文件
    if embedding_files:
        import shutil
        embeddings_path = output_dir / "query_embeddings.pt"
        shutil.copy(embedding_files[0], embeddings_path)
        print(f"  Copied embeddings to {embeddings_path}")
    
    # 复制 LLM 配置
    if llm_files:
        import shutil
        llm_path = output_dir / "llm_candidates.json"
        shutil.copy(llm_files[0], llm_path)
        print(f"  Copied LLM config to {llm_path}")
    
    print(f"\nConversion completed!")


def convert_from_csv(
    input_file: str,
    output_dir: str,
    query_col: str = "query",
    model_col: str = "model_name",
    perf_col: str = "performance"
):
    """
    从 CSV 文件转换
    
    Args:
        input_file: CSV 文件路径
        output_dir: 输出目录
        query_col: 查询列名
        model_col: 模型列名
        perf_col: 性能列名
    """
    if not HAS_PANDAS:
        print("Error: pandas is required for CSV conversion!")
        print("Install it with: pip install pandas")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting data from CSV...")
    
    # 读取 CSV
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} records from {input_file}")
    
    # 获取唯一查询
    unique_queries = df[query_col].unique().tolist()
    query_to_idx = {q: i for i, q in enumerate(unique_queries)}
    
    # 转换为路由数据格式
    records = []
    for _, row in df.iterrows():
        records.append({
            "query": row[query_col],
            "model_name": row[model_col],
            "performance": float(row[perf_col]),
            "embedding_id": query_to_idx[row[query_col]]
        })
    
    # 保存
    routing_path = output_dir / "routing_data.jsonl"
    with open(routing_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"  Saved routing data to {routing_path}")
    print(f"  Note: You need to generate query embeddings separately!")
    print(f"  Unique queries: {len(unique_queries)}")


def generate_embeddings(
    queries_file: str,
    output_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    为查询生成 embeddings
    
    需要安装 sentence-transformers: pip install sentence-transformers
    
    Args:
        queries_file: 包含查询的文件（每行一个查询或 JSONL）
        output_path: 输出 embedding 文件路径
        model_name: Sentence Transformer 模型名称
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed!")
        print("Install it with: pip install sentence-transformers")
        return
    
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    
    # 加载查询
    queries = []
    with open(queries_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    queries.append(data.get('query', line))
                except json.JSONDecodeError:
                    queries.append(line)
    
    # 去重并保持顺序
    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)
    
    print(f"Generating embeddings for {len(unique_queries)} unique queries...")
    embeddings = model.encode(unique_queries, show_progress_bar=True)
    
    # 保存
    torch.save(torch.from_numpy(embeddings), output_path)
    print(f"Saved embeddings to {output_path}")
    print(f"  Shape: {embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for GraphRouter training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Mock data
    mock_parser = subparsers.add_parser('mock', help='Generate mock data for testing')
    mock_parser.add_argument('--output', required=True, help='Output directory')
    mock_parser.add_argument('--num-queries', type=int, default=1000, help='Number of queries')
    mock_parser.add_argument('--num-llms', type=int, default=5, help='Number of LLMs')
    mock_parser.add_argument('--embedding-dim', type=int, default=768, help='Embedding dimension')
    mock_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Convert from LLMRouter
    llmrouter_parser = subparsers.add_parser('llmrouter', help='Convert from LLMRouter format')
    llmrouter_parser.add_argument('--input', required=True, help='LLMRouter data directory')
    llmrouter_parser.add_argument('--output', required=True, help='Output directory')
    llmrouter_parser.add_argument('--task', default=None, help='Task name filter')
    
    # Convert from CSV
    csv_parser = subparsers.add_parser('csv', help='Convert from CSV format')
    csv_parser.add_argument('--input', required=True, help='Input CSV file')
    csv_parser.add_argument('--output', required=True, help='Output directory')
    csv_parser.add_argument('--query-col', default='query', help='Query column name')
    csv_parser.add_argument('--model-col', default='model_name', help='Model column name')
    csv_parser.add_argument('--perf-col', default='performance', help='Performance column name')
    
    # Generate embeddings
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings for queries')
    embed_parser.add_argument('--input', required=True, help='Queries file')
    embed_parser.add_argument('--output', required=True, help='Output embeddings path')
    embed_parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2', 
                             help='Sentence Transformer model')
    
    args = parser.parse_args()
    
    if args.command == 'mock':
        generate_mock_data(
            output_dir=args.output,
            num_queries=args.num_queries,
            num_llms=args.num_llms,
            embedding_dim=args.embedding_dim,
            seed=args.seed
        )
    elif args.command == 'llmrouter':
        convert_from_llmrouter(
            input_dir=args.input,
            output_dir=args.output,
            task_name=args.task
        )
    elif args.command == 'csv':
        convert_from_csv(
            input_file=args.input,
            output_dir=args.output,
            query_col=args.query_col,
            model_col=args.model_col,
            perf_col=args.perf_col
        )
    elif args.command == 'embed':
        generate_embeddings(
            queries_file=args.input,
            output_path=args.output,
            model_name=args.model
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
