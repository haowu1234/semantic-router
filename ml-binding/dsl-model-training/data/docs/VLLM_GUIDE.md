# 使用本地 vLLM 服务器生成训练数据

## 快速开始

### 1. 在远程服务器上启动 vLLM

```bash
# 推荐模型: Qwen2.5-72B-Instruct (最佳中英文能力)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000 \
    --max-model-len 8192

# 或者使用较小的模型 (资源受限时)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000

# 或者使用 DeepSeek
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --port 8000
```

### 2. 建立网络连接

**方式 A: 直接连接 (同一网络)**
```bash
# 假设远程服务器 IP 为 192.168.1.100
export VLLM_URL="http://192.168.1.100:8000"
```

**方式 B: SSH 隧道 (推荐，更安全)**
```bash
# 在本地机器上建立隧道
ssh -L 8000:localhost:8000 user@remote-server

# 然后使用 localhost
export VLLM_URL="http://localhost:8000"
```

### 3. 生成数据

```bash
cd ml-binding/dsl-model-training/data

# 首先生成合成 DSL (如果还没有)
make synthetic

# 使用 vLLM 生成 NL 描述
python scripts/nl_generator.py \
    --api vllm \
    --vllm-url $VLLM_URL \
    --input synthetic/ \
    --output nl_pairs/ \
    --temperature 0.8

# 或者使用便捷脚本
bash scripts/generate_with_vllm.sh --url $VLLM_URL
```

## 完整数据生成流程

```bash
cd ml-binding/dsl-model-training/data

# Step 1: 提取种子数据
python scripts/extract_seeds.py \
    --repo-root ../../.. \
    --output seeds/

# Step 2: 生成合成 DSL (10000 个样本)
python scripts/cfg_generator.py \
    --count 10000 \
    --output synthetic/

# Step 3: 使用 vLLM 生成 NL 描述
python scripts/nl_generator.py \
    --api vllm \
    --vllm-url http://localhost:8000 \
    --input synthetic/ \
    --output nl_pairs/

# Step 4: 生成负样本 (用于 DPO)
python scripts/negative_sampler.py \
    --input synthetic/ \
    --output negative/

# Step 5: 验证并构建最终数据集
python scripts/validator.py \
    --seeds seeds/ \
    --synthetic synthetic/ \
    --nl-pairs nl_pairs/ \
    --negative negative/ \
    --output final/
```

## 参数说明

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--api vllm` | - | 使用 vLLM 后端 |
| `--vllm-url` | `http://localhost:8000` | vLLM 服务器地址 |
| `--model` | 自动检测 | 模型名称 (留空自动从服务器获取) |
| `--temperature` | 0.8 | 采样温度 (越高越多样) |
| `--limit` | 无限制 | 处理样本数限制 |
| `--resume` | false | 断点续传模式 |
| `--batch-size` | 100 | 保存检查点间隔 |

## 推荐模型

| 模型 | 显存需求 | 中文能力 | 推荐度 |
|:---|:---|:---|:---|
| Qwen2.5-72B-Instruct | ~150GB (TP4) | ⭐⭐⭐⭐⭐ | 最佳 |
| Qwen2.5-32B-Instruct | ~70GB (TP2) | ⭐⭐⭐⭐⭐ | 推荐 |
| DeepSeek-V3 | ~200GB (TP8) | ⭐⭐⭐⭐⭐ | 最强 |
| Qwen2.5-14B-Instruct | ~30GB | ⭐⭐⭐⭐ | 性价比 |
| Llama-3.1-70B-Instruct | ~150GB (TP4) | ⭐⭐⭐ | 英文优先 |

## 性能预估

使用 Qwen2.5-72B-Instruct (4×A100 80GB):

| 数据量 | 预估时间 | 输出 |
|:---|:---|:---|
| 1,000 DSL | ~30 分钟 | 6,000 NL-DSL 对 |
| 10,000 DSL | ~5 小时 | 60,000 NL-DSL 对 |
| 种子 250 + 合成 10,000 | ~5.5 小时 | 61,500 NL-DSL 对 |

## 断点续传

如果生成过程中断，可以使用 `--resume` 恢复:

```bash
python scripts/nl_generator.py \
    --api vllm \
    --vllm-url $VLLM_URL \
    --input synthetic/ \
    --output nl_pairs/ \
    --resume
```

脚本会自动跳过已处理的样本，从上次中断处继续。

## 输出格式

每种风格的 NL 描述保存为单独的 JSONL 文件:

```
nl_pairs/
├── en_formal.jsonl      # 英文正式
├── en_casual.jsonl      # 英文口语
├── en_technical.jsonl   # 英文技术
├── zh_formal.jsonl      # 中文正式
├── zh_casual.jsonl      # 中文口语
├── ambiguous.jsonl      # 模糊描述
└── generation_stats.json
```

每行一个样本:
```json
{
  "id": "cfg_L3_00001_en_casual",
  "instruction": "Convert the following natural language description into Signal DSL configuration.",
  "input": "Set up a router that blocks jailbreak attempts and routes math questions to deepseek-r1.",
  "output": "SIGNAL jailbreak jb1 {\n  threshold: 0.9\n}\n...",
  "style": "en_casual",
  "complexity": "L3",
  "source_id": "cfg_L3_00001",
  "valid": true
}
```
