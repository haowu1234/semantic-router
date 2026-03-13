#!/bin/bash
# Signal DSL Model Server - vLLM 启动脚本
# 使用 vLLM 提供 OpenAI 兼容的 API

set -e

# ============== 配置 ==============

# 默认配置
DEFAULT_PORT=8080
DEFAULT_MODEL_NAME="dsl-generator"
DEFAULT_BASE_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"

# 从命令行参数获取
CHECKPOINT_PATH="${1:-}"
PORT="${2:-$DEFAULT_PORT}"
MODEL_NAME="${3:-$DEFAULT_MODEL_NAME}"

# ============== 帮助信息 ==============

show_help() {
    echo "Usage: $0 <checkpoint_path> [port] [model_name]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path  Path to the model checkpoint (required)"
    echo "  port             Server port (default: $DEFAULT_PORT)"
    echo "  model_name       Model name for API (default: $DEFAULT_MODEL_NAME)"
    echo ""
    echo "Examples:"
    echo "  $0 ../checkpoints/stage2_sft/checkpoint-120"
    echo "  $0 ../checkpoints/stage2_sft/checkpoint-120 8080"
    echo "  $0 ../checkpoints/stage2_sft/checkpoint-120 8080 my-dsl-model"
    echo ""
    echo "For PEFT/LoRA checkpoints, use --lora-modules flag:"
    echo "  $0 --lora ../checkpoints/stage2_sft/checkpoint-120"
}

# ============== 参数检查 ==============

if [ -z "$CHECKPOINT_PATH" ] || [ "$CHECKPOINT_PATH" == "-h" ] || [ "$CHECKPOINT_PATH" == "--help" ]; then
    show_help
    exit 0
fi

# 检查是否是 LoRA checkpoint
IS_LORA=false
if [ "$CHECKPOINT_PATH" == "--lora" ]; then
    IS_LORA=true
    CHECKPOINT_PATH="${2:-}"
    PORT="${3:-$DEFAULT_PORT}"
    MODEL_NAME="${4:-$DEFAULT_MODEL_NAME}"
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path not found: $CHECKPOINT_PATH"
    exit 1
fi

# 检测是否是 PEFT checkpoint
if [ -f "$CHECKPOINT_PATH/adapter_config.json" ]; then
    IS_LORA=true
    echo "Detected PEFT/LoRA checkpoint"
fi

# ============== 启动服务 ==============

echo "=============================================="
echo "Signal DSL Model Server (vLLM)"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Port: $PORT"
echo "Model Name: $MODEL_NAME"
echo "Is LoRA: $IS_LORA"
echo "=============================================="

if [ "$IS_LORA" = true ]; then
    # LoRA 模式：加载基座模型 + LoRA adapter
    echo "Starting vLLM with LoRA adapter..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$DEFAULT_BASE_MODEL" \
        --enable-lora \
        --lora-modules "$MODEL_NAME=$CHECKPOINT_PATH" \
        --served-model-name "$MODEL_NAME" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-model-len 2048
else
    # 完整模型模式
    echo "Starting vLLM with full model..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$CHECKPOINT_PATH" \
        --served-model-name "$MODEL_NAME" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-model-len 2048
fi
