#!/bin/bash
#
# 使用本地 vLLM 服务器生成 NL-DSL 训练数据
#
# 使用方法:
#   1. 在远程服务器上启动 vLLM:
#      python -m vllm.entrypoints.openai.api_server \
#          --model Qwen/Qwen2.5-72B-Instruct \
#          --tensor-parallel-size 4 \
#          --port 8000
#
#   2. 运行此脚本:
#      ./generate_with_vllm.sh --url http://your-server:8000
#
#   3. 或者使用 SSH 隧道:
#      ssh -L 8000:localhost:8000 your-server
#      ./generate_with_vllm.sh
#

set -e

# 默认配置
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL=""  # 留空自动检测
DATA_DIR="$(dirname "$0")/.."
INPUT_DIR="${DATA_DIR}/synthetic"
OUTPUT_DIR="${DATA_DIR}/nl_pairs"
LIMIT=""
TEMPERATURE=0.8
BATCH_SIZE=100
RESUME=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --url)
            VLLM_URL="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --url URL         vLLM server URL (default: http://localhost:8000)"
            echo "  --model MODEL     Model name (auto-detected if not specified)"
            echo "  --input DIR       Input directory with DSL samples"
            echo "  --output DIR      Output directory for NL-DSL pairs"
            echo "  --limit N         Limit number of samples to process"
            echo "  --temperature T   Sampling temperature (default: 0.8)"
            echo "  --resume          Resume from existing output files"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 检查 vLLM 服务器是否可用
echo "Checking vLLM server at ${VLLM_URL}..."
if ! curl -s "${VLLM_URL}/v1/models" > /dev/null 2>&1; then
    echo "Error: Cannot connect to vLLM server at ${VLLM_URL}"
    echo ""
    echo "Please start vLLM server first:"
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "      --model Qwen/Qwen2.5-72B-Instruct \\"
    echo "      --tensor-parallel-size 4 \\"
    echo "      --port 8000"
    exit 1
fi

# 获取模型信息
if [ -z "$MODEL" ]; then
    MODEL=$(curl -s "${VLLM_URL}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')")
fi
echo "Using model: ${MODEL}"

# 检查输入目录
if [ ! -d "$INPUT_DIR" ] && [ ! -f "$INPUT_DIR" ]; then
    echo "Error: Input directory/file not found: ${INPUT_DIR}"
    echo ""
    echo "Please generate synthetic DSL samples first:"
    echo "  cd ${DATA_DIR}"
    echo "  make synthetic"
    exit 1
fi

# 构建命令
CMD="python3 $(dirname "$0")/nl_generator.py"
CMD="$CMD --api vllm"
CMD="$CMD --vllm-url ${VLLM_URL}"
CMD="$CMD --model ${MODEL}"
CMD="$CMD --input ${INPUT_DIR}"
CMD="$CMD --output ${OUTPUT_DIR}"
CMD="$CMD --temperature ${TEMPERATURE}"
CMD="$CMD --batch-size ${BATCH_SIZE}"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit ${LIMIT}"
fi

if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
fi

# 显示配置
echo ""
echo "=========================================="
echo "NL Generation Configuration"
echo "=========================================="
echo "vLLM URL:    ${VLLM_URL}"
echo "Model:       ${MODEL}"
echo "Input:       ${INPUT_DIR}"
echo "Output:      ${OUTPUT_DIR}"
echo "Temperature: ${TEMPERATURE}"
echo "Resume:      ${RESUME}"
[ -n "$LIMIT" ] && echo "Limit:       ${LIMIT}"
echo "=========================================="
echo ""

# 运行
echo "Starting NL generation..."
$CMD

echo ""
echo "Done! Output saved to: ${OUTPUT_DIR}"
