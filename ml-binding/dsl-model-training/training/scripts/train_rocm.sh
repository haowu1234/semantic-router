#!/bin/bash
# ROCm Training Script for AMD GPUs
# Usage: ./train_rocm.sh [stage]
#   stage: 1, 2, 3, or "all" (default: all)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TRAINING_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================"
echo "DSL Model Training - ROCm (AMD GPU)"
echo "============================================================"

# Check ROCm/GPU availability
echo -e "\n${YELLOW}[1/4] Checking ROCm environment...${NC}"

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available (ROCm): {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'    Memory: {props.total_memory / 1024**3:.1f} GB')
else:
    print('WARNING: No GPU detected!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: ROCm/GPU not available!${NC}"
    echo "Please install PyTorch with ROCm support:"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0"
    exit 1
fi

echo -e "${GREEN}ROCm environment OK!${NC}"

# Set ROCm optimizations
echo -e "\n${YELLOW}[2/4] Setting ROCm optimizations...${NC}"

# Disable HIP memory pool to reduce fragmentation
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

# Use hipBLASLt for better GEMM performance (if available)
export TORCH_BLAS_PREFER_HIPBLASLT=1

# Set visible devices (use all by default)
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

# Disable NCCL if single GPU (reduces overhead)
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "$GPU_COUNT" -eq 1 ]; then
    export NCCL_P2P_DISABLE=1
fi

echo "  PYTORCH_HIP_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF"
echo "  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "  GPU_COUNT=$GPU_COUNT"

# Check data
echo -e "\n${YELLOW}[3/4] Checking training data...${NC}"

DATA_DIR="./prepared_data"
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}ERROR: Training data not found at $DATA_DIR${NC}"
    echo "Please run: python scripts/prepare_data.py --data-dir ../data --output-dir ./prepared_data"
    exit 1
fi

for stage in 1 2 3; do
    case $stage in
        1) file="stage1_syntax_pt.jsonl" ;;
        2) file="stage2_sft.jsonl" ;;
        3) file="stage3_dpo.jsonl" ;;
    esac
    if [ -f "$DATA_DIR/$file" ]; then
        count=$(wc -l < "$DATA_DIR/$file")
        echo -e "  Stage $stage: ${GREEN}$count samples${NC}"
    else
        echo -e "  Stage $stage: ${RED}MISSING${NC}"
    fi
done

echo -e "${GREEN}Training data OK!${NC}"

# Start training
echo -e "\n${YELLOW}[4/4] Starting training...${NC}"

STAGE=${1:-all}
STAGES_ARG=""

case $STAGE in
    1) STAGES_ARG="--stages 1" ;;
    2) STAGES_ARG="--stages 2" ;;
    3) STAGES_ARG="--stages 3" ;;
    all|"") STAGES_ARG="--stages 1,2,3" ;;
    *)
        echo -e "${RED}Unknown stage: $STAGE${NC}"
        echo "Usage: $0 [1|2|3|all]"
        exit 1
        ;;
esac

echo "Running stages: $STAGE"
echo ""

# Run training with accelerate for better memory management
python3 scripts/train_all.py \
    --config configs/base.yaml \
    --data-dir ./prepared_data \
    --output-dir ./checkpoints \
    $STAGES_ARG

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "Checkpoints saved to: ./checkpoints/"
