.PHONY: all install prepare train-all train-stage1 train-stage2 train-stage3 evaluate clean

# Configuration
DATA_DIR ?= ../data
OUTPUT_DIR ?= ./checkpoints
PREPARED_DATA_DIR ?= $(DATA_DIR)/final
CONFIG ?= configs/base.yaml
MODEL ?= Qwen/Qwen2.5-3B
STAGES ?= 1,2,3

# Install dependencies
install:
	pip install -r requirements.txt

# Prepare training data
prepare:
	@echo "📦 Building training data via data pipeline..."
	$(MAKE) -C $(DATA_DIR) validate

# Train all stages
train-all: prepare
	@echo "🚀 Starting full training pipeline..."
	python scripts/train_all.py \
		--config $(CONFIG) \
		--data-dir $(PREPARED_DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--stages $(STAGES)

# Train individual stages
train-stage1:
	@echo "🔤 Stage 1: Syntax Pre-training..."
	python scripts/train_all.py \
		--config $(CONFIG) \
		--data-dir $(PREPARED_DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--stages 1 \
		--skip-eval

train-stage2:
	@echo "📝 Stage 2: SFT Training..."
	python scripts/train_all.py \
		--config $(CONFIG) \
		--data-dir $(PREPARED_DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--stages 2 \
		--stage1-checkpoint $(OUTPUT_DIR)/stage1_pt \
		--skip-eval

train-stage3:
	@echo "🎯 Stage 3: DPO Training..."
	python scripts/train_all.py \
		--config $(CONFIG) \
		--data-dir $(PREPARED_DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--stages 3 \
		--stage2-checkpoint $(OUTPUT_DIR)/stage2_sft

# Evaluate
evaluate:
	@echo "📊 Running evaluation..."
	python scripts/evaluate.py \
		--model $(OUTPUT_DIR)/stage3_dpo \
		--eval-data $(PREPARED_DATA_DIR)/eval_benchmark.jsonl \
		--config $(CONFIG)

# Quick test (small data, few steps)
quick-test:
	@echo "🧪 Running quick test..."
	$(MAKE) -C $(DATA_DIR) validate-skip
	python scripts/train_all.py \
		--config configs/base.yaml \
		--data-dir $(PREPARED_DATA_DIR) \
		--output-dir ./test_checkpoints \
		--stages 1 \
		--skip-eval

# Clean
clean:
	rm -rf $(OUTPUT_DIR)
	rm -rf ./logs
	rm -rf ./test_checkpoints

# Help
help:
	@echo "DSL Model Training Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install        Install dependencies"
	@echo "  make prepare        Prepare training data"
	@echo "  make train-all      Run full training pipeline (all 3 stages)"
	@echo "  make train-stage1   Run Stage 1 (syntax pre-training)"
	@echo "  make train-stage2   Run Stage 2 (SFT)"
	@echo "  make train-stage3   Run Stage 3 (DPO)"
	@echo "  make evaluate       Evaluate trained model"
	@echo "  make quick-test     Quick test with small data"
	@echo "  make clean          Clean generated files"
	@echo ""
	@echo "Configuration variables:"
	@echo "  DATA_DIR=$(DATA_DIR)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  MODEL=$(MODEL)"
	@echo "  STAGES=$(STAGES)"
