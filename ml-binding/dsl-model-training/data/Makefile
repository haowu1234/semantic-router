# DSL Model Training - Data Engineering Makefile
#
# 用法:
#   make all          - 完整数据集构建流程
#   make seeds        - 仅提取种子数据
#   make synthetic    - 生成合成DSL
#   make nl-pairs     - 生成NL描述 (需要API key)
#   make nl-vllm      - 使用本地vLLM生成NL描述 (推荐)
#   make negative     - 生成负样本
#   make validate     - 验证并构建最终数据集
#   make stats        - 生成数据集统计报告
#   make clean        - 清理生成的数据

SHELL := /bin/bash
PYTHON := python3

# 项目路径
REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null || echo "../..")
DATA_DIR := $(shell pwd)
SCRIPTS_DIR := $(DATA_DIR)/scripts

# 输出目录
SEEDS_DIR := $(DATA_DIR)/seeds
SYNTHETIC_DIR := $(DATA_DIR)/synthetic
NL_PAIRS_DIR := $(DATA_DIR)/nl_pairs
NEGATIVE_DIR := $(DATA_DIR)/negative
FINAL_DIR := $(DATA_DIR)/final
STATS_DIR := $(DATA_DIR)/stats

# Go 工具 (如果存在)
PARSER_BIN := $(REPO_ROOT)/bin/dsl-parser
DECOMPILER_BIN := $(REPO_ROOT)/bin/dsl-decompiler

# 数据量配置
SYNTHETIC_COUNT := 10000
NL_LIMIT := 0  # 0 = 不限制
NEGATIVE_RATIO := 0.5

# API 配置 (NL 生成)
NL_API := local  # local, openai, anthropic, vllm
NL_MODEL := gpt-4o

# vLLM 配置
VLLM_URL := http://localhost:8000
VLLM_MODEL :=  # 留空自动检测

.PHONY: all seeds synthetic nl-pairs nl-vllm negative validate stats clean help

help:
	@echo "DSL Model Training - Data Engineering"
	@echo ""
	@echo "用法: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  all        - 完整数据集构建流程"
	@echo "  seeds      - 提取种子数据 (从测试文件/YAML)"
	@echo "  synthetic  - 生成合成DSL (CFG随机游走)"
	@echo "  nl-pairs   - 生成NL描述 (需要API key或使用local模式)"
	@echo "  nl-vllm    - 使用本地vLLM生成NL描述 (推荐)"
	@echo "  negative   - 生成负样本 (DPO训练用)"
	@echo "  validate   - 验证数据并构建最终数据集"
	@echo "  stats      - 生成数据集统计报告"
	@echo "  clean      - 清理生成的数据"
	@echo ""
	@echo "配置变量:"
	@echo "  SYNTHETIC_COUNT=$(SYNTHETIC_COUNT)  - 合成样本数量"
	@echo "  NL_API=$(NL_API)                    - NL生成API (local/openai/anthropic/vllm)"
	@echo "  NL_MODEL=$(NL_MODEL)                - 使用的LLM模型"
	@echo "  VLLM_URL=$(VLLM_URL)                - vLLM服务器地址"
	@echo ""
	@echo "示例:"
	@echo "  make synthetic SYNTHETIC_COUNT=5000"
	@echo "  make nl-pairs NL_API=openai"
	@echo "  make nl-vllm VLLM_URL=http://192.168.1.100:8000"
	@echo "  make all"

all: seeds synthetic nl-pairs negative validate stats
	@echo "✅ 数据集构建完成!"
	@echo ""
	@cat $(STATS_DIR)/dataset_report.json 2>/dev/null || cat $(FINAL_DIR)/dataset_stats.json

# ============== 种子数据提取 ==============
seeds: $(SEEDS_DIR)/from_tests.jsonl

$(SEEDS_DIR)/from_tests.jsonl:
	@echo "📥 提取种子数据..."
	@mkdir -p $(SEEDS_DIR)
	$(PYTHON) $(SCRIPTS_DIR)/extract_seeds.py \
		--repo-root $(REPO_ROOT) \
		--output $(SEEDS_DIR) \
		$(if $(wildcard $(DECOMPILER_BIN)),--decompiler $(DECOMPILER_BIN),)
	@echo "✅ 种子数据提取完成: $(SEEDS_DIR)"

# ============== 合成数据生成 ==============
synthetic: $(SYNTHETIC_DIR)/synthetic_all.jsonl

$(SYNTHETIC_DIR)/synthetic_all.jsonl: 
	@echo "🔧 生成合成DSL数据 ($(SYNTHETIC_COUNT) 样本)..."
	@mkdir -p $(SYNTHETIC_DIR)
	$(PYTHON) $(SCRIPTS_DIR)/cfg_generator.py \
		--count $(SYNTHETIC_COUNT) \
		--output $(SYNTHETIC_DIR) \
		--seed 42
	@echo "✅ 合成数据生成完成: $(SYNTHETIC_DIR)"

# ============== NL 描述生成 ==============
nl-pairs: $(NL_PAIRS_DIR)/en_formal.jsonl

$(NL_PAIRS_DIR)/en_formal.jsonl: $(SYNTHETIC_DIR)/synthetic_all.jsonl $(SEEDS_DIR)/from_tests.jsonl
	@echo "📝 生成NL描述 (API: $(NL_API), Model: $(NL_MODEL))..."
	@mkdir -p $(NL_PAIRS_DIR)
	# 处理合成数据
	$(PYTHON) $(SCRIPTS_DIR)/nl_generator.py \
		--input $(SYNTHETIC_DIR) \
		--output $(NL_PAIRS_DIR) \
		--api $(NL_API) \
		--model $(NL_MODEL) \
		$(if $(NL_LIMIT),--limit $(NL_LIMIT),)
	# 处理种子数据
	$(PYTHON) $(SCRIPTS_DIR)/nl_generator.py \
		--input $(SEEDS_DIR) \
		--output $(NL_PAIRS_DIR)/seeds \
		--api $(NL_API) \
		--model $(NL_MODEL)
	@echo "✅ NL描述生成完成: $(NL_PAIRS_DIR)"

nl-pairs-dry-run:
	@echo "📝 生成NL描述 (dry-run, 本地规则生成)..."
	@mkdir -p $(NL_PAIRS_DIR)
	$(PYTHON) $(SCRIPTS_DIR)/nl_generator.py \
		--input $(SYNTHETIC_DIR) \
		--output $(NL_PAIRS_DIR) \
		--dry-run
	@echo "✅ NL描述生成完成 (dry-run): $(NL_PAIRS_DIR)"

# ============== vLLM NL 描述生成 ==============
# 使用本地 vLLM 服务器生成 NL 描述 (推荐)
# 
# 启动 vLLM:
#   python -m vllm.entrypoints.openai.api_server \
#       --model Qwen/Qwen2.5-72B-Instruct \
#       --tensor-parallel-size 4 --port 8000
#
# 用法:
#   make nl-vllm VLLM_URL=http://localhost:8000
#   make nl-vllm VLLM_URL=http://192.168.1.100:8000 VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct
#
nl-vllm: $(SYNTHETIC_DIR)/synthetic_all.jsonl $(SEEDS_DIR)/from_tests.jsonl
	@echo "📝 使用 vLLM 生成NL描述..."
	@echo "   vLLM URL: $(VLLM_URL)"
	@if [ -n "$(VLLM_MODEL)" ]; then echo "   Model: $(VLLM_MODEL)"; else echo "   Model: (自动检测)"; fi
	@# 检查 vLLM 服务器是否可用
	@curl -s "$(VLLM_URL)/v1/models" > /dev/null 2>&1 || (echo "❌ 无法连接到 vLLM 服务器: $(VLLM_URL)" && exit 1)
	@mkdir -p $(NL_PAIRS_DIR)
	# 处理合成数据
	$(PYTHON) $(SCRIPTS_DIR)/nl_generator.py \
		--input $(SYNTHETIC_DIR) \
		--output $(NL_PAIRS_DIR) \
		--api vllm \
		--vllm-url $(VLLM_URL) \
		$(if $(VLLM_MODEL),--model $(VLLM_MODEL),) \
		$(if $(NL_LIMIT),--limit $(NL_LIMIT),) \
		--resume
	# 处理种子数据
	$(PYTHON) $(SCRIPTS_DIR)/nl_generator.py \
		--input $(SEEDS_DIR) \
		--output $(NL_PAIRS_DIR)/seeds \
		--api vllm \
		--vllm-url $(VLLM_URL) \
		$(if $(VLLM_MODEL),--model $(VLLM_MODEL),) \
		--resume
	@echo "✅ NL描述生成完成: $(NL_PAIRS_DIR)"

nl-vllm-resume:
	@echo "📝 使用 vLLM 恢复生成NL描述 (断点续传)..."
	$(PYTHON) $(SCRIPTS_DIR)/nl_generator.py \
		--input $(SYNTHETIC_DIR) \
		--output $(NL_PAIRS_DIR) \
		--api vllm \
		--vllm-url $(VLLM_URL) \
		$(if $(VLLM_MODEL),--model $(VLLM_MODEL),) \
		--resume
	@echo "✅ NL描述生成完成: $(NL_PAIRS_DIR)"

# ============== 负样本生成 ==============
negative: $(NEGATIVE_DIR)/syntax_error.jsonl

$(NEGATIVE_DIR)/syntax_error.jsonl: $(SYNTHETIC_DIR)/synthetic_all.jsonl
	@echo "❌ 生成负样本 (比例: $(NEGATIVE_RATIO))..."
	@mkdir -p $(NEGATIVE_DIR)
	$(PYTHON) $(SCRIPTS_DIR)/negative_sampler.py \
		--input $(SYNTHETIC_DIR) \
		--output $(NEGATIVE_DIR) \
		--ratio $(NEGATIVE_RATIO)
	@echo "✅ 负样本生成完成: $(NEGATIVE_DIR)"

# ============== 数据验证与最终数据集构建 ==============
validate: $(FINAL_DIR)/stage1_syntax_pt.jsonl

$(FINAL_DIR)/stage1_syntax_pt.jsonl: $(SYNTHETIC_DIR)/synthetic_all.jsonl $(NL_PAIRS_DIR)/en_formal.jsonl $(NEGATIVE_DIR)/syntax_error.jsonl
	@echo "🔍 验证数据并构建最终数据集..."
	@mkdir -p $(FINAL_DIR)
	$(PYTHON) $(SCRIPTS_DIR)/validator.py \
		--input $(DATA_DIR) \
		--output $(FINAL_DIR) \
		$(if $(wildcard $(PARSER_BIN)),--parser-bin $(PARSER_BIN),--skip-validation)
	@echo "✅ 最终数据集构建完成: $(FINAL_DIR)"

validate-skip:
	@echo "🔍 构建最终数据集 (跳过验证)..."
	@mkdir -p $(FINAL_DIR)
	$(PYTHON) $(SCRIPTS_DIR)/validator.py \
		--input $(DATA_DIR) \
		--output $(FINAL_DIR) \
		--skip-validation
	@echo "✅ 最终数据集构建完成: $(FINAL_DIR)"

# ============== 统计报告 ==============
stats: $(STATS_DIR)/dataset_report.json

$(STATS_DIR)/dataset_report.json: $(FINAL_DIR)/stage1_syntax_pt.jsonl
	@echo "📊 生成数据集统计报告..."
	@mkdir -p $(STATS_DIR)
	@$(PYTHON) -c '\
import json; \
from pathlib import Path; \
stats = {}; \
for f in Path("$(FINAL_DIR)").glob("*.jsonl"): \
    count = sum(1 for _ in open(f)); \
    stats[f.stem] = count; \
stats["total"] = sum(stats.values()); \
Path("$(STATS_DIR)/dataset_report.json").write_text(json.dumps(stats, indent=2)); \
print("Dataset sizes:"); \
[print(f"  {k}: {v}") for k,v in stats.items()]; \
'
	@echo "✅ 统计报告已保存: $(STATS_DIR)/dataset_report.json"

# ============== 清理 ==============
clean:
	@echo "🧹 清理生成的数据..."
	rm -rf $(SEEDS_DIR) $(SYNTHETIC_DIR) $(NL_PAIRS_DIR) $(NEGATIVE_DIR) $(FINAL_DIR) $(STATS_DIR)
	@echo "✅ 清理完成"

clean-synthetic:
	rm -rf $(SYNTHETIC_DIR)

clean-nl:
	rm -rf $(NL_PAIRS_DIR)

clean-negative:
	rm -rf $(NEGATIVE_DIR)

clean-final:
	rm -rf $(FINAL_DIR) $(STATS_DIR)

# ============== 开发辅助 ==============
# 快速测试 (小数据量)
quick-test:
	@echo "🚀 快速测试模式 (小数据量)..."
	$(MAKE) seeds
	$(MAKE) synthetic SYNTHETIC_COUNT=100
	$(MAKE) nl-pairs-dry-run
	$(MAKE) negative NEGATIVE_RATIO=0.3
	$(MAKE) validate-skip
	$(MAKE) stats
	@echo "✅ 快速测试完成!"

# 检查依赖
check-deps:
	@echo "检查Python依赖..."
	@$(PYTHON) -c "import json, re, hashlib, subprocess" && echo "  ✅ 标准库: OK"
	@$(PYTHON) -c "import requests" 2>/dev/null && echo "  ✅ requests: OK" || echo "  ⚠️  requests: 未安装 (vLLM需要: pip install requests)"
	@$(PYTHON) -c "from openai import OpenAI" 2>/dev/null && echo "  ✅ openai: OK" || echo "  ⚠️  openai: 未安装 (OpenAI API需要)"
	@$(PYTHON) -c "import anthropic" 2>/dev/null && echo "  ✅ anthropic: OK" || echo "  ⚠️  anthropic: 未安装 (可选)"
	@echo ""
	@echo "检查Go工具..."
	@test -f $(PARSER_BIN) && echo "  ✅ parser: $(PARSER_BIN)" || echo "  ⚠️  parser: 未找到 (将使用简单验证)"
	@test -f $(DECOMPILER_BIN) && echo "  ✅ decompiler: $(DECOMPILER_BIN)" || echo "  ⚠️  decompiler: 未找到 (YAML反编译不可用)"

# 检查 vLLM 服务器
check-vllm:
	@echo "检查 vLLM 服务器..."
	@curl -s "$(VLLM_URL)/v1/models" > /dev/null 2>&1 && \
		echo "✅ vLLM 服务器可用: $(VLLM_URL)" && \
		echo "   模型: $$(curl -s "$(VLLM_URL)/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')")" \
		|| echo "❌ 无法连接到 vLLM 服务器: $(VLLM_URL)"
