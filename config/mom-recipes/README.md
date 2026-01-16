# Mixture-of-Models (MoM) Recipes

Pre-configured, benchmarked, and production-ready multi-model routing strategies for different optimization goals.

## Quick Start

```bash
# Use a recipe directly
vllm-sr serve --config config/mom-recipes/accuracy-optimized/config.yaml

# Or copy and customize
cp config/mom-recipes/accuracy-optimized/config.yaml my-config.yaml
vllm-sr serve --config my-config.yaml
```

## Available Recipes

| Recipe | Optimization Goal | Key Features | Best For |
|--------|------------------|--------------|----------|
| [accuracy-optimized](./accuracy-optimized/) | Maximum accuracy | Confidence-based escalation, multi-model verification | Research, high-stakes applications |
| [latency-optimized](./latency-optimized/) | Sub-200ms P95 | Fast model selection, domain routing | Real-time applications, chatbots |
| [cost-optimized](./cost-optimized/) | 60%+ token reduction | Size-aware routing, aggressive caching | High-volume production workloads |

## Recipe Structure

Each recipe follows this structure:

```
{recipe-name}/
├── README.md           # Recipe documentation
├── config.yaml         # Complete router configuration
├── evaluation/         # Benchmark results
│   ├── mmlu-pro.json   # MMLU-Pro evaluation results
│   ├── arc-challenge.json  # ARC Challenge results
│   └── summary.md      # Performance summary
└── examples/
    └── curl-examples.sh  # Usage examples
```

## Creating New Recipes

See [TEMPLATE.md](./TEMPLATE.md) for guidelines on creating new recipes.

## Evaluation

All recipes are validated against standard benchmarks:

- **MMLU-Pro**: Academic knowledge across 14 categories
- **ARC Challenge**: Science reasoning (grade 3-9)
- **GPQA**: Graduate-level science questions (optional)
- **TruthfulQA**: Truthfulness evaluation (optional)

Run evaluation:

```bash
# Evaluate a recipe against MMLU-Pro
python src/training/model_eval/mmlu_pro_vllm_eval.py \
  --endpoint http://127.0.0.1:8000/v1 \
  --models qwen3 \
  --samples-per-category 50 \
  --use-cot

# Evaluate against ARC Challenge
python src/training/model_eval/arc_challenge_vllm_eval.py \
  --endpoint http://127.0.0.1:8000/v1 \
  --models qwen3 \
  --samples 100
```

## Contributing

We welcome community contributions! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

When submitting a new recipe:

1. Follow the [TEMPLATE.md](./TEMPLATE.md) structure
2. Include evaluation results on at least MMLU-Pro and ARC Challenge
3. Document all configuration options and trade-offs
4. Provide usage examples
