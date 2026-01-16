# MoM Recipe Template

Use this template when creating a new MoM Recipe.

## Recipe Directory Structure

```
config/mom-recipes/{recipe-name}/
├── README.md           # Recipe documentation (required)
├── config.yaml         # Complete router configuration (required)
├── evaluation/         # Benchmark results (required)
│   ├── mmlu-pro.json   # MMLU-Pro evaluation results
│   ├── arc-challenge.json  # ARC Challenge results
│   └── summary.md      # Performance summary & analysis
└── examples/
    └── curl-examples.sh  # Usage examples
```

## README.md Template

```markdown
# {Recipe Name}

{One-line description of the recipe}

## Scenario & Optimization Goal

- **Target use case**: {research, production, real-time, etc.}
- **Primary optimization**: {accuracy, latency, cost, balanced}
- **Workload characteristics**: {QPS, query complexity, latency tolerance}
- **When to use**: {conditions when this recipe is recommended}
- **When NOT to use**: {conditions when this recipe is not suitable}

## Models Used

| Model | Role | Size | Provider |
|-------|------|------|----------|
| {model-name} | {fast initial / high-accuracy fallback / etc.} | {7B/14B/70B} | {vLLM/OpenAI/etc.} |

### Model Selection Algorithm

- **Algorithm**: {confidence / ratings / size-aware}
- **Escalation logic**: {describe when and how models are escalated}

## Evaluation Results

### Benchmark Scores

| Benchmark | Score | Baseline | Improvement |
|-----------|-------|----------|-------------|
| MMLU-Pro | {X}% | {Y}% | +{Z}% |
| ARC Challenge | {X}% | {Y}% | +{Z}% |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Latency P50 | {X}ms |
| Latency P95 | {X}ms |
| Latency P99 | {X}ms |
| Cost per 1M tokens | ${X} |
| Token efficiency | {X}% reduction |

## Signal Design

### Domain Signals
{Describe domain classification configuration}

### Keyword Signals (if applicable)
{Describe keyword patterns and use cases}

### Embedding Signals (if applicable)
{Describe semantic similarity thresholds}

## Decision Logic

### Decision Rules
{Describe AND/OR combinations and priorities}

### Algorithm Configuration
```yaml
# Example configuration snippet
strategy: "priority"
decisions:
  - name: "example_decision"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "example"
```

## Plugins

| Plugin | Status | Configuration |
|--------|--------|---------------|
| Semantic Cache | {enabled/disabled} | {threshold, TTL} |
| Hallucination Detection | {enabled/disabled} | {threshold, action} |
| PII Detection | {enabled/disabled} | {policy} |
| System Prompts | {enabled/disabled} | {per-decision} |

## Runtime Configuration

### Environment Variables

```bash
VLLM_ENDPOINT=http://localhost:8000/v1
VLLM_API_KEY=your-api-key
```

### Resource Requirements

- **CPU**: {minimum cores}
- **Memory**: {minimum RAM}
- **GPU**: {GPU requirements if any}

### vLLM Endpoint Requirements

{Describe required vLLM configuration}

## Usage

```bash
vllm-sr serve --config config/mom-recipes/{recipe-name}/config.yaml
```

## Trade-offs

| Aspect | This Recipe | Alternative |
|--------|-------------|-------------|
| {aspect} | {value} | {comparison} |

## Troubleshooting

### Common Issues

1. **{Issue}**: {Solution}
```

## config.yaml Requirements

The configuration file must include:

1. **vllm_endpoints**: At least one endpoint configuration
2. **model_config**: Configuration for each model used
3. **categories**: Domain categories for routing
4. **decisions**: Complete decision rules with modelRefs and plugins
5. **default_model**: Fallback model specification

## Evaluation Requirements

### Required Benchmarks

1. **MMLU-Pro** (`evaluation/mmlu-pro.json`)
   - Minimum 50 samples per category
   - Include per-category breakdown
   - Document CoT usage

2. **ARC Challenge** (`evaluation/arc-challenge.json`)
   - Minimum 100 samples
   - Include ARC-Easy and ARC-Challenge results

### Optional Benchmarks

- GPQA (graduate-level science)
- TruthfulQA (truthfulness)
- GSM8K (math reasoning)

### Evaluation JSON Format

```json
{
  "recipe_name": "{recipe-name}",
  "benchmark": "mmlu-pro",
  "timestamp": "2025-01-16T00:00:00Z",
  "configuration": {
    "samples_per_category": 50,
    "use_cot": true,
    "models": ["qwen3"]
  },
  "results": {
    "overall_accuracy": 0.85,
    "per_category": {
      "math": 0.82,
      "physics": 0.88
    }
  },
  "performance": {
    "total_time_seconds": 1234,
    "avg_latency_ms": 150,
    "p95_latency_ms": 280
  }
}
```

## Submission Checklist

- [ ] README.md with all required sections
- [ ] Complete config.yaml
- [ ] MMLU-Pro evaluation results
- [ ] ARC Challenge evaluation results
- [ ] evaluation/summary.md with analysis
- [ ] examples/curl-examples.sh
- [ ] Tested with `vllm-sr serve --config`
