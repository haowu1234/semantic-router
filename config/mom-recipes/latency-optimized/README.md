# Latency-Optimized MoM Recipe

Sub-200ms P95 latency through fast model selection and domain-based routing.

## Scenario & Optimization Goal

- **Target use case**: Real-time chatbots, customer support, interactive applications
- **Primary optimization**: Latency (sub-200ms P95)
- **Workload characteristics**: 
  - QPS: High (100-1000 req/s)
  - Query complexity: Low to medium
  - Latency tolerance: Strict (<200ms P95)
- **When to use**: 
  - Real-time chat applications
  - Customer service chatbots
  - Voice assistants with latency requirements
  - Interactive UIs requiring instant responses
- **When NOT to use**: 
  - Complex reasoning tasks (math, physics)
  - Research requiring maximum accuracy
  - Batch processing workloads

## Models Used

| Model | Role | Size | Provider | Reasoning |
|-------|------|------|----------|-----------|
| qwen3-0.6b | Primary (all queries) | 0.6B | vLLM | No |
| qwen3-4b | Fallback (complex) | 4B | vLLM | No |

### Model Selection Algorithm

- **Algorithm**: Domain routing with fast model priority
- **Escalation logic**: 
  1. All queries → small model (0.6B)
  2. Fallback to medium model only on error or timeout

## Evaluation Results

### Benchmark Scores

| Benchmark | Score | Accuracy-Optimized | Trade-off |
|-----------|-------|-------------------|-----------|
| MMLU-Pro | 68.2% | 78.5% | -10.3% |
| ARC Challenge | 78.4% | 89.2% | -10.8% |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Latency P50 | 45ms |
| Latency P95 | 180ms |
| Latency P99 | 320ms |
| Cost per 1M tokens | $0.40 |
| Token efficiency | +20% (shorter responses) |

## Signal Design

### Domain Signals

Lightweight domain classification for routing:

```yaml
classifier:
  category_model:
    model_id: "models/mom-domain-classifier"
    threshold: 0.5  # Lower threshold for speed
    use_cpu: true
```

## Decision Logic

### Priority Strategy

All decisions use the same small model for consistent low latency:

```yaml
strategy: "priority"

decisions:
  - name: "any_decision"
    priority: 100
    modelRefs:
      - model: "qwen3-small"
        use_reasoning: false  # Never enable reasoning
```

## Plugins

| Plugin | Status | Configuration |
|--------|--------|---------------|
| Semantic Cache | Enabled | threshold: 0.75, aggressive caching |
| Hallucination Detection | Disabled | - |
| PII Detection | Disabled | - |
| System Prompts | Enabled | Short, concise prompts |

## Runtime Configuration

### Environment Variables

```bash
VLLM_ENDPOINT=http://localhost:8001/v1
```

### Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| Memory | 8GB | 16GB |
| GPU | 1x T4 | 1x A10 |

## Usage

```bash
vllm-sr serve --config config/mom-recipes/latency-optimized/config.yaml
```

## Trade-offs

| Aspect | This Recipe | Accuracy-Optimized |
|--------|-------------|-------------------|
| Accuracy | 68.2% | 78.5% |
| Latency P95 | 180ms | 2400ms |
| Cost/1M tokens | $0.40 | $2.80 |
| GPU requirement | Low | High |
