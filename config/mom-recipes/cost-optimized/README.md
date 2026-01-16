# Cost-Optimized MoM Recipe

60%+ token reduction through size-aware routing and aggressive semantic caching.

## Scenario & Optimization Goal

- **Target use case**: High-volume production workloads, cost-sensitive deployments
- **Primary optimization**: Cost (60%+ token reduction)
- **Workload characteristics**: 
  - QPS: Very high (500-5000 req/s)
  - Query complexity: Mixed
  - Latency tolerance: Moderate (500ms acceptable)
- **When to use**: 
  - High-volume production workloads
  - Cost-sensitive deployments
  - Repetitive query patterns
  - Internal tools and APIs
- **When NOT to use**: 
  - Unique queries with no repetition
  - Applications requiring maximum accuracy
  - Real-time applications

## Models Used

| Model | Role | Size | Provider |
|-------|------|------|----------|
| qwen3-0.6b | Primary (simple) | 0.6B | vLLM |
| qwen3-4b | Secondary (complex) | 4B | vLLM |

### Model Selection Algorithm

- **Algorithm**: Size-aware routing with semantic cache priority
- **Logic**: 
  1. Check semantic cache first (75% threshold)
  2. Simple queries → small model
  3. Complex queries → medium model

## Evaluation Results

### Benchmark Scores

| Benchmark | Score | Trade-off |
|-----------|-------|-----------|
| MMLU-Pro | 71.4% | -7.1% vs accuracy |
| ARC Challenge | 82.1% | -7.1% vs accuracy |

### Cost Metrics

| Metric | Value |
|--------|-------|
| Cost per 1M tokens | $0.80 |
| Token reduction | 62% |
| Cache hit rate | 45% |
| Latency P95 | 450ms |

## Signal Design

### Semantic Cache (Primary)

Aggressive caching with lower threshold:

```yaml
semantic_cache:
  enabled: true
  similarity_threshold: 0.75
  max_entries: 50000
  ttl_seconds: 7200
```

### Query Complexity Detection

Route based on query length and domain:

```yaml
# Short queries (<50 tokens) → small model
# Long queries (>50 tokens) → medium model
```

## Plugins

| Plugin | Status | Configuration |
|--------|--------|---------------|
| Semantic Cache | Enabled | threshold: 0.75, 50K entries |
| Hallucination Detection | Disabled | - |
| PII Detection | Disabled | - |
| Response Compression | Enabled | max_tokens: 256 |

## Runtime Configuration

### Resource Requirements

| Resource | Minimum |
|----------|---------|
| CPU | 4 cores |
| Memory | 8GB |
| GPU | 1x T4 |

## Usage

```bash
vllm-sr serve --config config/mom-recipes/cost-optimized/config.yaml
```

## Trade-offs

| Aspect | This Recipe | Accuracy-Optimized |
|--------|-------------|-------------------|
| Accuracy | 71.4% | 78.5% |
| Cost/1M tokens | $0.80 | $2.80 |
| Cache hit rate | 45% | 15% |
| Best for | High volume | Research |
