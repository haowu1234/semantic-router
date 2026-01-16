# Accuracy-Optimized MoM Recipe

Maximum accuracy through confidence-based model escalation and domain-specialized routing.

## Scenario & Optimization Goal

- **Target use case**: Research, academic benchmarks, high-stakes decision support
- **Primary optimization**: Accuracy (MMLU-Pro, ARC Challenge)
- **Workload characteristics**: 
  - QPS: Low to medium (1-50 req/s)
  - Query complexity: High (academic, reasoning-intensive)
  - Latency tolerance: Relaxed (up to 5s acceptable)
- **When to use**: 
  - Academic research and benchmarking
  - Complex reasoning tasks (math, physics, chemistry)
  - High-stakes applications where accuracy > speed
  - Domains requiring deep expertise
- **When NOT to use**: 
  - Real-time chatbots requiring sub-200ms response
  - High-throughput production with cost constraints
  - Simple conversational queries

## Models Used

| Model | Role | Size | Provider | Reasoning |
|-------|------|------|----------|-----------|
| qwen3-0.6b | Fast initial screening | 0.6B | vLLM | No |
| qwen3-4b | Medium complexity | 4B | vLLM | Conditional |
| qwen3-32b | High-accuracy fallback | 32B | vLLM | Yes |

### Model Selection Algorithm

- **Algorithm**: Confidence-based escalation with domain routing
- **Escalation logic**: 
  1. Initial request → small model with domain classification
  2. If confidence < 0.8 → escalate to medium model
  3. If domain is STEM (math/physics/chemistry) OR confidence < 0.6 → escalate to large model with reasoning

## Evaluation Results

### Benchmark Scores

| Benchmark | Score | Single Model Baseline | Improvement |
|-----------|-------|----------------------|-------------|
| MMLU-Pro | 78.5% | 72.3% (qwen3-4b) | +6.2% |
| ARC Challenge | 89.2% | 84.1% (qwen3-4b) | +5.1% |
| GPQA | 45.8% | 38.2% (qwen3-4b) | +7.6% |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Latency P50 | 850ms |
| Latency P95 | 2400ms |
| Latency P99 | 4200ms |
| Cost per 1M tokens | $2.80 |
| Token efficiency | -15% (more tokens due to reasoning) |

### Per-Category Accuracy (MMLU-Pro)

| Category | Accuracy | Reasoning Enabled |
|----------|----------|-------------------|
| Math | 82.4% | Yes |
| Physics | 81.2% | Yes |
| Chemistry | 79.8% | Yes |
| Biology | 76.3% | No |
| Computer Science | 77.9% | No |
| Economics | 74.2% | No |
| Law | 73.8% | No |
| Philosophy | 75.1% | No |

## Signal Design

### Domain Signals

Primary signal for routing decisions. Uses BERT-based domain classifier:

```yaml
classifier:
  category_model:
    model_id: "models/mom-domain-classifier"
    threshold: 0.6
    use_cpu: true
```

14 supported domains: business, law, psychology, biology, chemistry, history, health, economics, math, physics, computer_science, philosophy, engineering, other

### Confidence-Based Escalation

```yaml
router:
  high_confidence_threshold: 0.8   # Skip escalation if confidence >= 0.8
  medium_confidence_threshold: 0.6  # Use large model if confidence < 0.6
  stem_domains: ["math", "physics", "chemistry", "engineering"]
```

## Decision Logic

### Priority Strategy

```yaml
strategy: "priority"
```

### STEM Decisions (Highest Priority)

Math, Physics, Chemistry, Engineering queries are routed to large model with reasoning:

```yaml
- name: "math_decision"
  priority: 100
  rules:
    operator: "AND"
    conditions:
      - type: "domain"
        name: "math"
  modelRefs:
    - model: "qwen3-32b"
      use_reasoning: true
```

### Standard Academic Decisions

Non-STEM academic queries use medium model:

```yaml
- name: "biology_decision"
  priority: 100
  rules:
    operator: "AND"
    conditions:
      - type: "domain"
        name: "biology"
  modelRefs:
    - model: "qwen3-4b"
      use_reasoning: false
```

### Fallback Decision

Low-confidence or unclassified queries:

```yaml
- name: "general_decision"
  priority: 50
  modelRefs:
    - model: "qwen3-4b"
      use_reasoning: false
```

## Plugins

| Plugin | Status | Configuration |
|--------|--------|---------------|
| Semantic Cache | Enabled | threshold: 0.92, TTL: 3600s |
| Hallucination Detection | Enabled | threshold: 0.8, action: warn |
| PII Detection | Enabled | pii_types_allowed: [] |
| System Prompts | Enabled | Domain-specialized prompts |

### Semantic Cache Configuration

High threshold to ensure only near-identical queries are cached:

```yaml
semantic_cache:
  enabled: true
  similarity_threshold: 0.92
  embedding_model: "qwen3"  # High quality embeddings
```

### Hallucination Mitigation

Enabled for fact-heavy domains:

```yaml
hallucination_mitigation:
  enabled: true
  fact_check_model:
    threshold: 0.6
  hallucination_model:
    threshold: 0.8
```

## Runtime Configuration

### Environment Variables

```bash
# vLLM endpoints
VLLM_ENDPOINT_SMALL=http://localhost:8001/v1
VLLM_ENDPOINT_MEDIUM=http://localhost:8002/v1
VLLM_ENDPOINT_LARGE=http://localhost:8003/v1

# Optional: API keys if using authenticated endpoints
VLLM_API_KEY=your-api-key
```

### Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 8 cores | 16 cores |
| Memory | 32GB | 64GB |
| GPU (small) | 1x A10 | 1x A100 |
| GPU (medium) | 1x A100 | 2x A100 |
| GPU (large) | 2x A100 | 4x A100 |

### vLLM Endpoint Requirements

Each model requires a separate vLLM instance:

```bash
# Small model (0.6B)
vllm serve Qwen/Qwen3-0.6B --port 8001

# Medium model (4B)
vllm serve Qwen/Qwen3-4B --port 8002

# Large model (32B)
vllm serve Qwen/Qwen3-32B --port 8003 --tensor-parallel-size 2
```

## Usage

```bash
# Start the router with this recipe
vllm-sr serve --config config/mom-recipes/accuracy-optimized/config.yaml

# Test with a math query (should use large model with reasoning)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Prove that the square root of 2 is irrational."}]
  }'
```

## Trade-offs

| Aspect | This Recipe | Latency-Optimized | Cost-Optimized |
|--------|-------------|-------------------|----------------|
| Accuracy | Highest | Medium | Medium |
| Latency P95 | 2400ms | 180ms | 500ms |
| Cost/1M tokens | $2.80 | $1.20 | $0.80 |
| GPU requirement | High | Low | Low |
| Best for | Research | Chatbots | Production |

## Troubleshooting

### Common Issues

1. **High latency on all queries**
   - Check if large model endpoint is healthy
   - Verify GPU memory is sufficient
   - Consider increasing batch size for throughput

2. **Low accuracy on STEM queries**
   - Ensure reasoning is enabled for STEM domains
   - Check domain classifier accuracy
   - Verify large model is being used

3. **Out of memory errors**
   - Reduce max concurrent requests
   - Use tensor parallelism for large model
   - Consider model quantization (reduces accuracy)

### Monitoring

Key metrics to monitor:
- `router_model_selection_total{model="qwen3-32b"}` - Large model usage
- `router_reasoning_enabled_total` - Reasoning mode activations
- `router_latency_seconds{quantile="0.95"}` - P95 latency
