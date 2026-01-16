# Latency-Optimized Recipe - Evaluation Summary

## Overview

Optimized for sub-200ms P95 latency at the cost of accuracy.

## Performance Metrics

| Metric | Value |
|--------|-------|
| Latency P50 | 45ms |
| Latency P95 | 180ms |
| Latency P99 | 320ms |
| Throughput | 850 req/s |

## Benchmark Scores

| Benchmark | Score | vs Accuracy-Optimized |
|-----------|-------|-----------------------|
| MMLU-Pro | 68.2% | -10.3% |
| ARC Challenge | 78.4% | -10.8% |

## Trade-off Analysis

This recipe prioritizes latency over accuracy:
- **13x faster** P95 latency (180ms vs 2400ms)
- **7x cheaper** per 1M tokens ($0.40 vs $2.80)
- **10% lower** accuracy on benchmarks

## Recommended Use Cases

1. Real-time chatbots
2. Customer support automation
3. Voice assistants
4. Interactive applications
