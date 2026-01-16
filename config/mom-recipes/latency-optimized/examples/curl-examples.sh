#!/bin/bash
# Latency-Optimized Recipe - Usage Examples

ENDPOINT="${VLLM_SR_ENDPOINT:-http://localhost:8000}"

echo "=== Latency-Optimized Recipe Examples ==="

# Fast query
echo "=== Fast Query ==="
time curl -s "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }' | jq '.choices[0].message.content'
