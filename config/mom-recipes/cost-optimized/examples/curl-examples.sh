#!/bin/bash
# Cost-Optimized Recipe - Usage Examples

ENDPOINT="${VLLM_SR_ENDPOINT:-http://localhost:8000}"

echo "=== Cost-Optimized Recipe Examples ==="

# Simple query (small model + cache)
echo "=== Simple Query (will be cached) ==="
curl -s "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }' | jq '.choices[0].message.content'

# Repeat query (should hit cache)
echo "=== Repeat Query (cache hit) ==="
curl -s "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }' | jq '.choices[0].message.content'
