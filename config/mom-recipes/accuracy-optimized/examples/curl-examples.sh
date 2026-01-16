#!/bin/bash
# Accuracy-Optimized Recipe - Usage Examples
# These examples demonstrate how to use the accuracy-optimized MoM recipe

ENDPOINT="${VLLM_SR_ENDPOINT:-http://localhost:8000}"

echo "=== Accuracy-Optimized MoM Recipe Examples ==="
echo "Endpoint: $ENDPOINT"
echo ""

# Example 1: Math query (uses large model with reasoning)
echo "=== Example 1: Math Query (Large Model + Reasoning) ==="
curl -s "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "Prove that the square root of 2 is irrational using proof by contradiction."
      }
    ],
    "temperature": 0.1
  }' | jq '.choices[0].message.content' | head -c 500
echo ""
echo ""

# Example 2: Physics query (uses large model with reasoning)
echo "=== Example 2: Physics Query (Large Model + Reasoning) ==="
curl -s "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "Derive the time dilation formula from special relativity using Lorentz transformations."
      }
    ],
    "temperature": 0.1
  }' | jq '.choices[0].message.content' | head -c 500
echo ""
echo ""

# Example 3: Biology query (uses medium model without reasoning)
echo "=== Example 3: Biology Query (Medium Model) ==="
curl -s "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "Explain the process of DNA replication and the role of different enzymes involved."
      }
    ],
    "temperature": 0.3
  }' | jq '.choices[0].message.content' | head -c 500
echo ""
echo ""

# Example 4: Chemistry query (uses large model with reasoning)
echo "=== Example 4: Chemistry Query (Large Model + Reasoning) ==="
curl -s "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "Balance the following redox reaction in acidic solution: MnO4- + Fe2+ → Mn2+ + Fe3+"
      }
    ],
    "temperature": 0.1
  }' | jq '.choices[0].message.content' | head -c 500
echo ""
echo ""

# Example 5: General query (uses medium model)
echo "=== Example 5: General Query (Medium Model) ==="
curl -s "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "What are the main differences between TCP and UDP protocols?"
      }
    ],
    "temperature": 0.3
  }' | jq '.choices[0].message.content' | head -c 500
echo ""
echo ""

# Example 6: Streaming response
echo "=== Example 6: Streaming Response ==="
curl -s "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "Explain the concept of entropy in thermodynamics."
      }
    ],
    "stream": true,
    "temperature": 0.2
  }' | head -20
echo ""
echo ""

# Example 7: Check router metrics
echo "=== Example 7: Router Metrics ==="
curl -s "$ENDPOINT/metrics" | grep -E "^router_" | head -20
echo ""

# Example 8: Health check
echo "=== Example 8: Health Check ==="
curl -s "$ENDPOINT/health" | jq '.'
echo ""

echo "=== Examples Complete ==="
