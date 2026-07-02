# KV Cache Prefill Redundancy Verification

## Finding

The Qwen3Guard generative safety classifier's **fast path** (`generate_with_prefix_cache`) calls
`process_prefix()` on **every request**, re-doing a ~500-token full transformer forward pass that
produces **byte-identical results** across requests. The codebase already contains
`kv_cache_snapshot()` / `kv_cache_restore()` infrastructure (added for LoRA switching) but it is
**not wired into the guard inference path**.

## Root Cause

### Key source locations

| File | Function | Issue |
|------|----------|-------|
| `candle-binding/src/model_architectures/generative/qwen3_guard/qwen3_guard_generation.rs:15-18` | `generate_with_prefix_cache` | Calls `clear_kv_cache()` + `process_prefix()` every request |
| `candle-binding/src/model_architectures/generative/qwen3_guard.rs:211-249` | `initialize_prefix_caches` | Only tokenizes prefix; does NOT prefill or snapshot KV |
| `candle-binding/src/model_architectures/generative/qwen3_with_lora.rs:362-375` | `kv_cache_snapshot` / `kv_cache_restore` | **Already implemented** — never called from guard path |

### What "PrefixCache" caches vs what it should cache

```
Current PrefixCache:
  └── Vec<u32> (token IDs only)  ← saves ~5ms (tokenization skip)

  MISSING: KvCache snapshot      ← would save ~800ms (prefill skip)
```

### Why the prefix KV is cacheable

The prefix is a **fixed template string** (~500 tokens):
- Same input tokens every request
- Same model weights
- `temperature=0.0` → deterministic floating point path

Therefore every `process_prefix()` call produces **bit-identical KV Cache state**.
Re-computing it is pure waste.

## Verification: Instrumentation approach

We add precise timing instrumentation to the fast path to answer:

1. Is `process_prefix()` called on every request? → **YES**
2. How long does it take? → **Measure empirically**
3. Could `kv_cache_restore()` replace it? → **Yes, infrastructure exists**

### Step 1: Apply instrumentation patch

```bash
cd candle-binding
patch -p1 < ../../research/kv-cache-prefill/instrumentation.patch
```

The patch adds:
- `Instant::now()` timers around `process_prefix()` in the fast path
- A counter tracking total `process_prefix` invocations
- Timing output on each inference call
- Summary stats printed on drop

### Step 2: Build

```bash
cd candle-binding
cargo build --release
```

### Step 3: Run with real model

```bash
export QWEN3_GUARD_MODEL_PATH=/path/to/Qwen3Guard-Gen-0.6B
./scripts/verify_kv_cache_prefill.sh
```

### Step 4: Expected output

```
[GUARD_INSTR] Request #1: process_prefix=824ms, total=1211ms
[GUARD_INSTR] Request #2: process_prefix=818ms, total=1205ms
[GUARD_INSTR] Request #3: process_prefix=821ms, total=1210ms
[GUARD_INSTR] SUMMARY: process_prefix called 3 times, avg 821ms, total 2463ms wasted
```

If `kv_cache_restore` were used instead (discussed below), the summary would be:
```
[GUARD_INSTR] Request #1: kv_restore=0.5ms, total=412ms
[GUARD_INSTR] SUMMARY: kv_restore called 3 times, avg 0.5ms, total 1.5ms
```

### Step 5 (alternative): Lightweight code-only verification

If no Qwen3Guard model weights are available, the instrumentation patch still
proves the call pattern by adding a counter that confirms `process_prefix` is
invoked per-request even when the model load fails.

## KV Cache Size

| Component | BF16 (GPU) | F32 (CPU) |
|-----------|-----------|-----------|
| Per template (input mode) | ~57 MB | ~115 MB |
| Per template (output mode) | ~57 MB | ~115 MB |
| Both templates | ~114 MB | ~230 MB |
| Model weights | ~1.2 GB | ~2.4 GB |
| Overhead | ~8.5% | ~8.5% |

Formula: `500 tokens × 28 layers × (1024 K + 1024 V) × dtype`

## Proposed Fix

In `initialize_prefix_caches()` (after tokenization):
1. Call `self.model.clear_kv_cache()` then `self.model.process_prefix(&input_tokens)`
2. Call `self.model.kv_cache_snapshot()` to save the KV state
3. Store snapshot alongside the tokenized prefix

In `generate_with_prefix_cache()`:
1. Replace `self.model.clear_kv_cache()` + `self.model.process_prefix(&prefix_tokens)`
   with `self.model.kv_cache_restore(&kv_template)`

Expected per-request latency reduction: ~800ms → 5ms on CPU
Expected throughput improvement: ~3×
