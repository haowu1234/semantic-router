# DSL × AI Deep Integration: Extension Design

## 1. Overview

The Signal DSL compiler (§ `config-dsl-visual-builder.md`) establishes a solid foundation: a 5-construct grammar (SIGNAL / ROUTE / PLUGIN / BACKEND / GLOBAL), a Go→WASM in-browser compiler, and three interaction modes (Visual / DSL / NL) sharing a single AST.

This document extends that foundation with **six AI-powered enhancement directions** that tighten the coupling between the DSL and artificial intelligence — from natural language understanding to autonomous configuration optimization.

### Current State

| Component | Status | Key Files |
|:---|:---|:---|
| Lexer + Parser (participle) | ✅ Shipped | `pkg/dsl/parser.go` |
| Compiler (AST → RouterConfig) | ✅ Shipped | `pkg/dsl/compiler.go` |
| Decompiler (RouterConfig → DSL) | ✅ Shipped | `pkg/dsl/decompiler.go` |
| Validator (3-level diagnostics) | ✅ Shipped | `pkg/dsl/validator.go` |
| WASM Bridge (5 APIs) | ✅ Shipped | `cmd/wasm/main_wasm.go` |
| Visual Builder (ReactFlow + Forms) | ✅ Shipped | `pages/BuilderPage.tsx` |
| DSL Editor (Monaco + Language Support) | ✅ Shipped | `pages/DslEditorPage.tsx`, `lib/dslLanguage.ts` |
| NL Mode | 🔲 Placeholder | `BuilderPage.tsx:536` — disabled, "coming soon" |

### Design Principles

1. **AST remains the single source of truth** — AI generates or modifies DSL text/AST, never bypasses the compiler.
2. **Deterministic > Probabilistic** — Wherever possible, constrain LLM output with grammar rules, schemas, or whitelists.
3. **Human-in-the-loop** — AI proposes, human approves. Critical changes require explicit confirmation.
4. **Incremental adoption** — Each direction is independently deployable; no "big bang" dependency chain.

---

## 2. Direction 1: NL → DSL Intelligent Mapping Engine

**Goal:** Implement the NL mode placeholder in `BuilderPage.tsx` (Phase 6 / Step 13) with production-grade reliability.

### 2.1 Intent Intermediate Representation (Intent IR)

Instead of asking an LLM to generate raw DSL text directly (error-prone, hard to validate), introduce a structured **Intent IR** as an intermediate layer:

```
Natural Language → LLM → Intent IR (JSON) → Deterministic Codegen → DSL Text → WASM Validate
```

#### Intent IR Schema

```jsonc
{
  "intents": [
    {
      "type": "signal",
      "signal_type": "domain",       // one of 13 signal types
      "name": "math",
      "fields": { "categories": ["mathematics", "calculus"] }
    },
    {
      "type": "route",
      "name": "math_decision",
      "condition": {
        "op": "AND",
        "operands": [
          { "signal_type": "domain", "signal_name": "math" },
          { "signal_type": "complexity", "signal_name": "hard" }
        ]
      },
      "models": [
        { "model": "deepseek-r1", "reasoning": true, "effort": "high" }
      ],
      "plugins": ["hallucination_check"],
      "algorithm": { "type": "confidence", "params": { "threshold": 0.8 } }
    },
    {
      "type": "plugin_template",
      "name": "hallucination_check",
      "plugin_type": "hallucination",
      "fields": { "threshold": 0.7 }
    },
    {
      "type": "backend",
      "backend_type": "semantic_cache",
      "name": "main_cache",
      "fields": { "similarity_threshold": 0.9, "max_entries": 10000 }
    }
  ]
}
```

#### Why Intent IR?

| Benefit | Explanation |
|:---|:---|
| Constrained output space | LLM generates JSON conforming to a schema (~50 possible fields) instead of free-form DSL text (~infinite token sequences) |
| Validation before codegen | Can check intent conflicts (e.g., two routes with identical conditions) before generating DSL |
| User confirmation at intent level | Easier for non-expert users to review structured intents than raw DSL |
| Retry-friendly | On LLM error, only regenerate the failing intent, not the entire config |

#### Codegen: Intent IR → DSL

A deterministic TypeScript function (runs client-side, no LLM needed):

```typescript
// lib/intentToDsl.ts
function intentIRToDSL(ir: IntentIR): string {
  const sections: string[] = [];

  // Emit signals
  for (const intent of ir.intents.filter(i => i.type === 'signal')) {
    sections.push(`SIGNAL ${intent.signal_type} ${intent.name} {`);
    for (const [k, v] of Object.entries(intent.fields)) {
      sections.push(`  ${k}: ${formatValue(v)}`);
    }
    sections.push('}');
  }

  // Emit plugin templates, routes, backends, global...
  // (each with deterministic formatting)

  return sections.join('\n');
}
```

#### Data Flow (Complete)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          NL Mode UI                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────────────┐  │
│  │  NL Input     │────▶│ Intent Panel │────▶│ Generated DSL      │  │
│  │  (textarea)   │     │ (review/edit)│     │ (Monaco readonly)  │  │
│  └──────────────┘     └──────────────┘     └────────────────────┘  │
│         │                     │                      │              │
│         ▼                     ▼                      ▼              │
│  POST /api/nl/generate   User confirms        WASM validate       │
│         │                  or edits              │                  │
│         ▼                     │                  ▼                  │
│  ┌──────────────┐            │           ┌────────────────┐       │
│  │  LLM API     │            │           │ Diagnostics    │       │
│  │  (Intent IR)  │◀──retry───┘           │ (0 errors)     │       │
│  └──────────────┘                        └────────┬───────┘       │
│                                                   │               │
│                                          [Accept] button          │
│                                                   │               │
│                                     Switch to Visual/DSL Mode     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Grammar-Constrained Decoding

For scenarios where direct DSL generation is preferred (advanced users, CLI tooling), leverage the existing EBNF grammar in `parser.go` to **constrain LLM token sampling**:

```
parser.go EBNF rules → Finite State Machine → Token Allow-list per step → LLM decoding mask
```

#### Implementation Strategy

1. **Extract grammar** — Convert participle lexer rules + parser struct tags to a standalone EBNF/GBNF file.
2. **Compile to FSM** — Use tools like `llama.cpp`'s grammar sampling or `outlines` library to build a token-level FSM.
3. **Integrate with vLLM** — Since the project already targets vLLM backends, use vLLM's native grammar-guided decoding (`guided_grammar` parameter).

```python
# Example: vLLM API call with grammar constraint
response = client.completions.create(
    model="qwen3-8b",
    prompt=f"Generate a Signal DSL configuration for: {user_nl_input}\n\n```dsl\n",
    extra_body={
        "guided_grammar": open("signal_dsl.gbnf").read()  # extracted from parser.go
    }
)
```

**Guarantee:** Every generated token sequence is syntactically valid DSL. WASM validation only needs to check semantic-level issues (undefined references, constraint violations).

### 2.3 Context-Aware Few-Shot Selection

Build a dynamic few-shot example retriever that selects the most relevant DSL snippets for the LLM prompt:

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│ User NL      │────▶│ Keyword Extractor │────▶│ Example Index  │
│ Input        │     │ (signal types,   │     │ (vector store) │
│              │     │  plugin types,   │     │                │
│              │     │  algorithm names) │     │ Top-K examples │
└─────────────┘     └──────────────────┘     └────────┬───────┘
                                                      │
                    ┌──────────────────┐               │
                    │ Current Config   │               │
                    │ (decompiled DSL) │───────────────┤
                    └──────────────────┘               │
                                                      ▼
                                              ┌──────────────┐
                                              │ LLM Prompt   │
                                              │ (system +    │
                                              │  few-shot +  │
                                              │  context +   │
                                              │  user query) │
                                              └──────────────┘
```

**Example Index Construction:**

| Category | Source | Count |
|:---|:---|:---|
| Signal examples | Extract from `dsl_test.go` test cases | ~50 |
| Route examples | Extract from `fullDSLExample` + test configs | ~30 |
| Plugin examples | Extract from `TestCompileAllPluginTypes` | ~10 |
| Full config examples | `config/` directory YAML → decompile → DSL | ~50 |
| Edge cases | `validator.go` test diagnostics | ~20 |

### 2.4 Backend API Design

```go
// dashboard/backend/handlers/nl_generate.go

// POST /api/nl/generate
type NLGenerateRequest struct {
    Prompt        string `json:"prompt"`          // User's natural language input
    CurrentDSL    string `json:"current_dsl"`     // Existing DSL (for incremental edits)
    Mode          string `json:"mode"`            // "full" | "incremental" | "fix"
    MaxRetries    int    `json:"max_retries"`     // Auto-fix retry limit (default: 3)
}

type NLGenerateResponse struct {
    IntentIR      IntentIR       `json:"intent_ir"`       // Structured intents
    GeneratedDSL  string         `json:"generated_dsl"`   // Final DSL text
    Diagnostics   []Diagnostic   `json:"diagnostics"`     // WASM validation results
    Explanation   string         `json:"explanation"`      // AI explanation of what was generated
    Confidence    float64        `json:"confidence"`       // 0-1 confidence score
    Retries       int            `json:"retries"`          // Number of auto-fix retries used
}
```

### 2.5 Dataset Pipeline for Fine-Tuning

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ dsl_generator.py │────▶│ nl_dsl_pairs.py  │────▶│ fine_tune.py    │
│                  │     │                  │     │                 │
│ Random compose:  │     │ For each DSL:    │     │ Train on (NL,   │
│ signals + routes │     │ LLM generates    │     │ IntentIR, DSL)  │
│ + plugins +      │     │ NL description   │     │ triplets        │
│ backends + global│     │                  │     │                 │
│                  │     │ Quality filter:  │     │ Model: Qwen3-8B │
│ Output: ~10K     │     │ DSL ←→ NL ←→ DSL│     │ LoRA rank: 16   │
│ valid DSL configs│     │ round-trip check │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

**Validation loop:** For each generated (NL, DSL) pair, verify: `NL → LLM → DSL₂ → WASM compile → YAML₂` should be semantically equivalent to original `DSL → YAML₁`.

---

## 3. Direction 2: Explainable AI Logic Execution Layer

**Goal:** Make runtime routing decisions transparent and debuggable through structured execution traces.

### 3.1 Signal Trace DAG

Extend the router's `extproc/server.go` request processing to emit structured traces:

```go
// pkg/config/trace.go

type SignalTrace struct {
    RequestID     string                 `json:"request_id"`
    Timestamp     time.Time              `json:"timestamp"`
    Input         string                 `json:"input"`          // User query (truncated)
    SignalsEval   []SignalEvalResult      `json:"signals_evaluated"`
    RouteMatched  string                 `json:"route_matched"`
    WhenEval      *BoolExprTrace         `json:"when_evaluation"`
    ModelSelected string                 `json:"model_selected"`
    AlgorithmUsed string                 `json:"algorithm_used"`
    PluginsApplied []string              `json:"plugins_applied"`
    LatencyBreakdown map[string]float64  `json:"latency_ms"`    // per-stage latency
}

type SignalEvalResult struct {
    SignalType  string  `json:"signal_type"`   // "domain", "embedding", etc.
    SignalName  string  `json:"signal_name"`
    Result      bool    `json:"result"`
    Confidence  float64 `json:"confidence"`    // 0-1
    RawScore    float64 `json:"raw_score"`     // Pre-threshold score
    Threshold   float64 `json:"threshold"`     // Configured threshold
    LatencyMs   float64 `json:"latency_ms"`
}

type BoolExprTrace struct {
    Operator  string            `json:"op"`       // "AND" | "OR" | "NOT" | "LEAF"
    Result    bool              `json:"result"`
    Children  []*BoolExprTrace  `json:"children,omitempty"`
    Signal    *SignalEvalResult `json:"signal,omitempty"`  // Only for LEAF nodes
}
```

#### Trace Storage & Query

```
Request → extproc.Process() → SignalTrace → Ring Buffer (last 10K)
                                          → Dashboard API: GET /api/traces?route=math_decision&limit=100
                                          → Optional: export to OpenTelemetry span
```

### 3.2 Trace Visualization in Dashboard

Extend the existing `ExpressionBuilder.tsx` (ReactFlow + Dagre) to render live traces:

```
┌─────────────────────────────────────────────────────────┐
│  Request Trace Viewer                                    │
│                                                          │
│  Input: "Solve ∫x²dx from 0 to 5"                      │
│                                                          │
│         ┌──────────────┐                                │
│         │     AND      │  ← result: TRUE                │
│         │  (route hit) │                                │
│         └──┬───────┬───┘                                │
│            │       │                                     │
│    ┌───────▼──┐ ┌──▼─────────┐                          │
│    │ domain   │ │ embedding  │                          │
│    │ (math)   │ │ (calculus) │                          │
│    │ ✅ 0.94  │ │ ✅ 0.87    │                          │
│    │ thr: 0.5 │ │ thr: 0.8   │                          │
│    │ 12ms     │ │ 45ms       │                          │
│    └──────────┘ └────────────┘                          │
│                                                          │
│  Route: math_decision → Model: deepseek-r1 (confidence) │
│  Plugins: [hallucination ✅, semantic_cache MISS]        │
│  Total: 89ms                                             │
└─────────────────────────────────────────────────────────┘
```

**Color coding:**
- 🟢 Green: signal matched (above threshold)
- 🔴 Red: signal not matched
- 🟡 Yellow: signal close to threshold (within 10%)
- 🔵 Blue: cache hit (skipped evaluation)

### 3.3 What-If Simulation Mode

Add a simulation endpoint that evaluates signals without actually routing:

```go
// POST /api/simulate
type SimulateRequest struct {
    Input        string            `json:"input"`           // Test query
    Overrides    map[string]Value  `json:"overrides"`       // Override signal values
    DSLOverride  string            `json:"dsl_override"`    // Optional: test with modified DSL
}

type SimulateResponse struct {
    Trace        SignalTrace `json:"trace"`
    AllRoutes    []RouteEvalResult `json:"all_routes"`      // Show ALL routes, not just winner
}
```

UI integration: In `BuilderPage.tsx`, add a "Test Query" input that runs simulation in real-time as users edit DSL.

### 3.4 Reverse Explanation: Trace → DSL Fix Suggestions

When routing results are unexpected, AI analyzes the trace and suggests DSL modifications:

```go
// POST /api/traces/{id}/explain
type ExplainResponse struct {
    Summary      string           `json:"summary"`        // "Request was routed to math_decision instead of coding_decision"
    RootCause    string           `json:"root_cause"`     // "domain(math) matched due to low threshold (0.3)"
    Suggestions  []DSLSuggestion  `json:"suggestions"`
}

type DSLSuggestion struct {
    Description  string `json:"description"`   // "Increase math signal threshold"
    DSLPatch     string `json:"dsl_patch"`      // Unified diff format
    Impact       string `json:"impact"`         // "Estimated 15% fewer false positives for math routing"
    Risk         string `json:"risk"`           // "low" | "medium" | "high"
}
```

---

## 4. Direction 3: Context-Aware Dynamic Rule Adjustment (Adaptive DSL)

**Goal:** Evolve DSL from static configuration to a self-tuning system that adapts to production traffic patterns.

### 4.1 Signal Self-Calibration

Extend the DSL grammar with an `adaptive` block:

```dsl
SIGNAL keyword urgent {
  patterns: ["urgent", "asap", "emergency", "critical"]
  threshold: 0.5
  ADAPTIVE {
    enabled: true
    feedback_signal: user_feedback          # Reference to feedback signal
    calibration_window: 1000                # Re-calibrate every N requests
    min_threshold: 0.3                      # Safety bounds
    max_threshold: 0.9
    target_precision: 0.85                  # Optimize for this precision
    auto_apply: false                       # Require human approval
  }
}
```

#### Calibration Algorithm

```
Every calibration_window requests:
  1. Collect (signal_result, user_feedback_score) pairs
  2. Compute precision/recall at current threshold
  3. If precision < target_precision:
       new_threshold = binary_search(min, max, target_precision)
  4. Generate DSL patch:
       - threshold: 0.5
       + threshold: 0.62
  5. If auto_apply: hot-reload config
     Else: create pending review in dashboard
```

#### Compiler Extension

```go
// pkg/dsl/compiler.go — new field in SignalDecl
type AdaptiveConfig struct {
    Enabled            bool    `json:"enabled"`
    FeedbackSignal     string  `json:"feedback_signal"`
    CalibrationWindow  int     `json:"calibration_window"`
    MinThreshold       float64 `json:"min_threshold"`
    MaxThreshold       float64 `json:"max_threshold"`
    TargetPrecision    float64 `json:"target_precision"`
    AutoApply          bool    `json:"auto_apply"`
}
```

The compiler emits this into `RouterConfig.SignalCalibration[]`, which the runtime consumes.

### 4.2 A/B Experiment Primitive

Introduce `EXPERIMENT` as a new top-level DSL construct (the 6th construct):

```dsl
EXPERIMENT model_comparison {
  description: "Compare deepseek-r1 vs gpt-4o for math queries"
  traffic_split: [50, 50]
  duration: 7d
  
  VARIANT control {
    MODEL gpt-4o
    ALGORITHM confidence { threshold: 0.8 }
  }
  
  VARIANT treatment {
    MODEL deepseek-r1 (reasoning = true, effort = "high")
    ALGORITHM confidence { threshold: 0.7 }
  }
  
  SUCCESS_METRIC {
    primary: user_feedback.score
    guardrail: latency_p99 < 2000
    min_sample_size: 500
  }
  
  AUTO_PROMOTE: true     # Auto-promote winner when statistically significant
  ROLLBACK_ON_GUARDRAIL: true
}
```

#### Compilation

The compiler expands `EXPERIMENT` into:

1. Two shadow routes with traffic-splitting headers
2. A metrics collection plugin
3. A promotion webhook that rewrites the DSL when experiment concludes

```yaml
# Compiled output
decisions:
  - name: model_comparison_control
    traffic_percentage: 50
    experiment_id: model_comparison
    experiment_variant: control
    modelRefs: [{ model: gpt-4o }]
    
  - name: model_comparison_treatment
    traffic_percentage: 50
    experiment_id: model_comparison
    experiment_variant: treatment
    modelRefs: [{ model: deepseek-r1, reasoning: true }]

experiment_config:
  model_comparison:
    duration: 168h
    success_metric: user_feedback.score
    guardrail: { latency_p99_ms: 2000 }
    min_sample_size: 500
    auto_promote: true
```

### 4.3 Time & Load-Aware Conditional Routing

Extend the WHEN expression's signal vocabulary with runtime context signals:

```dsl
# New built-in signal types (no explicit SIGNAL declaration needed)
ROUTE peak_hour_routing {
  PRIORITY 10
  WHEN time_window("09:00-18:00", "Asia/Shanghai") AND load("high")
  MODEL gpt-4o-mini
  ALGORITHM latency_aware { p99_target_ms: 500 }
}

ROUTE offpeak_quality_routing {
  PRIORITY 5
  WHEN NOT time_window("09:00-18:00", "Asia/Shanghai")
  MODEL deepseek-r1 (reasoning = true)
  ALGORITHM confidence { threshold: 0.9 }
}

ROUTE overflow_routing {
  PRIORITY 100
  WHEN error_rate("gpt-4o", "> 0.05") OR latency("gpt-4o", "> 3000ms")
  MODEL gpt-4o-mini    # Automatic fallback
  PLUGIN fast_response { max_tokens: 256 }
}
```

#### New Built-in Signal Functions

| Function | Arguments | Description |
|:---|:---|:---|
| `time_window(range, tz)` | `"HH:MM-HH:MM"`, IANA timezone | True during specified time window |
| `load(level)` | `"low"` / `"medium"` / `"high"` | Based on current QPS percentile |
| `error_rate(backend, cond)` | Backend name, comparison expression | Recent error rate for a specific backend |
| `latency(backend, cond)` | Backend name, comparison expression | Recent P50/P99 latency check |
| `day_of_week(days)` | `"Mon,Tue,..."` | True on specified days |

---

## 5. Direction 4: AI-Assisted DSL Authoring (Copilot-in-Editor)

**Goal:** Transform the Monaco DSL editor from syntax-aware to semantics-aware with LLM-powered intelligence.

### 5.1 Deep Context-Aware Completion

Current state: `dslLanguage.ts` provides completions based on SymbolTable (defined signal/plugin names). Enhancement: LLM-driven completions that understand user intent and configuration state.

#### Architecture

```
Keystroke → Monaco CompletionProvider
  → Local completions (SymbolTable, keywords)          — < 5ms
  → LLM completions (background, debounced 300ms)      — < 500ms
  → Merge & rank by relevance
```

#### Ranking Signals

| Signal | Weight | Example |
|:---|:---|:---|
| Uncovered signal types | High | No `pii` signal defined → suggest `SIGNAL pii` |
| Uncovered routes | High | Signal `medical` exists but no route uses it |
| Co-occurrence patterns | Medium | `embedding` signal often paired with `semantic_cache` backend |
| User history | Medium | User frequently uses `confidence` algorithm |
| DSL best practices | Low | Routes without plugins → suggest `hallucination` plugin |

#### Implementation

```typescript
// lib/dslLanguage.ts — enhanced CompletionProvider

class LLMCompletionProvider implements monaco.languages.CompletionItemProvider {
  async provideCompletionItems(model, position, context) {
    const localItems = this.getLocalCompletions(model, position);  // existing logic
    
    // Background LLM call (debounced)
    const llmItems = await this.getLLMCompletions({
      currentLine: model.getLineContent(position.lineNumber),
      surroundingContext: this.getSurroundingLines(model, position, 10),
      symbolTable: this.currentSymbolTable,
      cursorConstruct: this.detectConstruct(model, position),  // "inside ROUTE", "after WHEN", etc.
    });
    
    return { suggestions: [...localItems, ...llmItems] };
  }
}
```

### 5.2 Level 4 Diagnostics: AI Best-Practice Checks

Extend `validator.go`'s 3-level diagnostics with an AI-powered 4th level:

| Level | Type | Source | Example |
|:---|:---|:---|:---|
| 1 (Error) | Syntax | Parser | `unexpected token "{"` |
| 2 (Warning) | Reference | Validator | `signal "matth" not defined, did you mean "math"?` |
| 3 (Constraint) | Range | Validator | `threshold 1.5 out of range [0, 1]` |
| **4 (AI Insight)** | **Best Practice** | **LLM** | **"Route 'catch_all' uses gpt-4 without rate limiting — consider adding a ratelimit in GLOBAL"** |

#### AI Diagnostics Categories

```go
// pkg/dsl/ai_diagnostics.go

type AIDiagnosticRule struct {
    ID          string
    Category    string
    Check       func(ast *Program) []Diagnostic
}

var aiRules = []AIDiagnosticRule{
    {
        ID: "AI001",
        Category: "redundancy",
        // Two routes with semantically equivalent WHEN conditions
    },
    {
        ID: "AI002", 
        Category: "performance",
        // Low-priority route using expensive model without latency_aware algorithm
    },
    {
        ID: "AI003",
        Category: "security",
        // Route handling user input without any guardrail plugin
    },
    {
        ID: "AI004",
        Category: "coverage",
        // Signal space has combinations not covered by any route WHEN clause
    },
    {
        ID: "AI005",
        Category: "cost",
        // Multiple routes use the same high-cost model; suggest model consolidation
    },
}
```

### 5.3 Semantic Diff for Deploy Preview

Enhance the existing deploy preview (diff view in `dslStore.ts`) with AI-generated semantic explanations:

```typescript
// stores/dslStore.ts — enhanced deploy preview

interface SemanticDiff {
  textDiff: string;           // Existing: unified diff
  summary: string;            // AI: "3 changes affecting math and coding routes"
  changes: SemanticChange[];
}

interface SemanticChange {
  entity: string;             // "ROUTE math_decision"
  field: string;              // "MODEL"
  oldValue: string;           // "gpt-4o"
  newValue: string;           // "deepseek-r1"
  impact: string;             // "Math queries will use reasoning model; latency may increase ~200ms"
  risk: "low" | "medium" | "high";
}
```

#### Example Output

```
Deploy Preview — 3 changes detected

1. [ROUTE math_decision] MODEL: gpt-4o → deepseek-r1 (reasoning=true)
   Impact: Math queries gain chain-of-thought reasoning; latency +200ms estimated
   Risk: LOW — deepseek-r1 benchmarks higher on MATH-500

2. [PLUGIN pii_guard] Added to routes: math_decision, coding_decision
   Impact: All routed requests will undergo PII scanning (+50ms)
   Risk: LOW — no functional change to routing logic

3. [BACKEND semantic_cache main] similarity_threshold: 0.95 → 0.85
   Impact: Cache hit rate estimated to increase ~20%
   Risk: MEDIUM — may serve less precise cached responses
   Recommendation: Monitor cache quality metrics for 24h after deploy
```

---

## 6. Direction 5: DSL as AI Agent Action Space

**Goal:** Model the DSL configuration space as an RL agent's action space for autonomous optimization.

### 6.1 Autonomous Routing Optimizer Agent

```
┌─────────────────────────────────────────────────────────────────┐
│                    Optimization Loop                             │
│                                                                  │
│  ┌──────────────┐     ┌────────────────┐     ┌───────────────┐  │
│  │ Observe       │────▶│ Agent (RL/LLM) │────▶│ Propose DSL   │  │
│  │ Metrics       │     │                │     │ Patch          │  │
│  │               │     │ State: current │     │               │  │
│  │ - quality_avg │     │   DSL AST +    │     │ - adjust      │  │
│  │ - latency_p99 │     │   metrics      │     │   thresholds  │  │
│  │ - cost_per_req│     │                │     │ - swap models │  │
│  │ - error_rate  │     │ Reward:        │     │ - add plugins │  │
│  │ - cache_hit   │     │   f(quality,   │     │               │  │
│  │               │     │   -latency,    │     │               │  │
│  │               │     │   -cost,       │     │               │  │
│  │               │     │   -errors)     │     │               │  │
│  └──────────────┘     └────────────────┘     └───────┬───────┘  │
│                                                       │          │
│                              ┌────────────────────────▼───────┐  │
│                              │  Validation Gate               │  │
│                              │  1. WASM compile (syntax OK?)  │  │
│                              │  2. Shadow deploy (safe?)      │  │
│                              │  3. Human review (optional)    │  │
│                              └────────────────────────┬───────┘  │
│                                                       │          │
│                              ┌────────────────────────▼───────┐  │
│                              │  Apply / Rollback              │  │
│                              │  Monitor for regression_window │  │
│                              └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

#### Action Space Definition

```python
# Typed action space derived from DSL mutations (mirrors dslMutations.ts)
@dataclass
class DSLAction:
    action_type: Literal[
        "adjust_threshold",    # Change signal threshold
        "swap_model",          # Change route's model
        "add_plugin",          # Add plugin to route
        "remove_plugin",       # Remove plugin from route
        "adjust_algorithm",    # Change algorithm params
        "add_route",           # Create new route
        "modify_when",         # Change route condition
        "adjust_priority",     # Change route priority
    ]
    target: str               # Entity name (signal/route/plugin)
    params: dict              # Action-specific parameters

# Example actions
actions = [
    DSLAction("adjust_threshold", "domain_math", {"delta": +0.1}),
    DSLAction("swap_model", "coding_decision", {"new_model": "gpt-4o-mini"}),
    DSLAction("add_plugin", "catch_all", {"plugin_type": "hallucination", "threshold": 0.7}),
]
```

#### Safety Constraints

| Constraint | Enforcement |
|:---|:---|
| Max change per cycle | ≤ 3 actions (prevent cascading changes) |
| Threshold bounds | Signal thresholds must stay within [min, max] from ADAPTIVE config |
| Model whitelist | Can only swap to pre-approved models |
| Rollback trigger | If any metric degrades > 10% within 1h, auto-rollback |
| Human gate | Changes to GLOBAL or new ROUTEs always require approval |

### 6.2 DSL Knowledge Graph

Build a typed knowledge graph from the DSL's type system:

```
                    ┌─────────────┐
          ┌────────│  domain      │────────┐
          │        │  (signal)    │        │
          │        └─────────────┘        │
          │ common_algorithm               │ common_plugin
          ▼                                ▼
  ┌──────────────┐                ┌──────────────┐
  │  confidence   │                │ hallucination │
  │  (algorithm)  │                │  (plugin)     │
  └──────────────┘                └──────────────┘
          │                                │
          │ used_with                       │ requires_backend
          ▼                                ▼
  ┌──────────────┐                ┌──────────────┐
  │  embedding    │                │ vllm_endpoint │
  │  (signal)     │───depends────▶│  (backend)    │
  └──────────────┘                └──────────────┘

  ┌──────────────┐
  │  rl_driven    │───requires───▶ user_feedback (signal)
  │  (algorithm)  │───requires───▶ vllm_endpoint (backend, for feedback storage)
  └──────────────┘
```

#### Graph Edges (Comprehensive)

| Relation | From → To | Example |
|:---|:---|:---|
| `requires_signal` | Algorithm → Signal | `rl_driven` → `user_feedback` |
| `requires_backend` | Plugin → Backend | `semantic_cache` (plugin) → `semantic_cache` (backend) |
| `common_algorithm` | Signal → Algorithm | `domain` → `confidence`, `router_dc`, `gmtrouter` |
| `common_plugin` | Signal → Plugin | `domain` → `hallucination` |
| `depends_on` | Signal → Backend | `embedding` → `embedding_model` (backend) |
| `conflicts_with` | Algorithm → Algorithm | `static` ⊥ `rl_driven` (static ignores feedback) |
| `enhances` | Plugin → Plugin | `semantic_cache` + `memory` (shared context improves cache) |

#### Usage

1. **NL → DSL inference:** "I want RL-based routing" → Graph traversal → auto-add `SIGNAL user_feedback` + `BACKEND vllm_endpoint`
2. **Completion ranking:** Inside a ROUTE, suggest algorithms that have `common_algorithm` edges to the route's WHEN signals
3. **Completeness check:** `ALGORITHM rl_driven` used but no `SIGNAL user_feedback` → warning via graph `requires_signal` edge
4. **Migration assistant:** "I'm using static routing and want to upgrade" → Graph suggests incremental path: `static` → `confidence` → `rl_driven`

---

## 7. Direction 6: Multi-Modal DSL Interaction

**Goal:** Expand DSL input/output beyond text to voice, images, and diagrams.

### 7.1 Voice → DSL

```
Voice Input → ASR (Whisper) → Text → NL→DSL Engine (Direction 1) → DSL
```

Use cases:
- Hands-free configuration during incident response
- Accessibility for visually impaired operators
- Quick adjustments: "Increase the math route threshold to 0.8"

#### Incremental Voice Commands

```
Command: "Add a PII plugin to the coding route"
→ Parse intent: { action: "add_plugin", route: "coding_decision", plugin_type: "pii" }
→ dslMutations.addPlugin("coding_decision", "pii", { action: "mask" })
→ DSL updated in-place
```

### 7.2 Screenshot → DSL (Configuration Migration)

```
Screenshot of another routing platform's UI
    → Vision Model (GPT-4V / Qwen-VL)
    → Extract: routes, conditions, models, thresholds
    → Intent IR
    → DSL
```

Prompt template:

```
You are analyzing a screenshot of a routing/gateway configuration UI.
Extract the following information as Intent IR JSON:
1. Routing rules (conditions, target models/backends)
2. Signal/detection patterns (keywords, categories)
3. Plugin/middleware configurations
4. Threshold values and parameters

Output format: [Intent IR JSON schema]
```

### 7.3 Diagram → DSL (Reverse ExpressionBuilder)

```
Whiteboard/flowchart image
    → Vision Model
    → Extract nodes (signals, routes, models) and edges (conditions, data flow)
    → Build boolean expression tree from edge structure
    → Generate WHEN clauses + ROUTE declarations
```

This is the inverse of `ExpressionBuilder.tsx`'s visual → AST flow, but starting from arbitrary diagrams rather than the structured ReactFlow canvas.

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

| Task | Direction | Deliverable | Effort |
|:---|:---|:---|:---|
| Intent IR schema + codegen | §2.1 | `lib/intentToDsl.ts`, `types/intentIR.ts` | 3d |
| NL Generate API endpoint | §2.4 | `handlers/nl_generate.go` | 3d |
| Few-shot example index | §2.3 | `lib/nlExamples.ts` + extracted examples | 2d |
| NL Mode UI (replace placeholder) | §2.1 | `components/builder/NLMode.tsx` | 4d |
| LLM completion provider | §5.1 | Enhanced `dslLanguage.ts` | 3d |
| Semantic diff for deploy | §5.3 | Enhanced `dslStore.ts` deploy preview | 2d |

**Milestone: NL Mode MVP + Smart Editor** — Users can describe configs in natural language and get validated DSL.

### Phase 2: Observability (Weeks 5-8)

| Task | Direction | Deliverable | Effort |
|:---|:---|:---|:---|
| Signal Trace struct + ring buffer | §3.1 | `pkg/config/trace.go` | 3d |
| Trace API endpoints | §3.1 | `handlers/traces.go` | 2d |
| Trace visualization (ReactFlow) | §3.2 | `components/TraceViewer.tsx` | 4d |
| What-if simulation endpoint | §3.3 | `handlers/simulate.go` | 3d |
| Reverse explanation API | §3.4 | `handlers/explain.go` | 3d |

**Milestone: Explainable Routing** — Every routing decision has a visual, queryable trace.

### Phase 3: Adaptive & Experiments (Weeks 9-12)

| Task | Direction | Deliverable | Effort |
|:---|:---|:---|:---|
| ADAPTIVE block in grammar | §4.1 | Extended `ast.go`, `parser.go`, `compiler.go` | 4d |
| Calibration runtime loop | §4.1 | `pkg/config/calibration.go` | 4d |
| EXPERIMENT construct | §4.2 | New AST node + compiler + decompiler | 5d |
| Time/load signals | §4.3 | Built-in signal evaluators | 3d |
| AI diagnostics (Level 4) | §5.2 | `pkg/dsl/ai_diagnostics.go` | 4d |

**Milestone: Self-Tuning Router** — Signals auto-calibrate, experiments auto-promote.

### Phase 4: Agent & Multi-Modal (Weeks 13-16)

| Task | Direction | Deliverable | Effort |
|:---|:---|:---|:---|
| Knowledge graph construction | §6.2 | `pkg/dsl/knowledge_graph.go` | 3d |
| Grammar-constrained decoding | §2.2 | GBNF export + vLLM integration | 4d |
| Optimizer agent framework | §6.1 | `pkg/agent/optimizer.go` | 5d |
| Dataset pipeline | §2.5 | `src/vllm-sr/dataset/` scripts | 4d |
| Voice → DSL prototype | §7.1 | Whisper integration + NL pipeline | 3d |
| Screenshot → DSL prototype | §7.2 | Vision model integration | 3d |

**Milestone: Autonomous Optimization** — Agent proposes configuration improvements from production metrics.

### Priority Matrix

```
                    High Impact
                        │
         ┌──────────────┼──────────────┐
         │   P0          │              │
         │  NL Mode      │  Adaptive    │
         │  Smart Editor  │  Signals     │
         │  Signal Trace  │  Experiments │
  Low ───┼──────────────┼──────────────┼─── High
  Effort │              │              │    Effort
         │  Semantic Diff│  Knowledge   │
         │  AI Diag L4   │  Graph       │
         │              │  Agent       │
         │   P1          │   P2          │
         └──────────────┼──────────────┘
                        │
                    Low Impact
```

---

## 9. Key Technical Decisions

### 9.1 LLM Provider Strategy

| Scenario | Recommended Model | Reason |
|:---|:---|:---|
| NL → Intent IR | Qwen3-32B / GPT-4o | Complex structured output, needs high accuracy |
| DSL Completion | Qwen3-8B (fine-tuned) | Low latency critical, domain-specific |
| Semantic Diff | Qwen3-14B | Medium complexity, cost-sensitive |
| Trace Explanation | Qwen3-8B | Template-driven, fast |
| Agent Optimization | Qwen3-32B | Complex reasoning over metrics |

### 9.2 WASM vs Server-Side Processing

| Processing | Location | Reason |
|:---|:---|:---|
| DSL compile/validate | WASM (browser) | Sub-millisecond, no network latency |
| NL → Intent IR | Server (LLM API) | Requires LLM inference |
| Intent IR → DSL codegen | WASM (browser) | Deterministic, instant |
| Trace collection | Server (extproc) | Runtime data |
| AI diagnostics | Server (batch) | May need LLM for semantic analysis |
| Agent optimization | Server (async) | Long-running, needs metric access |

### 9.3 Backwards Compatibility

All new DSL constructs (ADAPTIVE, EXPERIMENT, time/load signals) are **additive**:

- Existing DSL without new constructs → compiles identically to before
- New constructs → compile to extended `RouterConfig` fields
- Old runtimes that don't understand new fields → ignore them gracefully
- Decompiler: if `RouterConfig` has calibration/experiment data → emit new constructs; otherwise → identical output
