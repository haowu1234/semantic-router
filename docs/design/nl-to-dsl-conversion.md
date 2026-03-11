# Natural Language → DSL Conversion: Technical Design

## 1. Problem Statement

The Signal DSL compiler provides a complete toolchain — Lexer, Parser (participle), Compiler (AST → `RouterConfig`), Decompiler, Validator (3-level diagnostics + QuickFix), and WASM bridge — but authoring DSL still requires domain knowledge of:

- **13 signal types** (`keyword`, `embedding`, `domain`, `fact_check`, `user_feedback`, `preference`, `language`, `context`, `complexity`, `modality`, `authz`, `jailbreak`, `pii`) and their field schemas
- **14 algorithm types** (`confidence`, `ratings`, `remom`, `elo`, `router_dc`, `automix`, `hybrid`, `rl_driven`, `gmtrouter`, `latency_aware`, `static`, `knn`, `kmeans`, `svm`) and configuration parameters
- **9 plugin types** (`system_prompt`, `semantic_cache`, `hallucination`, `memory`, `rag`, `header_mutation`, `router_replay`, `image_gen`, `fast_response`) and their field schemas
- **6 backend types** (`vllm_endpoint`, `provider_profile`, `embedding_model`, `semantic_cache`, `memory`, `response_api`)
- **Boolean expression composition** (AND / OR / NOT over signal references)

The NL Mode (currently a placeholder at `BuilderPage.tsx:536`, `disabled`, "Phase 6") aims to let users describe routing intent in natural language and get valid DSL.

### Design Goals

1. **Syntax-guaranteed output** — Every generated token sequence passes `parser.go`'s Lexer + Parser.
2. **Semantic validation** — Generated DSL passes `validator.go`'s 3-level checks (references, constraints).
3. **Self-healing** — Auto-repair loop for validator warnings/errors (max 3 retries).
4. **Incremental editing** — Support both "generate from scratch" and "modify existing DSL" modes.
5. **Sub-3-second latency** — Full NL→validated DSL in under 3 seconds for typical requests.
6. **Offline-capable** — WASM validation runs entirely client-side; only LLM inference requires network.

---

## 2. Conversion Architecture

### 2.1 Three-Stage Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     NL → DSL Conversion Pipeline                         │
│                                                                          │
│   Stage 1: Understanding          Stage 2: Generation       Stage 3: V&R │
│   ┌─────────────────────┐   ┌─────────────────────┐   ┌────────────────┐│
│   │ Input Analysis       │──▶│ Structured Gen       │──▶│ Validate &    ││
│   │                      │   │                      │   │ Repair         ││
│   │ • NL parsing         │   │ • Intent IR build    │   │               ││
│   │ • Intent extraction  │   │ • IR → DSL codegen   │   │ • WASM compile││
│   │ • Context injection  │   │ • Template selection  │   │ • Diagnostics ││
│   │ • Ambiguity detect   │   │ • Constraint enforce  │   │ • Auto-fix    ││
│   └─────────────────────┘   └─────────────────────┘   └────────────────┘│
│          ▲                          ▲                         │          │
│          │                          │                         │          │
│     Current DSL              Few-shot examples          Error feedback   │
│     SymbolTable              Knowledge Graph            (retry loop)     │
│     User history             DSL Schema defs                             │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Why Not Direct DSL Generation?

Direct LLM→DSL has fundamental reliability problems:

| Failure Mode | Example | Frequency |
|:---|:---|:---|
| **Lexer violation** | Missing quotes: `patterns: [urgent, asap]` (needs `["urgent", "asap"]`) | ~15% |
| **Parser violation** | Wrong nesting: `ROUTE { SIGNAL ... }` (SIGNAL is top-level only) | ~8% |
| **Reference error** | `WHEN domain("math")` but no `SIGNAL domain math` defined | ~20% |
| **Constraint error** | `threshold: 1.5` (valid range: 0.0–1.0) | ~10% |
| **Schema error** | `SIGNAL keyword { categories: [...] }` (wrong field for keyword type) | ~12% |
| **Semantic mismatch** | User said "PII protection" but LLM generated `hallucination` plugin | ~5% |

The three-stage pipeline isolates each concern: Stage 1 handles understanding, Stage 2 enforces structure, Stage 3 catches residual errors.

---

## 3. Stage 1: Input Text Analysis & Intent Extraction

### 3.1 Input Classification

Every NL input is classified into one of four **operation modes**:

```typescript
type NLOperationMode =
  | 'generate'        // "Create a config that routes math to reasoning models"
  | 'modify'          // "Add PII protection to the math route"
  | 'explain'         // "What does this config do?"  (no DSL change)
  | 'fix'             // "Fix the errors in my config"
```

Classification uses a lightweight prompt (or regex heuristics for common patterns):

```
generate triggers: "create", "build", "set up", "configure", "I need", "I want"
modify triggers:   "add", "remove", "change", "update", "increase", "decrease", "rename"
explain triggers:  "what does", "explain", "describe", "how does"
fix triggers:      "fix", "repair", "resolve", "the error"
```

### 3.2 Entity Extraction

Extract DSL-relevant entities from the NL input using the DSL's type system as a vocabulary:

#### Signal Type Detection

Map natural language concepts to the 13 signal types:

```typescript
const SIGNAL_NL_MAP: Record<string, { signalType: string; confidence: number }[]> = {
  // Domain signals
  "math": [{ signalType: "domain", confidence: 0.95 }],
  "mathematics": [{ signalType: "domain", confidence: 0.95 }],
  "coding": [{ signalType: "domain", confidence: 0.95 }],
  "programming": [{ signalType: "domain", confidence: 0.95 }],
  "medical": [{ signalType: "domain", confidence: 0.90 }],

  // Keyword signals
  "urgent": [{ signalType: "keyword", confidence: 0.90 }],
  "keywords": [{ signalType: "keyword", confidence: 0.95 }],
  "pattern matching": [{ signalType: "keyword", confidence: 0.85 }],

  // Embedding signals
  "semantic similarity": [{ signalType: "embedding", confidence: 0.95 }],
  "similar topics": [{ signalType: "embedding", confidence: 0.85 }],
  "vector search": [{ signalType: "embedding", confidence: 0.90 }],

  // Plugin-like but are signals
  "PII": [{ signalType: "pii", confidence: 0.95 }],
  "personal information": [{ signalType: "pii", confidence: 0.90 }],
  "jailbreak": [{ signalType: "jailbreak", confidence: 0.95 }],
  "prompt injection": [{ signalType: "jailbreak", confidence: 0.85 }],

  // Complexity
  "simple questions": [{ signalType: "complexity", confidence: 0.85 }],
  "hard problems": [{ signalType: "complexity", confidence: 0.85 }],
  "complex queries": [{ signalType: "complexity", confidence: 0.80 }],

  // Context
  "long context": [{ signalType: "context", confidence: 0.90 }],
  "short messages": [{ signalType: "context", confidence: 0.85 }],

  // Multi-signal (ambiguous NL maps to multiple types)
  "safety": [
    { signalType: "jailbreak", confidence: 0.70 },
    { signalType: "pii", confidence: 0.60 },
  ],
  "content filtering": [
    { signalType: "jailbreak", confidence: 0.65 },
    { signalType: "pii", confidence: 0.65 },
  ],
};
```

#### Algorithm Detection

```typescript
const ALGO_NL_MAP: Record<string, string> = {
  "confidence-based": "confidence",
  "confidence threshold": "confidence",
  "reinforcement learning": "rl_driven",
  "RL routing": "rl_driven",
  "Elo rating": "elo",
  "latency-optimized": "latency_aware",
  "low latency": "latency_aware",
  "cost-effective": "automix",
  "cascade": "automix",
  "reasoning mix": "remom",
  "mixture of models": "remom",
  "static": "static",
  "round robin": "static",
};
```

#### Model Detection

```typescript
const MODEL_NL_MAP: Record<string, { model: string; reasoning?: boolean; effort?: string }> = {
  "GPT-4": { model: "gpt-4" },
  "GPT-4o": { model: "gpt-4o" },
  "GPT-4o mini": { model: "gpt-4o-mini" },
  "DeepSeek R1": { model: "deepseek-r1", reasoning: true, effort: "high" },
  "reasoning model": { model: "deepseek-r1", reasoning: true },
  "cheap model": { model: "gpt-4o-mini" },
  "fast model": { model: "gpt-4o-mini" },
  "powerful model": { model: "gpt-4o" },
  "Qwen": { model: "qwen3-32b" },
};
```

### 3.3 Context Injection

Before LLM inference, assemble a rich context window:

```typescript
interface NLContext {
  // Current state (from dslStore)
  currentDSL: string;                    // Existing DSL text (for modify/fix modes)
  symbolTable: SymbolTable;              // From WASM signalValidate()
  diagnostics: Diagnostic[];             // Current validation errors

  // Schema knowledge (static, compiled into prompt)
  signalTypes: SignalTypeSchema[];       // All 13 types + field schemas
  pluginTypes: PluginTypeSchema[];       // All 9 types + field schemas
  algorithmTypes: AlgoTypeSchema[];      // All 14 types + field schemas
  backendTypes: BackendTypeSchema[];     // All 6 types + field schemas

  // Dynamic context
  fewShotExamples: FewShotExample[];     // Selected by relevance (§3.5)
  userHistory: string[];                 // Last 5 NL→DSL interactions
}
```

### 3.4 Ambiguity Detection & Resolution

When entity extraction yields ambiguous results (multiple signal types for one concept), the system either:

1. **Auto-resolve** (confidence gap > 0.2): Pick the highest-confidence match.
2. **Ask user** (confidence gap ≤ 0.2): Surface a clarification in the UI.

```typescript
// Example: "Add safety checks" → jailbreak(0.70) vs pii(0.60)
// Gap = 0.10 < 0.2 → Ask user:
//   "By 'safety checks', do you mean:
//    (a) Jailbreak/prompt injection detection
//    (b) PII/personal information protection
//    (c) Both"
```

### 3.5 Few-Shot Example Selection

#### Example Corpus Construction

Extract examples from three sources:

| Source | Extraction Method | Count |
|:---|:---|:---|
| `dsl_test.go` test cases | Parse `fullDSLExample` + individual test DSL strings | ~60 |
| `config/` YAML files | `WASM signalDecompile()` each YAML → DSL | ~50 |
| Synthetic pairs | `dsl_generator.py` (randomized valid DSL) + LLM NL descriptions | ~200 |

#### Retrieval Strategy

For each NL input, select top-3 examples using a two-phase retriever:

```
Phase 1: Keyword filter — match extracted entities (signal types, algorithm names, plugin types)
          against example metadata tags
Phase 2: Embedding similarity — encode NL input + example NL descriptions with embedding model,
          rank by cosine similarity
```

#### Example Format in Prompt

```
### Example 1
User: "Route math questions to a reasoning model with confidence-based selection"
DSL:
```dsl
SIGNAL domain math {
  description: "Mathematics and calculations"
  categories: ["mathematics"]
}

ROUTE math_decision {
  WHEN domain("math")
  MODEL "deepseek-r1" (reasoning = true, effort = "high")
  ALGORITHM confidence {
    threshold: 0.8
  }
}
```

### Example 2
User: "Add PII protection that masks sensitive data"
DSL:
```dsl
SIGNAL pii sensitive_data {
  action: "mask"
  threshold: 0.8
  pii_types_allowed: ["email", "phone"]
}
```
```

---

## 4. Stage 2: Structured Generation via Intent IR

### 4.1 Intent IR Schema (Complete)

The Intent IR is the bridge between LLM understanding and deterministic DSL codegen. It mirrors the DSL's 5 constructs but with relaxed constraints:

```typescript
// types/intentIR.ts

interface IntentIR {
  version: "1.0";
  operation: "generate" | "modify" | "fix";
  intents: Intent[];
}

// ─── Signal Intent ───
interface SignalIntent {
  type: "signal";
  signal_type: SignalType;            // One of 13 types
  name: string;                       // Signal name (auto-sanitized)
  fields: Record<string, unknown>;    // Type-specific fields
}

// ─── Route Intent ───
interface RouteIntent {
  type: "route";
  name: string;
  description?: string;
  priority?: number;
  condition: ConditionNode;           // Boolean expression tree
  models: ModelIntent[];
  algorithm?: AlgorithmIntent;
  plugins?: PluginRefIntent[];
}

interface ConditionNode {
  op: "AND" | "OR" | "NOT" | "SIGNAL_REF";
  // For AND/OR:
  operands?: ConditionNode[];
  // For NOT:
  operand?: ConditionNode;
  // For SIGNAL_REF:
  signal_type?: string;
  signal_name?: string;
}

interface ModelIntent {
  model: string;
  reasoning?: boolean;
  effort?: "low" | "medium" | "high";
  lora?: string;
  weight?: number;
}

interface AlgorithmIntent {
  algo_type: string;                  // One of 14 types
  params: Record<string, unknown>;
}

interface PluginRefIntent {
  name: string;                       // Template name or plugin type
  overrides?: Record<string, unknown>;
}

// ─── Plugin Template Intent ───
interface PluginTemplateIntent {
  type: "plugin_template";
  name: string;
  plugin_type: string;                // One of 9 types
  fields: Record<string, unknown>;
}

// ─── Backend Intent ───
interface BackendIntent {
  type: "backend";
  backend_type: string;               // One of 6 types
  name: string;
  fields: Record<string, unknown>;
}

// ─── Global Intent ───
interface GlobalIntent {
  type: "global";
  fields: Record<string, unknown>;
}

// ─── Modification Intent (for "modify" mode) ───
interface ModifyIntent {
  type: "modify";
  action: "add" | "update" | "delete";
  target_construct: "signal" | "route" | "plugin" | "backend" | "global";
  target_name: string;
  changes: Record<string, unknown>;   // Fields to change
}

type Intent = SignalIntent | RouteIntent | PluginTemplateIntent | BackendIntent | GlobalIntent | ModifyIntent;
```

### 4.2 LLM Prompt Design for Intent IR Generation

#### System Prompt (Condensed Schema)

```
You are a Signal DSL configuration expert. Given a natural language description, 
generate an Intent IR (JSON) that captures the user's routing configuration intent.

## DSL Type System (complete reference)

### Signal Types (use in condition.signal_type):
- keyword:    Pattern matching. Fields: { patterns: string[], threshold: 0-1, operator: "OR"|"AND", method: "keyword_match"|"bm25" }
- embedding:  Semantic similarity. Fields: { threshold: 0-1, candidates: string[], aggregation_method: "mean"|"max" }
- domain:     Topic classification. Fields: { description: string, categories: string[] }
- fact_check:  Factual verification. Fields: { description: string }
- user_feedback: User ratings. Fields: { description: string }
- preference:  User preference. Fields: { description: string }
- language:    Language detection. Fields: { description: string }
- context:     Token count. Fields: { min_tokens: int, max_tokens: int }
- complexity:  Query difficulty. Fields: { threshold: 0-1 }
- modality:    Input modality. Fields: { description: string }
- authz:       Authorization. Fields: { role: string, subjects: [{ user: string }] }
- jailbreak:   Injection detection. Fields: { method: string, threshold: 0-1 }
- pii:         PII detection. Fields: { action: "mask"|"block", threshold: 0-1, pii_types_allowed: string[] }

### Algorithm Types (use in algorithm.algo_type):
- confidence:    Confidence threshold. Params: { threshold: 0-1, confidence_method: string }
- ratings:       Quality ratings. Params: { initial_score: float }
- elo:           Elo system. Params: { initial_rating: int }
- rl_driven:     Reinforcement learning. Params: { exploration_rate: 0-1 }
- router_dc:     Contrastive learning. Params: { threshold: 0-1 }
- automix:       Cost-aware cascade. Params: { cascade_threshold: 0-1 }
- latency_aware: Latency-optimized. Params: { p99_target_ms: int }
- remom:         Reasoning mixture. Params: { threshold: 0-1 }
- gmtrouter:     RL-trained routing. Params: { model_path: string }
- hybrid:        Combined strategy. Params: { sub_algorithms: object[] }
- static:        Fixed routing. Params: {}
- knn/kmeans/svm: ML-based. Params: { ... }

### Plugin Types (use in plugins[].name or plugin_template.plugin_type):
- system_prompt:   Custom system prompt. Fields: { prompt: string }
- semantic_cache:  Response caching. Fields: { similarity_threshold: 0-1, max_entries: int }
- hallucination:   Output verification. Fields: { threshold: 0-1 }
- memory:          Conversation memory. Fields: { max_tokens: int }
- rag:             Retrieval-augmented. Fields: { source: string }
- header_mutation:  HTTP header rewrite. Fields: { ... }
- router_replay:   Request replay. Fields: { ... }
- image_gen:       Image generation. Fields: { ... }
- fast_response:   Quick replies. Fields: { max_tokens: int }

### Backend Types (use in backend.backend_type):
- vllm_endpoint:     Model endpoint. Fields: { host: string, port: int, model_name: string }
- provider_profile:  API provider. Fields: { provider: string, api_key_env: string, ... }
- embedding_model:   Embedding backend. Fields: { model_name: string, ... }
- semantic_cache:    Cache backend. Fields: { similarity_threshold: 0-1, max_entries: int }
- memory:            Memory backend. Fields: { max_entries: int }
- response_api:      Response API. Fields: { endpoint: string }

## Rules
1. EVERY signal referenced in a route condition MUST have a corresponding signal intent.
2. Field values must respect type constraints (thresholds: 0-1, ports: 1-65535).
3. For "modify" operation, only include the changed entities.
4. Use descriptive signal names (snake_case).
5. Output ONLY valid JSON conforming to the Intent IR schema. No explanation.
```

#### User Prompt Template

```
{few_shot_examples}

Current configuration context:
- Defined signals: {symbolTable.signals}
- Defined routes: {symbolTable.routes}
- Defined plugins: {symbolTable.plugins}
- Current errors: {diagnostics}

User request: "{user_nl_input}"
Operation mode: {operation_mode}

Generate the Intent IR JSON:
```

### 4.3 Intent IR → DSL Codegen (Deterministic)

This is a pure function — no LLM involved. It leverages the field schemas already defined in `dslMutations.ts`:

```typescript
// lib/intentToDsl.ts

export function intentIRToDSL(ir: IntentIR, existingDSL?: string): string {
  if (ir.operation === 'modify' && existingDSL) {
    return applyModifications(ir, existingDSL);
  }
  return generateFullDSL(ir);
}

function generateFullDSL(ir: IntentIR): string {
  const sections: string[] = [];

  // ─── Signals ───
  const signals = ir.intents.filter(i => i.type === 'signal') as SignalIntent[];
  if (signals.length > 0) {
    sections.push('# SIGNALS\n');
    for (const sig of signals) {
      sections.push(emitSignal(sig));
    }
  }

  // ─── Plugin Templates ───
  const templates = ir.intents.filter(i => i.type === 'plugin_template') as PluginTemplateIntent[];
  if (templates.length > 0) {
    sections.push('\n# PLUGINS\n');
    for (const tmpl of templates) {
      sections.push(emitPluginTemplate(tmpl));
    }
  }

  // ─── Routes ───
  const routes = ir.intents.filter(i => i.type === 'route') as RouteIntent[];
  if (routes.length > 0) {
    sections.push('\n# ROUTES\n');
    for (const route of routes) {
      sections.push(emitRoute(route));
    }
  }

  // ─── Backends ───
  const backends = ir.intents.filter(i => i.type === 'backend') as BackendIntent[];
  if (backends.length > 0) {
    sections.push('\n# BACKENDS\n');
    for (const be of backends) {
      sections.push(emitBackend(be));
    }
  }

  // ─── Global ───
  const globals = ir.intents.filter(i => i.type === 'global') as GlobalIntent[];
  if (globals.length > 0) {
    sections.push('\n# GLOBAL\n');
    for (const g of globals) {
      sections.push(emitGlobal(g));
    }
  }

  return sections.join('\n');
}

// ─── Emitters (guarantee lexer/parser compliance) ───

function emitSignal(sig: SignalIntent): string {
  const lines = [`SIGNAL ${sig.signal_type} ${sanitizeName(sig.name)} {`];
  for (const [key, value] of Object.entries(sig.fields)) {
    lines.push(`  ${key}: ${serializeValue(value)}`);
  }
  lines.push('}\n');
  return lines.join('\n');
}

function emitRoute(route: RouteIntent): string {
  const opts = route.description ? ` (description = "${escapeString(route.description)}")` : '';
  const lines = [`ROUTE ${sanitizeName(route.name)}${opts} {`];

  if (route.priority !== undefined) {
    lines.push(`  PRIORITY ${route.priority}`);
  }

  if (route.condition) {
    lines.push(`  WHEN ${emitCondition(route.condition)}`);
  }

  for (const model of route.models) {
    const opts = emitModelOpts(model);
    lines.push(`  MODEL "${model.model}"${opts}`);
  }

  if (route.algorithm) {
    lines.push(`  ALGORITHM ${route.algorithm.algo_type} {`);
    for (const [k, v] of Object.entries(route.algorithm.params)) {
      lines.push(`    ${k}: ${serializeValue(v)}`);
    }
    lines.push('  }');
  }

  if (route.plugins) {
    for (const plugin of route.plugins) {
      if (plugin.overrides && Object.keys(plugin.overrides).length > 0) {
        lines.push(`  PLUGIN ${sanitizeName(plugin.name)} {`);
        for (const [k, v] of Object.entries(plugin.overrides)) {
          lines.push(`    ${k}: ${serializeValue(v)}`);
        }
        lines.push('  }');
      } else {
        lines.push(`  PLUGIN ${sanitizeName(plugin.name)}`);
      }
    }
  }

  lines.push('}\n');
  return lines.join('\n');
}

function emitCondition(node: ConditionNode): string {
  switch (node.op) {
    case 'SIGNAL_REF':
      return `${node.signal_type}("${node.signal_name}")`;
    case 'AND':
      return node.operands!.map(o => emitCondition(o)).join(' AND ');
    case 'OR': {
      const parts = node.operands!.map(o => emitCondition(o));
      // Wrap AND sub-expressions in parens when inside OR
      return parts.join(' OR ');
    }
    case 'NOT':
      return `NOT ${emitCondition(node.operand!)}`;
  }
}

// ─── Value serialization (mirrors dslMutations.ts serializeValue) ───

function serializeValue(v: unknown): string {
  if (typeof v === 'string')  return `"${escapeString(v)}"`;
  if (typeof v === 'number')  return String(v);
  if (typeof v === 'boolean') return String(v);
  if (Array.isArray(v))       return `[${v.map(serializeValue).join(', ')}]`;
  if (typeof v === 'object' && v !== null) {
    const entries = Object.entries(v).map(([k, val]) => `${k}: ${serializeValue(val)}`);
    if (entries.length <= 3 && entries.every(e => !e.includes('{'))) {
      return `{ ${entries.join(', ')} }`;
    }
    return `{\n    ${entries.join('\n    ')}\n  }`;
  }
  return String(v);
}

function sanitizeName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_\-]/g, '_');
}

function escapeString(s: string): string {
  return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}
```

### 4.4 Modification Mode — Applying Intent IR to Existing DSL

For `modify` operations, the codegen leverages `dslMutations.ts` functions directly:

```typescript
// lib/intentToDsl.ts

function applyModifications(ir: IntentIR, existingDSL: string): string {
  let dsl = existingDSL;

  for (const intent of ir.intents) {
    if (intent.type !== 'modify') continue;
    const mod = intent as ModifyIntent;

    switch (mod.action) {
      case 'add':
        dsl = applyAdd(dsl, mod);
        break;
      case 'update':
        dsl = applyUpdate(dsl, mod);
        break;
      case 'delete':
        dsl = applyDelete(dsl, mod);
        break;
    }
  }

  // Also handle non-modify intents (new signals/routes added alongside modifications)
  const newSignals = ir.intents.filter(i => i.type === 'signal') as SignalIntent[];
  for (const sig of newSignals) {
    dsl = addSignal(dsl, sig.signal_type, sig.name, sig.fields);
  }

  const newRoutes = ir.intents.filter(i => i.type === 'route') as RouteIntent[];
  for (const route of newRoutes) {
    dsl = addRoute(dsl, route.name, {
      description: route.description,
      priority: route.priority,
      when: route.condition ? emitCondition(route.condition) : undefined,
      models: route.models,
      algorithm: route.algorithm,
      plugins: route.plugins,
    });
  }

  return dsl;
}

function applyAdd(dsl: string, mod: ModifyIntent): string {
  switch (mod.target_construct) {
    case 'signal':
      return addSignal(dsl, mod.changes.signal_type as string, mod.target_name, mod.changes.fields as Record<string, unknown>);
    case 'route':
      return addRoute(dsl, mod.target_name, mod.changes as RouteInput);
    case 'plugin':
      return addPlugin(dsl, mod.target_name, mod.changes.plugin_type as string, mod.changes.fields as Record<string, unknown>);
    case 'backend':
      return addBackend(dsl, mod.changes.backend_type as string, mod.target_name, mod.changes.fields as Record<string, unknown>);
    default:
      return dsl;
  }
}

function applyUpdate(dsl: string, mod: ModifyIntent): string {
  switch (mod.target_construct) {
    case 'signal':
      return updateSignal(dsl, mod.changes.signal_type as string, mod.target_name, mod.changes.fields as Record<string, unknown>);
    case 'route':
      return updateRoute(dsl, mod.target_name, mod.changes as RouteInput);
    case 'global':
      return updateGlobal(dsl, mod.changes.fields as Record<string, unknown>);
    default:
      return dsl;
  }
}

function applyDelete(dsl: string, mod: ModifyIntent): string {
  switch (mod.target_construct) {
    case 'signal':
      return deleteSignal(dsl, mod.changes.signal_type as string, mod.target_name);
    case 'route':
      return deleteRoute(dsl, mod.target_name);
    case 'plugin':
      return deletePlugin(dsl, mod.target_name, mod.changes.plugin_type as string);
    case 'backend':
      return deleteBackend(dsl, mod.changes.backend_type as string, mod.target_name);
    default:
      return dsl;
  }
}
```

---

## 5. Stage 3: Validation & Self-Repair Loop

### 5.1 Validation Pipeline

The generated DSL goes through the full WASM validation stack:

```typescript
// lib/nlValidation.ts

interface ValidationResult {
  isValid: boolean;
  dsl: string;
  diagnostics: Diagnostic[];
  errorCount: number;
  warningCount: number;
  symbolTable: SymbolTable;
  yaml?: string;           // Only if compilation succeeds
}

async function validateGeneratedDSL(dsl: string, wasm: WasmBridge): Promise<ValidationResult> {
  // Step 1: Validate (diagnostics + symbol table)
  const validateResult = await wasm.validate(dsl);

  // Step 2: If no errors, also compile to get YAML
  let yaml: string | undefined;
  if (validateResult.errorCount === 0) {
    const compileResult = await wasm.compile(dsl);
    if (compileResult.diagnostics.filter(d => d.level === 0).length === 0) {
      yaml = compileResult.yaml;
    }
  }

  return {
    isValid: validateResult.errorCount === 0,
    dsl,
    diagnostics: validateResult.diagnostics,
    errorCount: validateResult.errorCount,
    warningCount: validateResult.diagnostics.filter(d => d.level === 1).length,
    symbolTable: validateResult.symbols,
    yaml,
  };
}
```

### 5.2 Self-Repair Strategy

When validation fails, the system attempts automatic repair through three escalating strategies:

```typescript
// lib/nlRepair.ts

async function repairDSL(
  originalNL: string,
  generatedDSL: string,
  diagnostics: Diagnostic[],
  wasm: WasmBridge,
  llmClient: LLMClient,
  attempt: number,      // 1, 2, or 3
): Promise<string> {

  // Strategy 1: Deterministic QuickFix application (attempt 1)
  if (attempt === 1) {
    return applyQuickFixes(generatedDSL, diagnostics);
  }

  // Strategy 2: Targeted LLM repair (attempt 2)
  if (attempt === 2) {
    return llmTargetedRepair(generatedDSL, diagnostics, llmClient);
  }

  // Strategy 3: Full regeneration with error context (attempt 3)
  return llmFullRegeneration(originalNL, generatedDSL, diagnostics, llmClient);
}
```

#### Strategy 1: Deterministic QuickFix

Leverage `validator.go`'s QuickFix suggestions (returned via WASM as `diagnostic.fixes[]`):

```typescript
function applyQuickFixes(dsl: string, diagnostics: Diagnostic[]): string {
  // Sort fixes by line number (descending) to avoid position shifts
  const fixes = diagnostics
    .flatMap(d => (d.fixes || []).map(f => ({ ...f, line: d.line, column: d.column })))
    .sort((a, b) => b.line - a.line);

  let result = dsl;
  const lines = result.split('\n');

  for (const fix of fixes) {
    // Apply each fix by replacing the token at the diagnostic position
    const line = lines[fix.line - 1];
    if (line && fix.replacement) {
      // Find the token at the column position
      const before = line.substring(0, fix.column - 1);
      const after = line.substring(fix.column - 1);
      // Replace the first token-like sequence
      const replaced = after.replace(/[\w\-\.]+/, fix.replacement);
      lines[fix.line - 1] = before + replaced;
    }
  }

  return lines.join('\n');
}
```

This handles:
- Typos in signal names (Levenshtein → "did you mean math?")
- Unknown algorithm types → closest valid type
- Threshold out-of-range → clamped to [0, 1]
- Negative priority → set to 0

#### Strategy 2: Targeted LLM Repair

Send only the error context (not the full config) to the LLM for surgical fixes:

```
The following DSL has validation errors. Fix ONLY the errors listed below.
Do not change anything else. Return the complete corrected DSL.

Errors:
1. Line 5: signal "mathmatics" not defined (did you mean "math"?)
2. Line 12: threshold 1.5 out of range [0.0, 1.0]
3. Line 18: route "urgent_route" has no MODEL specified

Current DSL:
```dsl
{generated_dsl}
```
```

#### Strategy 3: Full Regeneration

If targeted repair fails, regenerate from scratch with the error history as negative examples:

```
Previous attempt generated this DSL but it had these errors:
{error_summary}

Please regenerate the Intent IR, avoiding these mistakes:
- Do not use signal name "mathmatics", use "math" instead
- Keep all thresholds between 0.0 and 1.0
- Every route must have at least one MODEL

Original user request: "{original_nl}"
```

### 5.3 Complete Repair Loop

```typescript
// lib/nlPipeline.ts

const MAX_RETRIES = 3;

async function nlToDSL(
  nlInput: string,
  context: NLContext,
  wasm: WasmBridge,
  llmClient: LLMClient,
  onProgress?: (step: string) => void,
): Promise<NLGenerateResult> {

  onProgress?.('Analyzing input...');

  // Stage 1: Classify & extract
  const mode = classifyOperation(nlInput);
  const entities = extractEntities(nlInput);

  onProgress?.('Generating configuration...');

  // Stage 2: LLM → Intent IR → DSL
  const intentIR = await generateIntentIR(nlInput, context, llmClient);
  let dsl = intentIRToDSL(intentIR, mode === 'modify' ? context.currentDSL : undefined);

  onProgress?.('Validating...');

  // Stage 3: Validate & repair loop
  let validation = await validateGeneratedDSL(dsl, wasm);
  let attempt = 0;

  while (!validation.isValid && attempt < MAX_RETRIES) {
    attempt++;
    onProgress?.(`Fixing issues (attempt ${attempt}/${MAX_RETRIES})...`);

    dsl = await repairDSL(
      nlInput, dsl, validation.diagnostics, wasm, llmClient, attempt
    );
    validation = await validateGeneratedDSL(dsl, wasm);
  }

  return {
    dsl,
    yaml: validation.yaml,
    diagnostics: validation.diagnostics,
    isValid: validation.isValid,
    intentIR,
    confidence: computeConfidence(validation, attempt),
    retries: attempt,
    explanation: generateExplanation(intentIR),
  };
}

function computeConfidence(validation: ValidationResult, retries: number): number {
  let score = 1.0;
  if (!validation.isValid) score -= 0.5;
  score -= retries * 0.1;              // Penalty per retry
  score -= validation.warningCount * 0.05; // Penalty per warning
  return Math.max(0, Math.min(1, score));
}
```

---

## 6. Syntax Tree Construction & Semantic Mapping

### 6.1 NL → AST Mapping Rules

Each NL pattern maps through Intent IR to specific AST nodes (defined in `ast.go`):

```
┌─────────────────────────┬──────────────────────┬─────────────────────────────┐
│ NL Pattern               │ Intent IR             │ AST Node (ast.go)            │
├─────────────────────────┼──────────────────────┼─────────────────────────────┤
│ "route X to model Y"    │ RouteIntent           │ RouteDecl                    │
│   "when condition Z"    │   .condition          │   .When: BoolExpr            │
│   "using algorithm A"   │   .algorithm          │   .Algorithm: AlgoSpec       │
│   "with plugin P"       │   .plugins            │   .Plugins: []PluginRef      │
│                          │                       │                              │
│ "detect keyword K"      │ SignalIntent(keyword) │ SignalDecl{keyword, K, ...}  │
│ "match topic T"         │ SignalIntent(domain)  │ SignalDecl{domain, T, ...}   │
│ "similar to S"          │ SignalIntent(embedding)│ SignalDecl{embedding, S, ...}│
│                          │                       │                              │
│ "X AND Y"               │ ConditionNode(AND)    │ BoolAnd{Left, Right}         │
│ "X OR Y"                │ ConditionNode(OR)     │ BoolOr{Left, Right}          │
│ "NOT X"                 │ ConditionNode(NOT)    │ BoolNot{Expr}                │
│ "when X happens"        │ ConditionNode(REF)    │ SignalRefExpr{type, name}    │
│                          │                       │                              │
│ "cache responses"       │ BackendIntent(cache)  │ BackendDecl{semantic_cache}  │
│ "connect to endpoint"   │ BackendIntent(vllm)   │ BackendDecl{vllm_endpoint}   │
│                          │                       │                              │
│ "default model is M"    │ GlobalIntent          │ GlobalDecl{default_model: M} │
│ "set strategy to S"     │ GlobalIntent          │ GlobalDecl{strategy: S}      │
└─────────────────────────┴──────────────────────┴─────────────────────────────┘
```

### 6.2 Boolean Expression Composition from NL

Complex NL conditions map to the `BoolExpr` tree:

```
NL: "Route to GPT-4 when it's a math question about calculus but not a simple one"

Analysis:
  - "math question"     → domain("math")
  - "about calculus"    → embedding("calculus")
  - "not a simple one"  → NOT complexity("simple")

Condition tree:
  AND
  ├── domain("math")
  ├── embedding("calculus")
  └── NOT
      └── complexity("simple")

Generated WHEN clause:
  WHEN domain("math") AND embedding("calculus") AND NOT complexity("simple")
```

**Operator precedence in NL mapping:**

| NL Connector | DSL Operator | Precedence | Example NL |
|:---|:---|:---|:---|
| "and", "also", "with", "plus" | AND | 2 (binds tighter) | "math and coding" |
| "or", "alternatively", "either...or" | OR | 1 (binds looser) | "math or physics" |
| "not", "except", "exclude", "but not" | NOT | 3 (unary, tightest) | "not simple" |

### 6.3 Implicit Dependency Resolution

When generating a route, auto-detect and create missing signals:

```typescript
function resolveImplicitDependencies(ir: IntentIR): IntentIR {
  const definedSignals = new Set<string>();

  // Collect explicitly defined signals
  for (const intent of ir.intents) {
    if (intent.type === 'signal') {
      definedSignals.add(`${intent.signal_type}:${intent.name}`);
    }
  }

  // Scan route conditions for signal references
  const missingSignals: SignalIntent[] = [];
  for (const intent of ir.intents) {
    if (intent.type === 'route') {
      const refs = collectSignalRefs(intent.condition);
      for (const ref of refs) {
        const key = `${ref.signal_type}:${ref.signal_name}`;
        if (!definedSignals.has(key)) {
          missingSignals.push(createDefaultSignal(ref.signal_type, ref.signal_name));
          definedSignals.add(key);
        }
      }
    }
  }

  // Also check algorithm dependencies (e.g., rl_driven needs user_feedback)
  for (const intent of ir.intents) {
    if (intent.type === 'route' && intent.algorithm) {
      const requiredSignals = ALGO_SIGNAL_DEPS[intent.algorithm.algo_type] || [];
      for (const req of requiredSignals) {
        if (!definedSignals.has(`${req.type}:${req.name}`)) {
          missingSignals.push(createDefaultSignal(req.type, req.name));
          definedSignals.add(`${req.type}:${req.name}`);
        }
      }
    }
  }

  return {
    ...ir,
    intents: [...missingSignals, ...ir.intents],
  };
}

// Knowledge graph: algorithm → required signals
const ALGO_SIGNAL_DEPS: Record<string, { type: string; name: string }[]> = {
  rl_driven: [{ type: 'user_feedback', name: 'feedback' }],
  ratings:   [{ type: 'user_feedback', name: 'feedback' }],
  elo:       [{ type: 'user_feedback', name: 'feedback' }],
};
```

---

## 7. Complex Expression Handling

### 7.1 Nested Boolean Expressions

For complex NL descriptions, the system builds nested condition trees:

```
NL: "Route to the reasoning model when the query is about math or physics, 
     and it's a complex problem, but not when the user is asking in Chinese"

Parse:
  Topic condition:  domain("math") OR domain("physics")
  Complexity:       complexity("hard")
  Language exclude:  NOT language("chinese")

Tree construction:
  AND
  ├── OR                           ← grouped by "or" between topics
  │   ├── domain("math")
  │   └── domain("physics")
  ├── complexity("hard")
  └── NOT language("chinese")

DSL output:
  WHEN (domain("math") OR domain("physics")) AND complexity("hard") AND NOT language("chinese")
```

**Parenthesization rules:**
- OR inside AND → requires parens (handled by `emitCondition()`)
- AND inside OR → no parens needed (AND binds tighter)
- NOT always binds to the immediately following expression

### 7.2 Multi-Route Generation

NL that implies multiple routing rules:

```
NL: "Set up three tiers: simple questions go to GPT-4o-mini, 
     medium complexity to GPT-4o, and hard problems to DeepSeek-R1 with reasoning"

Generated intents:
  Signal: complexity("simple"), complexity("medium"), complexity("hard")
  Route 1: simple_tier    → WHEN complexity("simple")    → MODEL gpt-4o-mini
  Route 2: medium_tier    → WHEN complexity("medium")    → MODEL gpt-4o
  Route 3: hard_tier      → WHEN complexity("hard")      → MODEL deepseek-r1 (reasoning=true)

Priority assignment: hard(10) > medium(5) > simple(1)  [higher = checked first]
```

### 7.3 Plugin Composition Patterns

Detect when the same plugin config is used across routes → extract template:

```typescript
function optimizePluginTemplates(ir: IntentIR): IntentIR {
  // Count plugin usage across routes
  const pluginUsage = new Map<string, { routes: string[]; fields: Record<string, unknown> }>();

  for (const intent of ir.intents) {
    if (intent.type === 'route') {
      for (const plugin of intent.plugins || []) {
        const key = `${plugin.name}:${JSON.stringify(plugin.overrides || {})}`;
        if (!pluginUsage.has(key)) {
          pluginUsage.set(key, { routes: [], fields: plugin.overrides || {} });
        }
        pluginUsage.get(key)!.routes.push(intent.name);
      }
    }
  }

  // Extract templates for plugins used 2+ times (matches decompiler.go logic)
  const templates: PluginTemplateIntent[] = [];
  const templateNameMap = new Map<string, string>();

  for (const [key, usage] of pluginUsage) {
    if (usage.routes.length >= 2) {
      const pluginType = key.split(':')[0];
      const templateName = `shared_${pluginType}`;
      templates.push({
        type: 'plugin_template',
        name: templateName,
        plugin_type: pluginType,
        fields: usage.fields,
      });
      templateNameMap.set(key, templateName);
    }
  }

  // Replace inline plugins with template references
  const updatedIntents = ir.intents.map(intent => {
    if (intent.type !== 'route') return intent;
    return {
      ...intent,
      plugins: (intent.plugins || []).map(plugin => {
        const key = `${plugin.name}:${JSON.stringify(plugin.overrides || {})}`;
        const templateName = templateNameMap.get(key);
        if (templateName) {
          return { name: templateName, overrides: {} };
        }
        return plugin;
      }),
    };
  });

  return {
    ...ir,
    intents: [...templates, ...updatedIntents],
  };
}
```

---

## 8. Error Handling & Edge Cases

### 8.1 Error Taxonomy

| Error Category | Detection Point | Handling |
|:---|:---|:---|
| **NL ambiguity** | Stage 1 (entity extraction) | Ask clarifying question or pick highest-confidence |
| **LLM format error** | Stage 2 (Intent IR parse) | Retry with stricter JSON schema prompt |
| **Schema violation** | Stage 2 (IR validation) | Map to closest valid schema via field similarity |
| **Syntax error** | Stage 3 (WASM parse) | Should never happen with codegen; if it does, full regenerate |
| **Reference error** | Stage 3 (WASM validate L2) | Auto-create missing signals via dependency resolution |
| **Constraint error** | Stage 3 (WASM validate L3) | Apply QuickFix (clamp values to valid ranges) |
| **Semantic mismatch** | User review | User edits Intent IR or regenerates with feedback |

### 8.2 LLM Output Parsing Robustness

```typescript
function parseIntentIR(llmOutput: string): IntentIR {
  // 1. Try direct JSON parse
  try {
    return JSON.parse(llmOutput);
  } catch {}

  // 2. Extract JSON from markdown code blocks
  const jsonMatch = llmOutput.match(/```(?:json)?\s*\n([\s\S]*?)\n```/);
  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[1]);
    } catch {}
  }

  // 3. Find first { ... } block
  const braceMatch = llmOutput.match(/\{[\s\S]*\}/);
  if (braceMatch) {
    try {
      return JSON.parse(braceMatch[0]);
    } catch {}
  }

  // 4. Last resort: ask LLM to fix its own output
  throw new IntentIRParseError(llmOutput);
}
```

### 8.3 Graceful Degradation

When the pipeline fails completely (all 3 retries exhausted):

```typescript
function handlePipelineFailure(
  nlInput: string,
  lastDSL: string,
  lastDiagnostics: Diagnostic[],
): NLGenerateResult {
  return {
    dsl: lastDSL,                      // Return best-effort DSL
    yaml: undefined,                    // No valid YAML
    diagnostics: lastDiagnostics,
    isValid: false,
    confidence: 0.1,
    retries: 3,
    // Guide user to fix remaining issues manually
    explanation: `Generated DSL has ${lastDiagnostics.length} remaining issues. ` +
      `Please switch to DSL mode to fix: ${lastDiagnostics.map(d => d.message).join('; ')}`,
    suggestedMode: 'dsl',              // Suggest switching to DSL editor
  };
}
```

---

## 9. Context Understanding & Incremental Editing

### 9.1 Conversation Memory

Maintain a session-level conversation history for multi-turn NL interactions:

```typescript
interface NLSession {
  id: string;
  turns: NLTurn[];
  currentDSL: string;
  symbolTable: SymbolTable;
}

interface NLTurn {
  userInput: string;
  intentIR: IntentIR;
  generatedDSL: string;
  accepted: boolean;       // Did user accept this generation?
}
```

This enables follow-up commands:

```
Turn 1: "Create a routing config for math and coding queries"
→ Generates: SIGNAL domain math, SIGNAL domain coding, ROUTE math_decision, ROUTE coding_decision

Turn 2: "Add PII protection to both routes"
→ Context: knows "both routes" = math_decision + coding_decision
→ Generates: ModifyIntent(add plugin pii to math_decision), ModifyIntent(add plugin pii to coding_decision)

Turn 3: "Increase the math threshold to 0.9"
→ Context: knows "the math threshold" = math_decision's confidence algorithm threshold
→ Generates: ModifyIntent(update math_decision algorithm.params.threshold = 0.9)
```

### 9.2 Pronoun & Reference Resolution

```typescript
function resolveReferences(
  nlInput: string,
  session: NLSession,
): string {
  // Replace pronouns/references with concrete entity names

  // "it" → last modified entity
  // "both" / "all" → all entities of the type being discussed
  // "the route" → if only one route mentioned in last turn, use that
  // "the threshold" → look up most recently mentioned numeric field

  let resolved = nlInput;

  // "both routes" → explicit names
  if (/both routes/i.test(resolved)) {
    const routes = session.symbolTable.routes;
    resolved = resolved.replace(/both routes/i, routes.join(' and '));
  }

  // "the math threshold" → identify field path
  if (/the (\w+) threshold/i.test(resolved)) {
    const match = resolved.match(/the (\w+) threshold/i);
    if (match) {
      const signalName = match[1];
      // Check if it's a signal threshold or algorithm threshold
      const signal = session.symbolTable.signals.find(s => s.name === signalName);
      if (signal) {
        resolved = resolved.replace(
          /the \w+ threshold/i,
          `the threshold of signal ${signal.type} ${signal.name}`
        );
      }
    }
  }

  return resolved;
}
```

### 9.3 Undo / Redo for NL Operations

NL operations integrate with the DSL Store's undo/redo stack:

```typescript
// In dslStore.ts — NL generates DSL text, which flows through normal setDslSource()
// This means every NL operation is automatically undo-able via Ctrl+Z

function acceptNLResult(result: NLGenerateResult) {
  // Push current DSL to undo stack
  undoStack.push(currentDSL);
  // Set new DSL (triggers compile + validate via WASM)
  setDslSource(result.dsl);
  // Record in NL session
  nlSession.turns[nlSession.turns.length - 1].accepted = true;
}

function rejectNLResult() {
  // Discard and allow retry
  nlSession.turns[nlSession.turns.length - 1].accepted = false;
}
```

---

## 10. Extensible Grammar Specification for Domain Adaptation

### 10.1 Plugin-Based Schema Extension

When new signal/plugin/algorithm types are added to the DSL, the NL pipeline adapts via a schema registry:

```typescript
// lib/nlSchemaRegistry.ts

interface NLSchemaEntry {
  // DSL identity
  construct: 'signal' | 'plugin' | 'algorithm' | 'backend';
  type_name: string;                    // e.g., "keyword", "confidence"

  // NL mapping
  nl_triggers: string[];                // Words/phrases that indicate this type
  nl_description: string;               // One-sentence description for LLM context
  nl_examples: string[];                // Example NL phrases that use this type

  // Field schema (mirrors dslMutations.ts getSignalFieldSchema)
  fields: FieldSchema[];

  // Dependencies
  requires_backend?: string[];          // Backend types this entity needs
  requires_signal?: string[];           // Signal types this entity needs
}

// Example: registering a hypothetical new signal type
const newSignalSchema: NLSchemaEntry = {
  construct: 'signal',
  type_name: 'sentiment',
  nl_triggers: ['sentiment', 'mood', 'emotion', 'feeling', 'positive', 'negative'],
  nl_description: 'Detects the emotional sentiment of user queries (positive/negative/neutral)',
  nl_examples: [
    'Route positive sentiment queries to the cheerful model',
    'Detect when users are frustrated',
  ],
  fields: [
    { name: 'threshold', type: 'float', min: 0, max: 1, default: 0.5 },
    { name: 'categories', type: 'string[]', default: ['positive', 'negative', 'neutral'] },
  ],
  requires_backend: [],
  requires_signal: [],
};

// Register → automatically available in NL entity extraction + LLM system prompt
schemaRegistry.register(newSignalSchema);
```

### 10.2 Automatic LLM Prompt Regeneration

When the schema registry changes, the system prompt is automatically rebuilt:

```typescript
function buildSystemPrompt(registry: NLSchemaRegistry): string {
  const sections: string[] = [SYSTEM_PROMPT_HEADER];

  // Signal types section
  sections.push('### Signal Types');
  for (const entry of registry.getByConstruct('signal')) {
    sections.push(`- ${entry.type_name}: ${entry.nl_description}`);
    sections.push(`  Fields: ${entry.fields.map(f => `${f.name}(${f.type})`).join(', ')}`);
  }

  // Similarly for plugins, algorithms, backends
  // ...

  return sections.join('\n');
}
```

### 10.3 Grammar Extension Points

The parser (`parser.go`) is designed to be extensible through field-based constructs. Adding a new signal type requires:

1. **No parser changes** — New signal types just add entries to the validation whitelist in `validator.go`
2. **Compiler change** — Add a `compile{Type}Signal()` method in `compiler.go`
3. **Decompiler change** — Add reverse mapping in `decompiler.go`
4. **NL schema entry** — Register in `nlSchemaRegistry.ts`

The grammar itself (`SIGNAL <type> <name> { fields }`) is generic enough that arbitrary field structures are supported without grammar changes.

---

## 11. Backend API Specification

### 11.1 Endpoints

```go
// dashboard/backend/handlers/nl_generate.go

// POST /api/nl/generate — Main NL→DSL endpoint
// POST /api/nl/explain  — Explain existing DSL in natural language
// POST /api/nl/suggest  — Get NL-based suggestions for improving DSL
```

### 11.2 Request/Response Types

```go
// POST /api/nl/generate
type NLGenerateRequest struct {
    Prompt      string   `json:"prompt"`               // User's NL input
    CurrentDSL  string   `json:"current_dsl,omitempty"` // For modify/fix modes
    Mode        string   `json:"mode"`                  // "generate" | "modify" | "fix"
    SessionID   string   `json:"session_id,omitempty"`  // Multi-turn session
    MaxRetries  int      `json:"max_retries,omitempty"` // Default: 3
}

type NLGenerateResponse struct {
    DSL          string        `json:"dsl"`               // Generated/modified DSL
    YAML         string        `json:"yaml,omitempty"`     // Compiled YAML (if valid)
    IntentIR     json.RawMessage `json:"intent_ir"`        // Structured intent for UI review
    Diagnostics  []Diagnostic  `json:"diagnostics"`        // Remaining issues
    IsValid      bool          `json:"is_valid"`
    Confidence   float64       `json:"confidence"`         // 0-1
    Retries      int           `json:"retries"`            // Retries used
    Explanation  string        `json:"explanation"`         // NL explanation of what was generated
    SessionID    string        `json:"session_id"`
}

// POST /api/nl/explain
type NLExplainRequest struct {
    DSL string `json:"dsl"`
}

type NLExplainResponse struct {
    Summary     string              `json:"summary"`       // One-paragraph summary
    Entities    []EntityExplanation `json:"entities"`      // Per-construct explanations
}

type EntityExplanation struct {
    Construct   string `json:"construct"`   // "SIGNAL", "ROUTE", etc.
    Name        string `json:"name"`
    Description string `json:"description"` // NL explanation
}
```

### 11.3 Server-Side Pipeline

```go
func (h *NLHandler) handleGenerate(w http.ResponseWriter, r *http.Request) {
    var req NLGenerateRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        writeError(w, 400, "INVALID_REQUEST", err.Error())
        return
    }

    // 1. Build context
    ctx := h.buildContext(req)

    // 2. Call LLM for Intent IR
    intentIR, err := h.llmClient.GenerateIntentIR(ctx, req.Prompt)
    if err != nil {
        writeError(w, 500, "LLM_ERROR", err.Error())
        return
    }

    // 3. Resolve dependencies
    intentIR = resolveImplicitDependencies(intentIR)

    // 4. Codegen: Intent IR → DSL
    dslText := intentIRToDSL(intentIR, req.CurrentDSL)

    // 5. Validate using the actual DSL compiler
    prog, parseErrs := dsl.Parse(dslText)
    var diagnostics []dsl.Diagnostic
    if prog != nil {
        diagnostics, _ = dsl.ValidateWithSymbols(dslText)
    }

    // 6. Self-repair loop
    retries := 0
    maxRetries := req.MaxRetries
    if maxRetries == 0 { maxRetries = 3 }

    for len(filterErrors(diagnostics)) > 0 && retries < maxRetries {
        retries++
        dslText = h.repair(req.Prompt, dslText, diagnostics, retries)
        diagnostics, _ = dsl.ValidateWithSymbols(dslText)
    }

    // 7. Compile to YAML if valid
    var yamlOutput string
    if len(filterErrors(diagnostics)) == 0 {
        cfg, compileErrs := dsl.CompileAST(prog)
        if len(compileErrs) == 0 {
            yamlOutput, _ = dsl.EmitUserYAML(cfg)
        }
    }

    // 8. Generate explanation
    explanation := h.explainIntentIR(intentIR)

    json.NewEncoder(w).Encode(NLGenerateResponse{
        DSL:         dslText,
        YAML:        yamlOutput,
        IntentIR:    marshalIntentIR(intentIR),
        Diagnostics: diagnostics,
        IsValid:     len(filterErrors(diagnostics)) == 0,
        Confidence:  computeConfidence(diagnostics, retries),
        Retries:     retries,
        Explanation:  explanation,
        SessionID:   req.SessionID,
    })
}
```

---

## 12. Performance Budget

| Stage | Target Latency | Method |
|:---|:---|:---|
| NL classification | < 10ms | Regex heuristics (client-side) |
| Entity extraction | < 50ms | Static mapping tables (client-side) |
| Context assembly | < 20ms | SymbolTable from WASM cache |
| Few-shot selection | < 100ms | Pre-indexed example corpus |
| LLM Intent IR generation | < 2000ms | Streaming; Qwen3-32B w/ JSON schema constraint |
| Intent IR → DSL codegen | < 5ms | Deterministic TypeScript (client-side) |
| WASM validation | < 10ms | `signalValidate()` — already sub-ms |
| QuickFix application | < 5ms | String manipulation (client-side) |
| LLM repair (if needed) | < 1500ms | Targeted prompt, smaller context |
| **Total (happy path)** | **< 2.2s** | |
| **Total (with 1 repair)** | **< 3.7s** | |

### Optimization Strategies

1. **Streaming UI** — Show Intent IR as it's generated (user reviews while codegen runs)
2. **Speculative validation** — Start WASM validation before LLM finishes (validate partial output)
3. **Prompt caching** — Cache the system prompt + schema (doesn't change between requests)
4. **Model selection** — Use Qwen3-8B (fine-tuned) for repair attempts (faster, domain-specific)

---

## 13. File Deliverables

```
# Frontend
dashboard/frontend/src/
├── types/
│   └── intentIR.ts                    # Intent IR type definitions
├── lib/
│   ├── intentToDsl.ts                 # Intent IR → DSL codegen
│   ├── nlPipeline.ts                  # Full NL→DSL pipeline orchestrator
│   ├── nlValidation.ts                # WASM validation wrapper
│   ├── nlRepair.ts                    # Self-repair strategies
│   ├── nlEntityExtraction.ts          # NL entity → DSL type mapping
│   ├── nlSchemaRegistry.ts            # Extensible schema registry
│   └── nlExamples.ts                  # Few-shot example corpus + retriever
├── components/builder/
│   └── NLMode.tsx                     # NL Mode UI (replaces placeholder)
└── stores/
    └── nlStore.ts                     # NL session state (Zustand)

# Backend
dashboard/backend/handlers/
└── nl_generate.go                     # NL→DSL API endpoint

# Dataset Pipeline
src/vllm-sr/dataset/
├── dsl_generator.py                   # Random valid DSL generator
├── nl_dsl_pairs.py                    # NL↔DSL training pair generator
└── fine_tune.py                       # Qwen3-8B LoRA fine-tuning

# Tests
src/semantic-router/pkg/dsl/
└── nl_integration_test.go             # End-to-end NL→DSL→YAML tests
dashboard/frontend/e2e/
└── nl-mode.spec.ts                    # Playwright E2E for NL Mode
```

---

## 14. Validation Criteria

| Test Case | Input | Expected Output | Validates |
|:---|:---|:---|:---|
| Basic domain routing | "Route math questions to GPT-4o" | SIGNAL domain math + ROUTE with MODEL gpt-4o | Entity extraction, codegen |
| Multi-signal condition | "Math or physics queries with high complexity" | WHEN (domain("math") OR domain("physics")) AND complexity("hard") | Boolean composition |
| Plugin addition | "Add hallucination check to all routes" | ModifyIntent per route | Modify mode |
| Threshold adjustment | "Lower the confidence threshold to 0.6" | ModifyIntent with updated param | Context resolution |
| Invalid threshold auto-fix | User says "threshold 2.0" | QuickFix clamps to 1.0 | Self-repair |
| Typo recovery | "Signal domian math" | Validator QuickFix → "domain" | Levenshtein repair |
| Multi-route generation | "Three tiers: simple, medium, hard" | 3 routes with priority ordering | Complex generation |
| Template extraction | "Same cache config on 3 routes" | PLUGIN template + 3 refs | Optimization |
| Follow-up command | Turn 1: create, Turn 2: "add PII to it" | Resolves "it" from session | Context memory |
| Full round-trip | NL → DSL → YAML → decompile → DSL₂ | DSL ≈ DSL₂ (semantically) | Pipeline integrity |
