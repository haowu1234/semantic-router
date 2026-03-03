# NL → DSL Tool Calling 架构设计

## 1. 问题背景

当前 NL → DSL 转换采用 **Monolithic Prompt + Structured Output** 方案：LLM 在一个巨大的 system prompt（~5000 tokens）中接收 40+ 种 DSL 类型定义，输出完整的 Intent IR JSON。这种方案存在三个核心问题：

| 问题 | 表现 | 根因 |
|:---|:---|:---|
| **类型幻觉** | `plugin_type: "pii"`, `backend_type: "vector_store"` 误用 | LLM 在自然语言描述中记不住 enum 边界 |
| **Prompt 膨胀** | System prompt ~5000 tokens, 含 40 种类型完整字段定义 | 所有类型信息 "塞满" 而非 "按需检索" |
| **输出脆弱** | 一次性输出 200+ 行 JSON，任何位置错误导致整体失败 | 输出粒度太大，无法局部重试 |

### 设计目标

1. **消灭类型幻觉** — 通过 JSON Schema `enum` 硬约束替代自然语言提示
2. **精简 prompt** — 从 ~5000 tokens 降至 ~800 tokens，类型约束由 tool schema 承载
3. **提升首次通过率** — 从 ~60-70% 提升至 ~85-95%
4. **保持兼容性** — 不支持 tool calling 的 LLM 自动 fallback 到现有 structured output 方案

---

## 2. 核心洞察

当前架构让 LLM 同时承担两个职责：

1. **理解意图** — 用户说什么 → 需要哪些 DSL 构造
2. **序列化结构** — 把理解组装成正确的 JSON 嵌套结构

第 2 步是 LLM 最容易出错的地方。**Tool Calling 把第 2 步从 "LLM 记忆" 转移到 "API 约束"。** LLM 只需理解意图并选择正确的 tool，具体的类型枚举和字段结构由 JSON Schema 硬性约束。

### 对比

```
当前方案（Structured Output）:
  System: "你是 DSL 专家，这是 40 种类型定义，这是 JSON Schema，这是 10 条规则..."
  User: "6 个 examples + 用户输入"
  → LLM 一次性输出整个 Intent IR JSON（~100-300 行）
  → response_format: { type: 'json_object' }

Tool Calling 方案:
  System: "你是 DSL 路由配置助手"（~800 tokens）
  User: "用户输入"
  Tools: [create_signal, create_route, create_plugin, ...]
  → LLM 发出多次 parallel tool calls，每个 call 对应一个 DSL 实体
  → 前端执行 tool → 拼装 Intent IR → 转 DSL
```

---

## 3. Tool 设计

### 3.1 Tool 划分策略

经过三种方案对比，选择 **方案 C：混合两层 tool**：

| 维度 | 方案 A（1 个大 tool） | 方案 B（N 个实体 tool） | 方案 C（混合两层） |
|:---|:---|:---|:---|
| Schema 复杂度 | 极高（嵌套 union） | 低（每个 tool 扁平） | 低 |
| LLM 调用轮次 | 1 轮 | 1 轮（parallel calls） | 1-2 轮 |
| 类型约束强度 | 中（大 schema LLM 易忽略） | 强（enum 短） | 强 |
| 按需信息获取 | 不支持 | 不支持 | 支持 |
| 主动验证 | 不支持 | 不支持 | 支持 |

方案 C 包含 **7 个 tools**：

- **Layer 1: Entity Creation** — `create_signal`, `create_route`, `create_plugin`, `create_backend`, `set_global`
- **Layer 2: Discovery** — `lookup_type_info`（按需获取类型字段定义）
- **Layer 3: Modify** — `modify_entity`（增量编辑已有配置）

### 3.2 完整 Tool JSON Schema

#### Tool 1: `create_signal`

```jsonc
{
  "type": "function",
  "function": {
    "name": "create_signal",
    "description": "Declare a SIGNAL in the DSL config. Every signal referenced in a route must be declared.",
    "parameters": {
      "type": "object",
      "properties": {
        "signal_type": {
          "type": "string",
          "enum": ["keyword", "embedding", "domain", "fact_check", "user_feedback",
                   "preference", "language", "context", "complexity", "modality",
                   "authz", "jailbreak", "pii"],
          "description": "Signal type. NOTE: jailbreak and pii are signal types, NOT plugin types."
        },
        "name": {
          "type": "string",
          "description": "Unique signal name in snake_case (e.g., 'math', 'urgent_request')"
        },
        "fields": {
          "type": "object",
          "description": "Type-specific fields. Use lookup_type_info to check available fields.",
          "additionalProperties": true
        }
      },
      "required": ["signal_type", "name", "fields"]
    }
  }
}
```

**关键设计**：`signal_type` 使用 `enum` 硬约束 13 种合法值。LLM 的 tool calling 引擎会在 token 采样阶段强制只输出 enum 中的值 → **彻底消灭 `plugin_type: "pii"` 类的幻觉**。

#### Tool 2: `create_route`

```jsonc
{
  "type": "function",
  "function": {
    "name": "create_route",
    "description": "Declare a ROUTE that matches signals and dispatches to models.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": { 
          "type": "string", 
          "description": "Route name in snake_case" 
        },
        "description": { 
          "type": "string", 
          "description": "Human-readable route description" 
        },
        "priority": { 
          "type": "integer", 
          "description": "Higher = matched first. Default 10. Safety routes should use 100.",
          "default": 10 
        },
        "condition": {
          "type": "string",
          "description": "Boolean expression over signals. Format: SIGNAL_TYPE(NAME) with AND/OR/NOT operators. Examples: 'domain(math)', 'jailbreak(detect) OR pii(detect)', 'NOT jailbreak(detect) AND domain(coding)', '(domain(math) OR domain(physics)) AND complexity(hard)'"
        },
        "models": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "model": { "type": "string", "description": "Model name (e.g., 'gpt-4o', 'qwen3:70b')" },
              "reasoning": { "type": "boolean" },
              "effort": { "type": "string", "enum": ["low", "medium", "high"] },
              "weight": { "type": "number", "description": "Weight for weighted routing" }
            },
            "required": ["model"]
          },
          "minItems": 1
        },
        "algorithm": {
          "type": "object",
          "properties": {
            "algo_type": {
              "type": "string",
              "enum": ["confidence", "ratings", "remom", "static", "elo", "router_dc",
                       "automix", "hybrid", "rl_driven", "gmtrouter", "latency_aware",
                       "knn", "kmeans", "svm"]
            },
            "params": { "type": "object", "additionalProperties": true }
          },
          "required": ["algo_type"]
        },
        "plugins": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": { "type": "string", "description": "Plugin template name" },
              "overrides": { "type": "object", "additionalProperties": true }
            },
            "required": ["name"]
          }
        }
      },
      "required": ["name", "condition", "models"]
    }
  }
}
```

**关键设计**：

1. **`condition` 使用字符串表达式**而非递归 JSON Schema。原因：
   - 递归 `$ref` / `oneOf` 在 DeepSeek/Qwen 等 LLM 中支持不完整
   - 字符串表达式 `"domain(math) AND complexity(hard)"` 对 LLM 更自然
   - 前端解析成 `ConditionNode` 树的逻辑不复杂（见 §4.3）

2. **`algo_type` 使用 `enum`** 硬约束 14 种合法值

#### Tool 3: `create_plugin`

```jsonc
{
  "type": "function",
  "function": {
    "name": "create_plugin",
    "description": "Declare a PLUGIN template. Referenced by routes via plugin name.",
    "parameters": {
      "type": "object",
      "properties": {
        "plugin_type": {
          "type": "string",
          "enum": ["semantic_cache", "memory", "system_prompt", "header_mutation",
                   "hallucination", "router_replay", "rag", "image_gen", "fast_response"],
          "description": "Plugin type. Do NOT use 'jailbreak' or 'pii' here — those are signal types."
        },
        "name": { 
          "type": "string", 
          "description": "Plugin template name (referenced in routes)" 
        },
        "fields": { 
          "type": "object", 
          "additionalProperties": true 
        }
      },
      "required": ["plugin_type", "name", "fields"]
    }
  }
}
```

**关键设计**：`plugin_type` 的 `enum` 中不含 `"jailbreak"` 和 `"pii"` → **不可能输出 `plugin_type: "pii"`**。

#### Tool 4: `create_backend`

```jsonc
{
  "type": "function",
  "function": {
    "name": "create_backend",
    "description": "Declare a BACKEND (inference endpoint, API provider, vector store, etc.)",
    "parameters": {
      "type": "object",
      "properties": {
        "backend_type": {
          "type": "string",
          "enum": ["vllm_endpoint", "provider_profile", "embedding_model",
                   "semantic_cache", "memory", "response_api", "vector_store",
                   "image_gen_backend"]
        },
        "name": { "type": "string" },
        "fields": { "type": "object", "additionalProperties": true }
      },
      "required": ["backend_type", "name", "fields"]
    }
  }
}
```

#### Tool 5: `set_global`

```jsonc
{
  "type": "function",
  "function": {
    "name": "set_global",
    "description": "Set GLOBAL configuration (default_model, strategy, etc.)",
    "parameters": {
      "type": "object",
      "properties": {
        "default_model": { "type": "string", "description": "Default model name" },
        "strategy": { 
          "type": "string", 
          "enum": ["priority", "weighted", "round_robin"],
          "description": "Global routing strategy"
        }
      }
    }
  }
}
```

#### Tool 6: `lookup_type_info`（按需发现）

```jsonc
{
  "type": "function",
  "function": {
    "name": "lookup_type_info",
    "description": "Look up field schema and constraints for a specific DSL type. Call this when unsure about required/optional fields for a type.",
    "parameters": {
      "type": "object",
      "properties": {
        "construct": { 
          "type": "string", 
          "enum": ["signal", "plugin", "algorithm", "backend"] 
        },
        "type_name": { "type": "string" }
      },
      "required": ["construct", "type_name"]
    }
  }
}
```

**执行逻辑**：从 `NLSchemaRegistry` 查找对应条目，返回字段定义：

```typescript
// 示例返回
{
  "type_name": "keyword",
  "construct": "signal",
  "description": "Matches queries containing specific keywords or patterns",
  "fields": {
    "required": ["keywords"],
    "optional": {
      "operator": { "type": "string", "options": ["any", "all"], "default": "any" },
      "method": { "type": "string", "options": ["regex", "bm25", "ngram"], "default": "regex" },
      "threshold": { "type": "number", "range": [0, 1], "default": 0.5 },
      "case_sensitive": { "type": "boolean", "default": false }
    }
  }
}
```

#### Tool 7: `modify_entity`（增量编辑）

```jsonc
{
  "type": "function",
  "function": {
    "name": "modify_entity",
    "description": "Modify an existing entity in the current DSL configuration. Use for 'modify' mode.",
    "parameters": {
      "type": "object",
      "properties": {
        "action": { 
          "type": "string", 
          "enum": ["add", "update", "delete"] 
        },
        "target_construct": { 
          "type": "string", 
          "enum": ["signal", "route", "plugin", "backend", "global"] 
        },
        "target_name": { "type": "string", "description": "Name of the entity to modify" },
        "changes": { 
          "type": "object", 
          "additionalProperties": true,
          "description": "Fields to change"
        }
      },
      "required": ["action", "target_construct", "target_name"]
    }
  }
}
```

### 3.3 `fields` 约束策略

`create_signal` / `create_plugin` 的 `fields` 参数使用 `additionalProperties: true`，没有约束内部字段值（如 keyword 的 `method: "regex" | "bm25" | "ngram"`）。

三种解法比较：

| 方案 | 做法 | 优点 | 缺点 |
|:---|:---|:---|:---|
| A: 独立 tool | 每种类型一个 tool（`create_keyword_signal` 等） | 精确约束 | **44 个 tools**，LLM 选择困难 |
| B: 先查后填 | 必须先 `lookup_type_info` 再创建 | 完整约束 | 多一轮调用 |
| **C: 混合（推荐）** | 顶层 enum 硬约束，fields 灵活 + WASM 验证兜底 | 平衡 | fields 内部值可能错误 |

**选择方案 C**，因为：
- 核心幻觉（`plugin_type: "pii"`）已被顶层 `enum` 消灭
- `fields` 内部的枚举值错误（如 `method: "neural"` 而非 `"classifier"`）相对少见
- 错误可被 WASM 验证捕获并通过 `validate_config` tool 自动修复
- 44 个 tool 会导致 tool description 本身占用大量 tokens

---

## 4. 执行流程

### 4.1 Collect Mode（区别于 Interactive Mode）

项目已有的 `ToolRegistry`（`tools/registry.ts`）支持 **Interactive Mode**：tool result 回传 LLM → LLM 继续对话。

NL DSL tools 需要新的 **Collect Mode**：

```
Interactive Mode (现有):
  LLM → tool_call → execute → result → LLM → tool_call → execute → ...

Collect Mode (NL DSL):
  LLM → [parallel tool_calls] → 全部收集 → 组装 IntentIR → 验证
                                                                 ↓
                                            如有错误 → 回传 LLM 修正
                                            验证通过 → 完成
```

### 4.2 完整执行流程

```
User: "数学问题路由到 GPT-4o，代码问题路由到 DeepSeek，检测越狱并拦截"

┌─────────────────────────────────────────────────────────────┐
│ Round 1: LLM 理解意图 → 发出 parallel tool calls            │
│                                                              │
│  create_signal("domain", "math", {description: "Math", ...})│
│  create_signal("domain", "coding", {description: "Code"...})│
│  create_signal("jailbreak", "jailbreak_detect", {method:    │
│    "classifier", threshold: 0.9})                            │
│  create_plugin("fast_response", "block_jailbreak",          │
│    {message: "Blocked."})                                    │
│  create_route("safety_route", ..., priority: 100,           │
│    condition: "jailbreak(jailbreak_detect)")                 │
│  create_route("math_route", ...,                            │
│    condition: "domain(math)",                                │
│    models: [{model: "gpt-4o"}])                              │
│  create_route("coding_route", ...,                          │
│    condition: "domain(coding)",                              │
│    models: [{model: "deepseek-v3"}])                         │
│  create_backend("provider_profile", "openai", {...})        │
│  set_global({default_model: "gpt-4o-mini", strategy:        │
│    "priority"})                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼  前端执行所有 tool calls
┌──────────────────────────────────────────────────────────────┐
│ Tool Executor: 收集 → 组装 IntentIR → 转 DSL → WASM 验证    │
│                                                              │
│  1. 每个 create_* call → 转为一个 Intent                    │
│  2. 解析 condition 字符串 → ConditionNode 树                 │
│  3. intentIRToDSL(ir) → DSL text                             │
│  4. WASM validate(dsl) → diagnostics                         │
│                                                              │
│  验证通过? → 完成，返回 DSL + YAML                          │
│  验证失败? → 进入 Round 2                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ (only if errors)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Round 2: 回传验证结果给 LLM                                  │
│                                                              │
│  Tool results: [                                             │
│    { name: "create_signal", content: "OK" },                 │
│    ...                                                        │
│    { name: "validate_result", content: {                     │
│      is_valid: false,                                        │
│      diagnostics: ["Line 5: unknown field 'method'..."]      │
│    }}                                                        │
│  ]                                                           │
│                                                              │
│  LLM 看到错误 → lookup_type_info("signal", "jailbreak")     │
│              → 发出修正的 create_signal/modify_entity call    │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 Condition 表达式解析

`create_route` 的 `condition` 参数是字符串表达式而非递归 JSON。前端需要解析为 `ConditionNode` 树：

```typescript
// lib/nlConditionParser.ts

/**
 * Parse a condition expression string into a ConditionNode tree.
 * Grammar:
 *   expr     = or_expr
 *   or_expr  = and_expr ("OR" and_expr)*
 *   and_expr = not_expr ("AND" not_expr)*
 *   not_expr = "NOT" not_expr | primary
 *   primary  = "(" expr ")" | SIGNAL_TYPE "(" SIGNAL_NAME ")"
 *
 * Examples:
 *   "domain(math)"
 *   "domain(math) AND complexity(hard)"
 *   "NOT jailbreak(detect) AND (domain(math) OR domain(coding))"
 */
export function parseConditionExpr(expr: string): ConditionNode {
  const tokens = tokenize(expr)
  let pos = 0

  function parseOrExpr(): ConditionNode {
    let left = parseAndExpr()
    while (pos < tokens.length && tokens[pos] === 'OR') {
      pos++ // consume OR
      const right = parseAndExpr()
      left = { op: 'OR', operands: [left, right] }
    }
    return left
  }

  function parseAndExpr(): ConditionNode {
    let left = parseNotExpr()
    while (pos < tokens.length && tokens[pos] === 'AND') {
      pos++ // consume AND
      const right = parseNotExpr()
      left = { op: 'AND', operands: [left, right] }
    }
    return left
  }

  function parseNotExpr(): ConditionNode {
    if (pos < tokens.length && tokens[pos] === 'NOT') {
      pos++ // consume NOT
      const operand = parseNotExpr()
      return { op: 'NOT', operand }
    }
    return parsePrimary()
  }

  function parsePrimary(): ConditionNode {
    if (tokens[pos] === '(') {
      pos++ // consume (
      const node = parseOrExpr()
      pos++ // consume )
      return node
    }
    // SIGNAL_TYPE(SIGNAL_NAME)
    const signalType = tokens[pos++]
    pos++ // consume (
    const signalName = tokens[pos++]
    pos++ // consume )
    return { op: 'SIGNAL_REF', signal_type: signalType, signal_name: signalName }
  }

  return parseOrExpr()
}
```

### 4.4 Tool Call 到 IntentIR 的转换

```typescript
// lib/nlToolExecutor.ts

interface CollectedToolCalls {
  signals: Array<{ signal_type: string; name: string; fields: Record<string, unknown> }>
  routes: Array<{ name: string; description?: string; priority?: number; 
                  condition: string; models: ModelIntent[]; 
                  algorithm?: AlgorithmIntent; plugins?: PluginRefIntent[] }>
  plugins: Array<{ plugin_type: string; name: string; fields: Record<string, unknown> }>
  backends: Array<{ backend_type: string; name: string; fields: Record<string, unknown> }>
  global?: { default_model?: string; strategy?: string }
  modifications: Array<{ action: string; target_construct: string; 
                          target_name: string; changes?: Record<string, unknown> }>
}

function assembleIntentIR(calls: CollectedToolCalls, mode: 'generate' | 'modify'): IntentIR {
  const intents: Intent[] = []

  // Convert signals
  for (const sig of calls.signals) {
    intents.push({ type: 'signal', ...sig })
  }

  // Convert plugins
  for (const plugin of calls.plugins) {
    intents.push({ type: 'plugin_template', ...plugin })
  }

  // Convert routes (parse condition string → ConditionNode)
  for (const route of calls.routes) {
    intents.push({
      type: 'route',
      name: route.name,
      description: route.description,
      priority: route.priority,
      condition: parseConditionExpr(route.condition),
      models: route.models,
      algorithm: route.algorithm,
      plugins: route.plugins,
    })
  }

  // Convert backends
  for (const be of calls.backends) {
    intents.push({ type: 'backend', ...be })
  }

  // Convert global
  if (calls.global) {
    intents.push({ type: 'global', fields: calls.global })
  }

  // Convert modifications
  for (const mod of calls.modifications) {
    intents.push({ type: 'modify', ...mod } as ModifyIntent)
  }

  return { version: '1.0', operation: mode, intents }
}
```

---

## 5. System Prompt 设计

### 5.1 Tool Calling 模式（~800 tokens）

```
You are a Signal DSL routing configuration assistant.
Given the user's natural language description, create the routing configuration 
by calling the provided tools.

Key rules:
1. Every signal referenced in a route condition must be declared via create_signal.
2. Use create_plugin for plugins, then reference them by name in create_route.
3. Always set a GLOBAL config with default_model and strategy.
4. Use lookup_type_info if unsure about field names for a specific type.
5. Priority: higher number = matched first. Safety routes should use 100.
6. jailbreak and pii are SIGNAL types — never use them as plugin_type.
7. Condition syntax: SIGNAL_TYPE(NAME) with AND/OR/NOT. 
   Use parentheses for grouping: (domain(math) OR domain(physics)) AND complexity(hard)

Common patterns:
- Safety guard: create_signal("jailbreak") + create_plugin("fast_response") + high-priority route
- Domain routing: create_signal("domain") per domain + one route per domain
- Semantic matching: create_signal("embedding") with candidates list
- Complexity tiers: create_signal("complexity") per tier + priority-ordered routes
- Weighted models: multiple models in one route with weight field
```

对比：

| 维度 | 当前 Structured Output | Tool Calling |
|:---|:---|:---|
| System prompt | ~5000 tokens | ~800 tokens |
| 类型定义位置 | 自然语言描述在 prompt 中 | JSON Schema `enum` 在 tools 中 |
| 字段约束 | 自然语言 rules | `enum` + `required` + `additionalProperties` |
| Few-shot examples | ~3000 tokens 在 user prompt | 不需要（tool schema 本身就是约束） |

### 5.2 Modify 模式附加 Context

当 mode = 'modify' 时，在 system prompt 末尾追加：

```
Current configuration context:
- Defined signals: domain("math"), domain("coding"), jailbreak("jailbreak_detect")
- Defined routes: math_route, coding_route, safety_route
- Defined plugins: block_jailbreak
- Available models: gpt-4o, deepseek-v3

Use modify_entity tool to make changes to existing entities.
Use create_signal / create_plugin / create_route to add new entities.
```

---

## 6. enum 动态生成

Tool schema 中的 `enum` 值不应硬编码，而是从 `NLSchemaRegistry` 动态生成：

```typescript
// lib/nlToolDefinitions.ts

import { defaultRegistry } from './nlSchemaRegistry'
import type { ToolDefinition } from '../tools/types'

/**
 * Generate tool definitions with enum values dynamically pulled from the schema registry.
 * When new DSL types are registered, tool schemas automatically update.
 */
export function generateNLToolDefinitions(
  registry = defaultRegistry
): ToolDefinition[] {
  const signalTypes = registry.getByConstruct('signal').map(e => e.type_name)
  const pluginTypes = registry.getByConstruct('plugin').map(e => e.type_name)
  const algoTypes = registry.getByConstruct('algorithm').map(e => e.type_name)
  const backendTypes = registry.getByConstruct('backend').map(e => e.type_name)

  return [
    {
      type: 'function',
      function: {
        name: 'create_signal',
        description: 'Declare a SIGNAL. Every signal referenced in a route must be declared.',
        parameters: {
          type: 'object',
          properties: {
            signal_type: {
              type: 'string',
              enum: signalTypes,  // ← 从 registry 动态获取
              description: 'Signal type. jailbreak and pii are signal types, NOT plugin types.',
            },
            name: { type: 'string', description: 'Signal name in snake_case' },
            fields: { type: 'object', description: 'Type-specific fields', additionalProperties: true },
          },
          required: ['signal_type', 'name', 'fields'],
        },
      },
    },
    // ... create_plugin (enum: pluginTypes), create_route (algo_type enum: algoTypes), etc.
  ]
}
```

**好处**：保持 `NLSchemaRegistry` 作为唯一数据源。新增 DSL 类型只需在 registry 中注册，tool schema 自动包含。

---

## 7. `lookup_type_info` 执行实现

```typescript
// lib/nlToolExecutor.ts

function executeLookupTypeInfo(
  args: { construct: string; type_name: string },
  registry: NLSchemaRegistry = defaultRegistry
): object {
  const entry = registry.get(args.construct, args.type_name)
  if (!entry) {
    return { error: `Unknown ${args.construct} type: "${args.type_name}". Check valid types.` }
  }

  const required = entry.fields.filter(f => f.required)
  const optional = entry.fields.filter(f => !f.required)

  return {
    type_name: entry.type_name,
    construct: entry.construct,
    description: entry.nl_description,
    fields: {
      required: required.map(f => ({
        name: f.key,
        type: f.type,
        options: f.options ?? undefined,
        placeholder: f.placeholder ?? undefined,
      })),
      optional: optional.map(f => ({
        name: f.key,
        type: f.type,
        options: f.options ?? undefined,
        default: f.default ?? undefined,
      })),
    },
    requires_backend: entry.requires_backend ?? [],
    requires_signal: entry.requires_signal ?? [],
  }
}
```

**LLM 使用场景**：

```
LLM 思考: 用户说"检测越狱" → 我需要 jailbreak signal → 字段有哪些？
LLM: tool_call lookup_type_info("signal", "jailbreak")
→ 返回: { fields: { required: [{name: "method", options: ["classifier", "contrastive"]}], 
                     optional: [{name: "threshold", type: "number", default: 0.9}] } }
LLM: tool_call create_signal("jailbreak", "jailbreak_detect", {method: "classifier", threshold: 0.9})
```

---

## 8. LLM Client 改造

### 8.1 接口扩展

当前 `LLMClient` 接口：

```typescript
interface LLMClient {
  generateIntentIR(systemPrompt: string, userPrompt: string): Promise<IntentIR>
}
```

扩展为：

```typescript
interface LLMClient {
  /** 现有方案：Structured Output */
  generateIntentIR(systemPrompt: string, userPrompt: string): Promise<IntentIR>
  
  /** Tool Calling 方案 */
  generateWithTools?(
    systemPrompt: string, 
    userPrompt: string, 
    tools: ToolDefinition[],
    toolExecutor: (name: string, args: unknown) => Promise<unknown>,
    options?: { maxRounds?: number }
  ): Promise<IntentIR>

  /** 能力检测 */
  supportsToolCalling?(): boolean
}
```

### 8.2 Tool Calling Client 实现

```typescript
// lib/nlToolCallingClient.ts

export class ToolCallingLLMClient implements LLMClient {
  private endpoint: string
  private apiKey: string
  private model: string

  async generateWithTools(
    systemPrompt: string,
    userPrompt: string,
    tools: ToolDefinition[],
    toolExecutor: (name: string, args: unknown) => Promise<unknown>,
    options: { maxRounds?: number } = {}
  ): Promise<IntentIR> {
    const maxRounds = options.maxRounds ?? 3
    const messages: ChatMessage[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ]

    const collected: CollectedToolCalls = {
      signals: [], routes: [], plugins: [], backends: [],
      global: undefined, modifications: [],
    }

    for (let round = 0; round < maxRounds; round++) {
      const response = await this.callLLM(messages, tools)

      // If LLM returns a text response (no tool calls), done
      if (!response.tool_calls || response.tool_calls.length === 0) {
        break
      }

      // Process all tool calls
      const toolResults: ToolResult[] = []
      for (const call of response.tool_calls) {
        const args = JSON.parse(call.function.arguments)
        
        // Collect into appropriate bucket
        this.collectToolCall(call.function.name, args, collected)
        
        // Execute tool (for lookup_type_info, this returns actual data)
        const result = await toolExecutor(call.function.name, args)
        toolResults.push({
          callId: call.id,
          name: call.function.name,
          content: result,
        })
      }

      // Append assistant message + tool results
      messages.push({ role: 'assistant', tool_calls: response.tool_calls })
      for (const result of toolResults) {
        messages.push({
          role: 'tool',
          tool_call_id: result.callId,
          content: JSON.stringify(result.content),
        })
      }

      // If no lookup_type_info calls in this round, we're done
      const hasLookup = response.tool_calls.some(
        c => c.function.name === 'lookup_type_info'
      )
      if (!hasLookup) break
    }

    // Assemble IntentIR from collected tool calls
    return assembleIntentIR(collected, 'generate')
  }

  private collectToolCall(name: string, args: unknown, collected: CollectedToolCalls): void {
    switch (name) {
      case 'create_signal':
        collected.signals.push(args as any)
        break
      case 'create_route':
        collected.routes.push(args as any)
        break
      case 'create_plugin':
        collected.plugins.push(args as any)
        break
      case 'create_backend':
        collected.backends.push(args as any)
        break
      case 'set_global':
        collected.global = args as any
        break
      case 'modify_entity':
        collected.modifications.push(args as any)
        break
      // lookup_type_info: not collected, result returned to LLM
    }
  }

  supportsToolCalling(): boolean {
    return true
  }
}
```

---

## 9. Pipeline 集成

### 9.1 双模式分支

```typescript
// lib/nlPipeline.ts — 修改 generateDSL 函数

async function generateDSL(
  nlInput: string,
  context: NLPromptContext,
  wasm: WasmBridge,
  llmClient: LLMClient,
  registry: NLSchemaRegistry = defaultRegistry,
): Promise<NLGenerateResult> {

  // 选择模式
  if (llmClient.supportsToolCalling?.() && llmClient.generateWithTools) {
    return generateDSLWithToolCalling(nlInput, context, wasm, llmClient, registry)
  }
  
  // Fallback: 现有 Structured Output 方案
  return generateDSLWithStructuredOutput(nlInput, context, wasm, llmClient, registry)
}
```

### 9.2 Tool Calling 模式流程

```typescript
async function generateDSLWithToolCalling(
  nlInput: string,
  context: NLPromptContext,
  wasm: WasmBridge,
  llmClient: LLMClient,
  registry: NLSchemaRegistry,
): Promise<NLGenerateResult> {
  
  // 1. 构建精简 system prompt
  const systemPrompt = buildToolCallingSystemPrompt(context)
  
  // 2. 生成 tool definitions
  const tools = generateNLToolDefinitions(registry)
  
  // 3. Tool executor
  const toolExecutor = (name: string, args: unknown) => {
    if (name === 'lookup_type_info') {
      return Promise.resolve(executeLookupTypeInfo(args as any, registry))
    }
    // 其他 tool: 返回 "OK"（收集模式，不需要实际执行结果）
    return Promise.resolve({ status: 'ok' })
  }
  
  // 4. LLM + Tool Calling → IntentIR
  const intentIR = await llmClient.generateWithTools!(
    systemPrompt, nlInput, tools, toolExecutor, { maxRounds: 3 }
  )
  
  // 5. IntentIR → DSL (复用现有 codegen)
  let dsl = intentIRToDSL(intentIR, context.mode === 'modify' ? context.currentDSL : undefined)
  
  // 6. WASM 验证
  let validation = await validateGeneratedDSL(dsl, wasm)
  
  // 7. 如果验证失败，走现有 repair loop（保持兼容）
  let attempt = 0
  while (!validation.isValid && attempt < 3) {
    attempt++
    dsl = await repairDSL(nlInput, dsl, validation.diagnostics, wasm, llmClient, attempt)
    validation = await validateGeneratedDSL(dsl, wasm)
  }
  
  return {
    dsl,
    yaml: validation.yaml,
    diagnostics: validation.diagnostics,
    isValid: validation.isValid,
    intentIR,
    confidence: computeConfidence(validation, attempt),
    retries: attempt,
    mode: 'tool_calling',
  }
}
```

---

## 10. Parallel Tool Calling 兼容性

### 10.1 LLM 支持情况

| LLM | Parallel Tool Calls | Tool Calling |
|:---|:---:|:---:|
| OpenAI GPT-4o / GPT-4o-mini | ✅ | ✅ |
| Anthropic Claude 3.5/4 | ✅ | ✅ |
| DeepSeek-V3 API | ✅ | ✅ |
| Qwen (DashScope) | ✅ | ✅ |
| Google Gemini | ✅ | ✅ |
| vLLM + Hermes/Qwen | ⚠️ 取决于模型和 `--tool-call-parser` | ⚠️ |
| Ollama | ⚠️ 部分模型支持 | ⚠️ |

### 10.2 Fallback 策略

```
检测 LLM 能力
    │
    ├── supports tool_calling AND parallel_tool_calls?
    │       → Tool Calling 模式
    │       → System prompt ~800 tokens
    │       → 7 个 tools
    │       → 1-2 轮调用
    │
    └── 不支持?
            → Structured Output 模式（现有方案）
            → System prompt ~5000 tokens
            → response_format: json_object
            → 1 轮 + repair loop
```

### 10.3 能力检测方式

```typescript
function supportsToolCalling(modelName: string): boolean {
  const toolCapablePatterns = [
    /gpt-4/i, /gpt-3\.5-turbo/i,      // OpenAI
    /claude/i,                           // Anthropic
    /deepseek/i,                         // DeepSeek
    /qwen/i,                             // Qwen
    /gemini/i,                           // Google
  ]
  return toolCapablePatterns.some(p => p.test(modelName))
}
```

更稳健的方式：在第一次调用时尝试带 `tools` 参数的请求。如果 API 返回 `400`（参数不支持），标记该模型不支持 tool calling 并 fallback。

---

## 11. 与现有 Tool Registry 的融合

### 11.1 Category 扩展

在 `tools/types.ts` 的 `ToolMetadata.category` 中新增 `'dsl'` 分类：

```typescript
export interface ToolMetadata {
  category: 'search' | 'code' | 'file' | 'image' | 'custom' | 'dsl'  // ← 新增
  // ...
}
```

### 11.2 注册 NL Tools

```typescript
// tools/nlDslTools.ts

import { toolRegistry } from './registry'
import { generateNLToolDefinitions } from '../lib/nlToolDefinitions'

export function registerNLDslTools(): void {
  const tools = generateNLToolDefinitions()
  
  for (const def of tools) {
    toolRegistry.register({
      metadata: {
        id: def.function.name,
        displayName: def.function.name.replace(/_/g, ' '),
        category: 'dsl',
        enabled: true,
        version: '1.0.0',
      },
      definition: def,
      executor: async (args) => {
        // NL DSL tools 在 collect mode 下不真正执行
        // 只有 lookup_type_info 需要返回实际数据
        if (def.function.name === 'lookup_type_info') {
          return executeLookupTypeInfo(args as any)
        }
        return { status: 'ok' }
      },
    })
  }
}
```

### 11.3 获取 Tool Definitions

```typescript
// In pipeline
const nlTools = toolRegistry.getByCategory('dsl').map(t => t.definition)
```

---

## 12. 收益量化预期

| 指标 | 当前 Structured Output | Tool Calling | 改善 |
|:---|:---|:---|:---|
| 类型幻觉率 | ~15-20%（pii/vector_store 等） | ~0%（enum 硬约束） | **消除** |
| System prompt tokens | ~5000 | ~800 | **-84%** |
| Tool schema tokens | 0 | ~1500（由 API 层承载） | 不计入 prompt |
| 字段值错误率 | ~10% | ~5%（fields 仍灵活） | -50% |
| 首次验证通过率 | ~60-70% | ~85-95% | +20-30% |
| 平均 LLM 轮次 | 1.5-2（含 repair） | 1.1-1.3 | -30% |
| 模型兼容性 | 100% | ~80%（需 fallback） | 需双模式 |

---

## 13. 实施路线

### Phase 0: 当前 prompt 优化（前置工作）

先完成对现有 Structured Output 方案的 7 项优化（作为 fallback 的质量基线）：

| 优化项 | 工作量 |
|:---|:---|
| Few-shot 移入 System Prompt | 小 |
| 添加 Negative Examples | 小 |
| Rules 分级（CRITICAL/IMPORTANT） | 小 |
| 精简 Intent IR Schema | 小 |
| Repair prompt 引用类型约束 | 小 |

### Phase 1: Tool Schema 定义

| 模块 | 文件 | 工作量 |
|:---|:---|:---|
| Tool definitions 生成器 | `lib/nlToolDefinitions.ts` | 中 |
| Condition 表达式解析器 | `lib/nlConditionParser.ts` | 中 |
| Tool call → IntentIR 转换 | `lib/nlToolExecutor.ts` | 中 |
| 单元测试 | `lib/nlToolDefinitions.test.ts`, `lib/nlConditionParser.test.ts` | 中 |

### Phase 2: Tool Calling Client

| 模块 | 文件 | 工作量 |
|:---|:---|:---|
| ToolCallingLLMClient 实现 | `lib/nlToolCallingClient.ts` | 高 |
| Tool Calling system prompt | `lib/nlPromptBuilder.ts`（新增函数） | 小 |
| LLMClient 接口扩展 | 修改现有接口 | 小 |

### Phase 3: Pipeline 集成

| 模块 | 文件 | 工作量 |
|:---|:---|:---|
| Pipeline 双模式分支 | `lib/nlPipeline.ts` | 中 |
| 能力检测逻辑 | `stores/nlStore.ts` | 小 |
| Tool Registry 融合 | `tools/nlDslTools.ts` | 小 |
| E2E 测试 | `lib/nlPipeline.test.ts` | 中 |

### Phase 4: LLM 自主验证

| 模块 | 文件 | 工作量 |
|:---|:---|:---|
| `validate_config` tool 实现 | `lib/nlToolExecutor.ts`（扩展） | 小 |
| 验证结果回传 LLM 逻辑 | `lib/nlToolCallingClient.ts`（扩展） | 中 |
| 替代硬编码 repair loop | `lib/nlPipeline.ts`（重构） | 中 |

**总体评估：约 2-3 天开发量，风险可控。**

---

## 14. 文件交付物

```
dashboard/frontend/src/
├── lib/
│   ├── nlToolDefinitions.ts          # 7 个 tool 的 JSON Schema 生成器（从 registry 动态生成 enum）
│   ├── nlToolDefinitions.test.ts     # Tool schema 生成测试
│   ├── nlConditionParser.ts          # 条件表达式字符串 → ConditionNode 解析器
│   ├── nlConditionParser.test.ts     # 解析器测试（包含复杂嵌套 AND/OR/NOT 用例）
│   ├── nlToolExecutor.ts             # Tool call 收集器 + IntentIR 组装 + lookup_type_info 执行
│   ├── nlToolCallingClient.ts        # Tool Calling LLM Client（multi-round tool call loop）
│   ├── nlPromptBuilder.ts            # 扩展：新增 buildToolCallingSystemPrompt()
│   └── nlPipeline.ts                 # 扩展：双模式分支（tool calling / structured output）
├── tools/
│   └── nlDslTools.ts                 # NL DSL tools 注册到现有 ToolRegistry
│   └── types.ts                      # 扩展：ToolMetadata.category 新增 'dsl'
└── stores/
    └── nlStore.ts                    # 扩展：createLLMClient 支持两种模式
```

---

## 15. 关键设计决策总结

| 决策 | 选择 | 理由 |
|:---|:---|:---|
| Tool 数量 | 7 个（5 create + 1 lookup + 1 modify） | 不要过细拆分，避免 LLM 选择困难 |
| `condition` 格式 | 字符串表达式而非递归 JSON Schema | 兼容性好，LLM 更容易生成正确结果 |
| `fields` 约束 | `additionalProperties: true` + WASM 兜底 | 避免 44 个独立 tool 的方案 |
| 执行模式 | Collect Mode（收集后批量组装） | 区别于现有 Interactive Mode |
| enum 数据源 | 从 NLSchemaRegistry 动态生成 | 保持单一数据源 |
| fallback 策略 | Tool calling 优先，Structured Output 兜底 | 保持 100% 模型兼容性 |
| Repair 机制 | Tool calling 模式仍复用现有 repair loop | 降低实施风险 |
