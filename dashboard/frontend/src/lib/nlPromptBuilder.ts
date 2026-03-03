/**
 * NL → DSL Prompt Builder
 *
 * Constructs the LLM system prompt and user prompt for Intent IR generation.
 * Dynamically pulls type information from the NL Schema Registry so that
 * when new DSL types are registered, the prompt automatically includes them.
 *
 * The LLM outputs a structured Intent IR JSON (not raw DSL text), which is
 * then deterministically converted to DSL by intentToDsl.ts.
 */

import { defaultRegistry, type NLSchemaRegistry } from './nlSchemaRegistry'

// ─────────────────────────────────────────────
// System Prompt
// ─────────────────────────────────────────────

/**
 * Build the full system prompt for Intent IR generation.
 * Registry is injected for testability; defaults to the singleton.
 */
export function buildSystemPrompt(registry: NLSchemaRegistry = defaultRegistry): string {
  const typeSection = registry.buildSystemPromptSection()
  return `${SYSTEM_PROMPT_HEADER}

## DSL Type System (complete reference)

${typeSection}

## Intent IR JSON Schema

${INTENT_IR_SCHEMA}

## Rules

${RULES_SECTION}`.trim()
}

// ─────────────────────────────────────────────
// User Prompt
// ─────────────────────────────────────────────

export interface NLPromptContext {
  /** User's natural language input */
  userInput: string
  /** Operation mode */
  mode: 'generate' | 'modify'
  /** Current symbol table (for modify mode context) */
  symbols?: {
    signals: Array<{ name: string; type: string }>
    routes: string[]
    plugins: string[]
    backends: Array<{ name: string; type: string }>
    models: string[]
  }
  /** Current DSL text (for modify mode) */
  currentDSL?: string
  /** Diagnostics from current compilation (for fix/context) */
  diagnostics?: Array<{ level: string; message: string }>
  /** Pre-extracted suggested types from NL Schema Registry trigger matching */
  suggestedTypes?: Array<{ construct: string; type_name: string }>
}

/**
 * Build the user prompt that includes few-shot examples and context.
 */
export function buildUserPrompt(context: NLPromptContext): string {
  const parts: string[] = []

  // Few-shot examples
  parts.push(FEW_SHOT_EXAMPLES)
  parts.push('')

  // Pre-extracted type hints (from trigger matching)
  if (context.suggestedTypes && context.suggestedTypes.length > 0) {
    const grouped: Record<string, string[]> = {}
    for (const st of context.suggestedTypes) {
      if (!grouped[st.construct]) grouped[st.construct] = []
      grouped[st.construct].push(st.type_name)
    }
    parts.push('Detected relevant DSL types from user input (use these as hints):')
    for (const [construct, types] of Object.entries(grouped)) {
      parts.push(`- ${construct}: ${types.join(', ')}`)
    }
    parts.push('')
  }

  // Current configuration context (for modify mode)
  if (context.mode === 'modify' && context.symbols) {
    parts.push('Current configuration context:')
    if ((context.symbols.signals ?? []).length > 0) {
      const sigList = context.symbols.signals
        .map(s => `${s.type}("${s.name}")`)
        .join(', ')
      parts.push(`- Defined signals: ${sigList}`)
    }
    if ((context.symbols.routes ?? []).length > 0) {
      parts.push(`- Defined routes: ${context.symbols.routes.join(', ')}`)
    }
    if ((context.symbols.plugins ?? []).length > 0) {
      parts.push(`- Defined plugins: ${context.symbols.plugins.join(', ')}`)
    }
    if ((context.symbols.backends ?? []).length > 0) {
      const beList = context.symbols.backends
        .map(b => `${b.type}("${b.name}")`)
        .join(', ')
      parts.push(`- Defined backends: ${beList}`)
    }
    if ((context.symbols.models ?? []).length > 0) {
      parts.push(`- Available models: ${context.symbols.models.join(', ')}`)
    }
    parts.push('')
  }

  // Current errors
  if (context.diagnostics && context.diagnostics.length > 0) {
    parts.push('Current compilation errors:')
    for (const d of context.diagnostics) {
      parts.push(`- [${d.level}] ${d.message}`)
    }
    parts.push('')
  }

  // The actual request
  parts.push(`User request: "${context.userInput}"`)
  parts.push(`Operation mode: ${context.mode}`)
  parts.push('')
  parts.push('Generate the Intent IR JSON:')

  return parts.join('\n')
}

// ─────────────────────────────────────────────
// Repair Prompt
// ─────────────────────────────────────────────

export interface RepairPromptContext {
  /** Original NL input */
  originalNL: string
  /** The DSL that failed validation */
  failedDSL: string
  /** Validation diagnostics */
  diagnostics: Array<{ level: string; message: string; line?: number }>
  /** Repair attempt number (2 = targeted, 3 = full regen) */
  attempt: number
  /** Operation mode from the original request */
  mode?: 'generate' | 'modify'
}

/**
 * Build repair prompt for when validation fails.
 * Attempt 2: targeted fix prompt. Attempt 3: full regeneration with error context.
 */
export function buildRepairPrompt(ctx: RepairPromptContext): string {
  if (ctx.attempt === 2) {
    return buildTargetedRepairPrompt(ctx)
  }
  return buildFullRegenerationPrompt(ctx)
}

function buildTargetedRepairPrompt(ctx: RepairPromptContext): string {
  const errorList = ctx.diagnostics
    .map((d, i) => {
      const lineInfo = d.line ? `Line ${d.line}: ` : ''
      return `${i + 1}. ${lineInfo}${d.message}`
    })
    .join('\n')

  return `The following DSL has validation errors. Fix ONLY the errors listed below.
Do not change anything else. Return the complete corrected Intent IR JSON.

Errors:
${errorList}

Current DSL:
\`\`\`
${ctx.failedDSL}
\`\`\`

Original user request: "${ctx.originalNL}"

Generate the corrected Intent IR JSON:`
}

function buildFullRegenerationPrompt(ctx: RepairPromptContext): string {
  const errorSummary = ctx.diagnostics
    .map(d => `- ${d.message}`)
    .join('\n')

  return `Previous attempt generated DSL but it had these errors:
${errorSummary}

Please regenerate the Intent IR from scratch, avoiding these mistakes.

Original user request: "${ctx.originalNL}"
Operation mode: ${ctx.mode ?? 'generate'}

Generate the Intent IR JSON:`
}

// ─────────────────────────────────────────────
// Static Content
// ─────────────────────────────────────────────

const SYSTEM_PROMPT_HEADER = `You are a Signal DSL configuration expert. Given a natural language description, generate an Intent IR (Intermediate Representation) as a JSON object that captures the user's routing configuration intent.

The Intent IR will be deterministically converted to valid DSL text by the system. You should NOT output raw DSL — only the structured JSON.`

const INTENT_IR_SCHEMA = `\`\`\`typescript
interface IntentIR {
  version: "1.0";
  operation: "generate" | "modify";
  intents: Intent[];
}

// Discriminated union — each intent has a "type" field:

interface SignalIntent {
  type: "signal";
  signal_type: string;    // One of the signal types listed above
  name: string;           // snake_case identifier
  fields: Record<string, unknown>;
}

interface RouteIntent {
  type: "route";
  name: string;
  description?: string;
  priority?: number;      // Higher = matched first
  condition: ConditionNode;
  models: ModelIntent[];
  algorithm?: { algo_type: string; params: Record<string, unknown> };
  plugins?: { name: string; overrides?: Record<string, unknown> }[];
}

interface ConditionNode {
  op: "AND" | "OR" | "NOT" | "SIGNAL_REF";
  operands?: ConditionNode[];   // For AND/OR
  operand?: ConditionNode;      // For NOT
  signal_type?: string;          // For SIGNAL_REF
  signal_name?: string;          // For SIGNAL_REF
}

interface ModelIntent {
  model: string;           // e.g. "gpt-4o", "qwen3:70b"
  reasoning?: boolean;
  effort?: "low" | "medium" | "high";
  lora?: string;
  param_size?: string;
  weight?: number;
  reasoning_family?: string;
}

interface PluginTemplateIntent {
  type: "plugin_template";
  name: string;
  plugin_type: string;
  fields: Record<string, unknown>;
}

interface BackendIntent {
  type: "backend";
  backend_type: string;
  name: string;
  fields: Record<string, unknown>;
}

interface GlobalIntent {
  type: "global";
  fields: Record<string, unknown>;
}

interface ModifyIntent {
  type: "modify";
  action: "add" | "update" | "delete";
  target_construct: "signal" | "route" | "plugin" | "backend" | "global";
  target_name: string;
  target_signal_type?: string;
  target_plugin_type?: string;
  target_backend_type?: string;
  changes?: Record<string, unknown>;
}
\`\`\``

const RULES_SECTION = `1. EVERY signal referenced in a route condition MUST have a corresponding SignalIntent.
2. Field values must respect type constraints (thresholds: 0-1, ports: 1-65535).
3. For "modify" operation, only include the changed entities using ModifyIntent.
4. Use descriptive signal names in snake_case.
5. Output ONLY valid JSON conforming to the Intent IR schema. No explanation or markdown.
6. Route priority: higher numbers are matched first. Default to 10 if unspecified.
7. Every route MUST have at least one model in the models array.
8. Plugin references in routes should match plugin_template names if templates are defined.
9. ONLY use type names listed in the "DSL Type System" section above. Do NOT invent new signal_type, plugin_type, backend_type, or algo_type values. For example, "jailbreak" and "pii" are signal types, NOT plugin types.
10. When a field has enumerated options listed (e.g., method: "regex" | "bm25" | "ngram"), you MUST choose from those options only.`

const FEW_SHOT_EXAMPLES = `## Examples

### Example 1: Basic routing
User: "Route math questions to GPT-4o and coding questions to DeepSeek"
\`\`\`json
{
  "version": "1.0",
  "operation": "generate",
  "intents": [
    { "type": "signal", "signal_type": "domain", "name": "math", "fields": { "description": "Mathematics and quantitative reasoning", "mmlu_categories": ["math"] } },
    { "type": "signal", "signal_type": "domain", "name": "coding", "fields": { "description": "Programming and software development", "mmlu_categories": ["computer_science"] } },
    { "type": "route", "name": "math_route", "description": "Math questions", "priority": 10, "condition": { "op": "SIGNAL_REF", "signal_type": "domain", "signal_name": "math" }, "models": [{ "model": "gpt-4o", "reasoning": false }] },
    { "type": "route", "name": "coding_route", "description": "Coding questions", "priority": 10, "condition": { "op": "SIGNAL_REF", "signal_type": "domain", "signal_name": "coding" }, "models": [{ "model": "deepseek-v3", "reasoning": false }] },
    { "type": "backend", "backend_type": "provider_profile", "name": "openai", "fields": { "provider": "openai" } },
    { "type": "global", "fields": { "default_model": "gpt-4o-mini", "strategy": "priority" } }
  ]
}
\`\`\`

### Example 2: Complex routing with safety
User: "Set up a 3-tier system: simple questions to qwen2.5:3b, complex ones to qwen3:70b with reasoning, and block jailbreak attempts"
\`\`\`json
{
  "version": "1.0",
  "operation": "generate",
  "intents": [
    { "type": "signal", "signal_type": "complexity", "name": "simple_query", "fields": { "threshold": 0.3 } },
    { "type": "signal", "signal_type": "complexity", "name": "complex_query", "fields": { "threshold": 0.7 } },
    { "type": "signal", "signal_type": "jailbreak", "name": "jailbreak_detect", "fields": { "method": "classifier", "threshold": 0.9 } },
    { "type": "plugin_template", "name": "block_jailbreak", "plugin_type": "fast_response", "fields": { "message": "Request blocked for safety reasons.", "enabled": true } },
    { "type": "route", "name": "safety_route", "description": "Block jailbreak attempts", "priority": 100, "condition": { "op": "SIGNAL_REF", "signal_type": "jailbreak", "signal_name": "jailbreak_detect" }, "models": [{ "model": "qwen2.5:3b" }], "plugins": [{ "name": "block_jailbreak" }] },
    { "type": "route", "name": "complex_route", "description": "Complex queries", "priority": 20, "condition": { "op": "SIGNAL_REF", "signal_type": "complexity", "signal_name": "complex_query" }, "models": [{ "model": "qwen3:70b", "reasoning": true, "effort": "high" }] },
    { "type": "route", "name": "simple_route", "description": "Simple queries", "priority": 10, "condition": { "op": "SIGNAL_REF", "signal_type": "complexity", "signal_name": "simple_query" }, "models": [{ "model": "qwen2.5:3b", "reasoning": false }] },
    { "type": "backend", "backend_type": "vllm_endpoint", "name": "ollama", "fields": { "address": "127.0.0.1", "port": 11434, "type": "ollama" } },
    { "type": "global", "fields": { "default_model": "qwen2.5:3b", "strategy": "priority" } }
  ]
}
\`\`\`

### Example 3: Modify existing config (note: pii is a SIGNAL type, not a plugin type)
User: "Add hallucination detection to the math_route and increase priority to 50"
\`\`\`json
{
  "version": "1.0",
  "operation": "modify",
  "intents": [
    { "type": "signal", "signal_type": "pii", "name": "pii_detect", "fields": { "method": "classifier", "threshold": 0.8 } },
    { "type": "plugin_template", "name": "verify_output", "plugin_type": "hallucination", "fields": { "enabled": true, "method": "nli" } },
    { "type": "modify", "action": "update", "target_construct": "route", "target_name": "math_route", "changes": { "priority": 50, "plugins": [{ "name": "verify_output" }] } }
  ]
}
\`\`\`

### Example 4: Semantic matching with cache
User: "Route AI-related queries to GPT-4o with semantic caching"
\`\`\`json
{
  "version": "1.0",
  "operation": "generate",
  "intents": [
    { "type": "signal", "signal_type": "embedding", "name": "ai_topics", "fields": { "threshold": 0.75, "candidates": ["machine learning", "neural network", "deep learning", "artificial intelligence"], "aggregation_method": "max" } },
    { "type": "plugin_template", "name": "ai_cache", "plugin_type": "semantic_cache", "fields": { "similarity_threshold": 0.95, "max_entries": 10000, "enabled": true } },
    { "type": "route", "name": "ai_route", "description": "AI-related queries", "priority": 10, "condition": { "op": "SIGNAL_REF", "signal_type": "embedding", "signal_name": "ai_topics" }, "models": [{ "model": "gpt-4o" }], "plugins": [{ "name": "ai_cache" }] },
    { "type": "backend", "backend_type": "semantic_cache", "name": "cache_store", "fields": {} },
    { "type": "backend", "backend_type": "provider_profile", "name": "openai", "fields": { "provider": "openai" } },
    { "type": "global", "fields": { "default_model": "gpt-4o-mini", "strategy": "priority" } }
  ]
}
\`\`\`

### Example 5: Multi-condition with AND/OR
User: "Route urgent math questions to GPT-4o, but if it's either physics or chemistry, route to Claude"
\`\`\`json
{
  "version": "1.0",
  "operation": "generate",
  "intents": [
    { "type": "signal", "signal_type": "keyword", "name": "urgent", "fields": { "operator": "any", "keywords": ["urgent", "asap", "emergency"], "method": "regex", "case_sensitive": false } },
    { "type": "signal", "signal_type": "domain", "name": "math", "fields": { "description": "Mathematics", "mmlu_categories": ["math"] } },
    { "type": "signal", "signal_type": "domain", "name": "physics", "fields": { "description": "Physics", "mmlu_categories": ["physics"] } },
    { "type": "signal", "signal_type": "domain", "name": "chemistry", "fields": { "description": "Chemistry", "mmlu_categories": ["chemistry"] } },
    { "type": "route", "name": "urgent_math", "description": "Urgent math questions", "priority": 20, "condition": { "op": "AND", "operands": [{ "op": "SIGNAL_REF", "signal_type": "keyword", "signal_name": "urgent" }, { "op": "SIGNAL_REF", "signal_type": "domain", "signal_name": "math" }] }, "models": [{ "model": "gpt-4o" }] },
    { "type": "route", "name": "science_route", "description": "Physics or chemistry", "priority": 10, "condition": { "op": "OR", "operands": [{ "op": "SIGNAL_REF", "signal_type": "domain", "signal_name": "physics" }, { "op": "SIGNAL_REF", "signal_type": "domain", "signal_name": "chemistry" }] }, "models": [{ "model": "claude-3.5-sonnet" }] },
    { "type": "backend", "backend_type": "provider_profile", "name": "openai", "fields": { "provider": "openai" } },
    { "type": "backend", "backend_type": "provider_profile", "name": "anthropic", "fields": { "provider": "anthropic" } },
    { "type": "global", "fields": { "default_model": "gpt-4o-mini", "strategy": "priority" } }
  ]
}
\`\`\`

### Example 6: NOT condition with algorithm and weighted models
User: "For non-jailbreak queries, use confidence-based cascade with 70% weight on GPT-4o and 30% on GPT-4o-mini"
\`\`\`json
{
  "version": "1.0",
  "operation": "generate",
  "intents": [
    { "type": "signal", "signal_type": "jailbreak", "name": "jailbreak_detect", "fields": { "method": "classifier", "threshold": 0.9 } },
    { "type": "plugin_template", "name": "block_unsafe", "plugin_type": "fast_response", "fields": { "message": "Request blocked for safety reasons.", "enabled": true } },
    { "type": "route", "name": "safety_block", "description": "Block jailbreak attempts", "priority": 100, "condition": { "op": "SIGNAL_REF", "signal_type": "jailbreak", "signal_name": "jailbreak_detect" }, "models": [{ "model": "gpt-4o-mini" }], "plugins": [{ "name": "block_unsafe" }] },
    { "type": "route", "name": "safe_route", "description": "Non-jailbreak queries with cascade", "priority": 10, "condition": { "op": "NOT", "operand": { "op": "SIGNAL_REF", "signal_type": "jailbreak", "signal_name": "jailbreak_detect" } }, "models": [{ "model": "gpt-4o", "weight": 70 }, { "model": "gpt-4o-mini", "weight": 30 }], "algorithm": { "algo_type": "confidence", "params": { "threshold": 0.8 } } },
    { "type": "backend", "backend_type": "provider_profile", "name": "openai", "fields": { "provider": "openai" } },
    { "type": "global", "fields": { "default_model": "gpt-4o-mini", "strategy": "priority" } }
  ]
}
\`\`\``
