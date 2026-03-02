/**
 * Intent IR (Intermediate Representation) for NL → DSL conversion.
 *
 * The Intent IR is a structured JSON format that serves as the bridge between
 * LLM natural language understanding and deterministic DSL code generation.
 * It mirrors the DSL's 5 constructs (SIGNAL, ROUTE, PLUGIN, BACKEND, GLOBAL)
 * but with relaxed constraints — allowing the LLM to output structured data
 * rather than raw DSL text.
 *
 * Flow:  NL input → LLM → Intent IR (JSON) → intentToDsl() → DSL text → WASM validate
 */

import type { SignalType, AlgorithmType } from '../lib/dslMutations'

// ─────────────────────────────────────────────
// Top-level IR
// ─────────────────────────────────────────────

export interface IntentIR {
  /** Schema version for forward compatibility */
  version: '1.0'
  /** Operation mode */
  operation: 'generate' | 'modify' | 'fix'
  /** Ordered list of intents (ordering matters for DSL output) */
  intents: Intent[]
}

/** Discriminated union of all intent types */
export type Intent =
  | SignalIntent
  | RouteIntent
  | PluginTemplateIntent
  | BackendIntent
  | GlobalIntent
  | ModifyIntent

// ─────────────────────────────────────────────
// Signal Intent
// ─────────────────────────────────────────────

export interface SignalIntent {
  type: 'signal'
  /** One of 13 signal types */
  signal_type: SignalType
  /** Signal name (e.g., "math", "urgent_request") — auto-sanitized to valid identifier */
  name: string
  /** Type-specific fields matching getSignalFieldSchema() keys */
  fields: Record<string, unknown>
}

// ─────────────────────────────────────────────
// Route Intent
// ─────────────────────────────────────────────

export interface RouteIntent {
  type: 'route'
  /** Route name (valid identifier) */
  name: string
  /** Human-readable description */
  description?: string
  /** Route priority (lower = higher priority, default 10) */
  priority?: number
  /** Boolean expression tree over signal references */
  condition: ConditionNode
  /** One or more model references */
  models: ModelIntent[]
  /** Optional algorithm selection */
  algorithm?: AlgorithmIntent
  /** Optional plugin references (by template name) */
  plugins?: PluginRefIntent[]
}

/**
 * Boolean expression tree for WHEN clauses.
 * Mirrors BoolExprNode from dsl.ts but without position info.
 */
export type ConditionNode =
  | { op: 'AND'; operands: ConditionNode[] }
  | { op: 'OR'; operands: ConditionNode[] }
  | { op: 'NOT'; operand: ConditionNode }
  | { op: 'SIGNAL_REF'; signal_type: string; signal_name: string }

export interface ModelIntent {
  /** Model name (e.g., "gpt-4o", "deepseek-r1") */
  model: string
  /** Whether this is a reasoning model */
  reasoning?: boolean
  /** Reasoning effort level */
  effort?: 'low' | 'medium' | 'high'
  /** LoRA adapter name */
  lora?: string
  /** Model parameter size hint */
  param_size?: string
  /** Weight for multi-model routing */
  weight?: number
  /** Reasoning model family hint */
  reasoning_family?: string
}

export interface AlgorithmIntent {
  /** One of 14 algorithm types */
  algo_type: AlgorithmType
  /** Algorithm-specific parameters matching getAlgorithmFieldSchema() keys */
  params: Record<string, unknown>
}

export interface PluginRefIntent {
  /** Plugin template name (references a PLUGIN declaration) or inline plugin type */
  name: string
  /** Optional field overrides for inline plugin usage within a route */
  overrides?: Record<string, unknown>
}

// ─────────────────────────────────────────────
// Plugin Template Intent
// ─────────────────────────────────────────────

export interface PluginTemplateIntent {
  type: 'plugin_template'
  /** Plugin declaration name (used as reference in routes) */
  name: string
  /** One of 9 plugin types */
  plugin_type: string
  /** Type-specific fields matching getPluginFieldSchema() keys */
  fields: Record<string, unknown>
}

// ─────────────────────────────────────────────
// Backend Intent
// ─────────────────────────────────────────────

export interface BackendIntent {
  type: 'backend'
  /** One of 8 backend types */
  backend_type: string
  /** Backend name (used as reference in plugins like RAG) */
  name: string
  /** Backend-specific fields */
  fields: Record<string, unknown>
}

// ─────────────────────────────────────────────
// Global Intent
// ─────────────────────────────────────────────

export interface GlobalIntent {
  type: 'global'
  /** Global configuration fields */
  fields: Record<string, unknown>
}

// ─────────────────────────────────────────────
// Modify Intent (for incremental edits)
// ─────────────────────────────────────────────

export interface ModifyIntent {
  type: 'modify'
  /** Modification action */
  action: 'add' | 'update' | 'delete'
  /** Which DSL construct to modify */
  target_construct: 'signal' | 'route' | 'plugin' | 'backend' | 'global'
  /** Name of the target entity */
  target_name: string
  /** For signals: signal_type is needed to locate the block */
  target_signal_type?: string
  /** For plugins: plugin_type is needed */
  target_plugin_type?: string
  /** For backends: backend_type is needed */
  target_backend_type?: string
  /** Fields to add/update (ignored for "delete" action) */
  changes?: Record<string, unknown>
}

// ─────────────────────────────────────────────
// NL Pipeline Types
// ─────────────────────────────────────────────

/** Result from the NL → DSL pipeline */
export interface NLGenerateResult {
  /** Generated or modified DSL text */
  dsl: string
  /** The Intent IR that was produced (for debugging / UI preview) */
  intentIR: IntentIR
  /** Whether any auto-repair was applied */
  repaired: boolean
  /** Number of repair attempts */
  repairAttempts: number
  /** Diagnostics from final validation (should be empty on success) */
  diagnostics: Array<{ level: string; message: string }>
}

/** Request to the NL generate API */
export interface NLGenerateRequest {
  /** Natural language prompt from the user */
  prompt: string
  /** Current DSL source (for "modify" mode context) */
  current_dsl?: string
  /** Operation mode hint */
  mode: 'generate' | 'modify'
  /** Optional: current symbol table for context-aware generation */
  symbols?: {
    signals: Array<{ name: string; type: string }>
    models: string[]
    plugins: string[]
    backends: Array<{ name: string; type: string }>
    routes: string[]
  }
}

/** NL session for multi-turn conversation */
export interface NLSession {
  /** Session ID */
  id: string
  /** Conversation turns */
  turns: NLTurn[]
  /** Timestamp of session creation */
  created_at: number
}

export interface NLTurn {
  /** User's natural language input */
  input: string
  /** Generated Intent IR */
  intent_ir?: IntentIR
  /** Generated DSL */
  dsl?: string
  /** Whether the user accepted the result */
  accepted?: boolean
  /** Error message if generation failed */
  error?: string
  /** Timestamp */
  timestamp: number
}
