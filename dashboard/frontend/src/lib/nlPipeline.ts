/**
 * NL → DSL Pipeline Orchestrator
 *
 * Coordinates the full 3-stage pipeline:
 *   Stage 1: Input classification + context assembly
 *   Stage 2: LLM → Intent IR → DSL codegen
 *   Stage 3: WASM validation + self-repair loop (max 3 retries)
 *
 * This module is the single entry point for the NL Mode UI component.
 * All dependencies (WASM, LLM) are injected for testability.
 */

import type { Diagnostic, SymbolTable, WasmBridge } from '@/types/dsl'
import type { IntentIR, Intent } from '@/types/intentIR'
import { intentIRToDSL, resolveImplicitDependencies } from './intentToDsl'
import { validateGeneratedDSL, filterErrors, formatDiagnostics } from './nlValidation'
import type { ValidationResult } from './nlValidation'
import { repairDSL } from './nlRepair'
import type { LLMClient } from './nlRepair'
import { buildSystemPrompt, buildUserPrompt } from './nlPromptBuilder'
import type { NLPromptContext } from './nlPromptBuilder'

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

export const MAX_RETRIES = 3

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/** Pipeline context — assembled from current editor state */
export interface NLContext {
  /** Current DSL source in the editor (for modify mode) */
  currentDSL?: string
  /** Current symbol table from WASM validation */
  symbols?: SymbolTable
  /** Current diagnostics (for fix mode) */
  diagnostics?: Diagnostic[]
}

/** Progress callback for UI updates */
export type NLProgressCallback = (step: NLProgressStep) => void

export type NLProgressStep =
  | { stage: 'classifying' }
  | { stage: 'generating'; message: string }
  | { stage: 'validating' }
  | { stage: 'repairing'; attempt: number; maxRetries: number; strategy: string }
  | { stage: 'done'; isValid: boolean }
  | { stage: 'error'; message: string }

/** Final result of the NL → DSL pipeline */
export interface NLGenerateResult {
  /** Generated or modified DSL text */
  dsl: string
  /** Compiled YAML (if validation passed) */
  yaml?: string
  /** Remaining diagnostics */
  diagnostics: Diagnostic[]
  /** Whether the DSL is error-free */
  isValid: boolean
  /** The Intent IR produced by the LLM */
  intentIR: IntentIR
  /** Confidence score [0, 1] — decreases with retries and warnings */
  confidence: number
  /** Number of repair retries used */
  retries: number
  /** Natural language explanation of what was generated */
  explanation: string
}

/** Session turn for multi-turn conversation support */
export interface NLTurn {
  /** User's NL input */
  userInput: string
  /** Generated Intent IR */
  intentIR: IntentIR
  /** Resulting DSL */
  generatedDSL: string
  /** Whether user accepted this result */
  accepted: boolean
}

/** Multi-turn session state */
export interface NLSession {
  id: string
  turns: NLTurn[]
  currentDSL: string
  symbolTable: SymbolTable | null
}

// ─────────────────────────────────────────────
// Main Pipeline Entry Point
// ─────────────────────────────────────────────

/**
 * Run the full NL → DSL pipeline.
 *
 * @param nlInput     - User's natural language input
 * @param context     - Current editor context (DSL, symbols, diagnostics)
 * @param wasm        - WASM bridge for validation/compilation
 * @param llmClient   - LLM client for Intent IR generation
 * @param onProgress  - Optional progress callback for UI
 * @returns Complete generation result with DSL, YAML, diagnostics, confidence
 */
export async function nlToDSL(
  nlInput: string,
  context: NLContext,
  wasm: WasmBridge,
  llmClient: LLMClient,
  onProgress?: NLProgressCallback,
): Promise<NLGenerateResult> {
  // ── Stage 1: Classify & extract context ──────────
  onProgress?.({ stage: 'classifying' })

  const mode = classifyOperation(nlInput, context)

  // ── Stage 2: LLM → Intent IR → DSL ──────────────
  onProgress?.({ stage: 'generating', message: 'Building prompts...' })

  const systemPrompt = buildSystemPrompt()
  const promptContext = buildPromptContext(nlInput, mode, context)
  const userPrompt = buildUserPrompt(promptContext)

  onProgress?.({ stage: 'generating', message: 'Calling LLM...' })

  let intentIR: IntentIR
  try {
    intentIR = await llmClient.generateIntentIR(systemPrompt, userPrompt)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    onProgress?.({ stage: 'error', message: `LLM call failed: ${msg}` })
    throw new Error(`LLM generation failed: ${msg}`)
  }

  // Resolve implicit dependencies (auto-create missing signals)
  intentIR = resolveImplicitDependencies(intentIR)

  onProgress?.({ stage: 'generating', message: 'Converting to DSL...' })

  const existingDSL = mode === 'modify' ? context.currentDSL : undefined
  let dsl = intentIRToDSL(intentIR, existingDSL)

  // ── Stage 3: Validate & repair loop ──────────────
  onProgress?.({ stage: 'validating' })

  let validation = validateGeneratedDSL(dsl, wasm)
  let attempt = 0

  while (!validation.isValid && attempt < MAX_RETRIES) {
    attempt++

    const strategyName = attempt === 1 ? 'QuickFix' : attempt === 2 ? 'Targeted LLM' : 'Full regeneration'
    onProgress?.({
      stage: 'repairing',
      attempt,
      maxRetries: MAX_RETRIES,
      strategy: strategyName,
    })

    const repairResult = await repairDSL(
      nlInput,
      dsl,
      validation.diagnostics,
      wasm,
      llmClient,
      systemPrompt,
      attempt,
      existingDSL,
    )

    dsl = repairResult.dsl
    validation = repairResult.validation

    // If repair produced a new Intent IR (strategies 2 & 3), we could update it
    // but for now we keep the original for explanation purposes
  }

  const confidence = computeConfidence(validation, attempt)
  const explanation = generateExplanation(intentIR)

  onProgress?.({ stage: 'done', isValid: validation.isValid })

  return {
    dsl,
    yaml: validation.yaml,
    diagnostics: validation.diagnostics,
    isValid: validation.isValid,
    intentIR,
    confidence,
    retries: attempt,
    explanation,
  }
}

// ─────────────────────────────────────────────
// Stage 1: Input Classification
// ─────────────────────────────────────────────

/**
 * Classify whether the NL input is a "generate" or "modify" operation.
 *
 * Heuristics:
 * - If there's existing DSL and the input uses modification language → modify
 * - If input mentions specific existing entities → modify
 * - Otherwise → generate
 */
export function classifyOperation(
  nlInput: string,
  context: NLContext,
): 'generate' | 'modify' {
  // No existing DSL → always generate
  if (!context.currentDSL?.trim()) return 'generate'

  const lower = nlInput.toLowerCase()

  // Explicit modification language
  const modifyPatterns = [
    /\b(add|append|insert)\b.*\b(to|into|for)\b/,
    /\b(remove|delete|drop)\b/,
    /\b(change|update|modify|set|adjust)\b/,
    /\b(increase|decrease|lower|raise|bump)\b/,
    /\b(enable|disable|turn on|turn off)\b/,
    /\b(replace|swap)\b/,
    /\bfix\b/,
  ]

  for (const pattern of modifyPatterns) {
    if (pattern.test(lower)) return 'modify'
  }

  // References to existing entities (e.g., "the math route", "my signal")
  if (context.symbols) {
    const entityNames = [
      ...context.symbols.routes,
      ...context.symbols.signals.map(s => s.name),
      ...context.symbols.plugins,
      ...context.symbols.backends.map(b => b.name),
    ]
    for (const name of entityNames) {
      if (lower.includes(name.toLowerCase())) return 'modify'
    }
  }

  return 'generate'
}

// ─────────────────────────────────────────────
// Context Assembly
// ─────────────────────────────────────────────

function buildPromptContext(
  nlInput: string,
  mode: 'generate' | 'modify',
  context: NLContext,
): NLPromptContext {
  const promptCtx: NLPromptContext = {
    userInput: nlInput,
    mode,
  }

  if (context.symbols) {
    promptCtx.symbols = {
      signals: context.symbols.signals.map(s => ({ name: s.name, type: s.type })),
      routes: context.symbols.routes,
      plugins: context.symbols.plugins,
      backends: context.symbols.backends.map(b => ({ name: b.name, type: b.type })),
      models: context.symbols.models,
    }
  }

  if (mode === 'modify' && context.currentDSL) {
    promptCtx.currentDSL = context.currentDSL
  }

  if (context.diagnostics && context.diagnostics.length > 0) {
    promptCtx.diagnostics = context.diagnostics.map(d => ({
      level: d.level,
      message: d.message,
    }))
  }

  return promptCtx
}

// ─────────────────────────────────────────────
// Confidence Scoring
// ─────────────────────────────────────────────

/**
 * Compute a confidence score [0, 1] based on validation state and retry count.
 *
 * Scoring:
 *   - Start at 1.0
 *   - -0.5 if still invalid
 *   - -0.1 per retry attempt
 *   - -0.05 per remaining warning
 */
export function computeConfidence(validation: ValidationResult, retries: number): number {
  let score = 1.0

  if (!validation.isValid) score -= 0.5
  score -= retries * 0.1
  score -= validation.warningCount * 0.05

  return Math.max(0, Math.min(1, score))
}

// ─────────────────────────────────────────────
// Explanation Generation
// ─────────────────────────────────────────────

/**
 * Generate a natural language explanation of what the Intent IR will produce.
 */
export function generateExplanation(ir: IntentIR): string {
  const parts: string[] = []

  const signals = ir.intents.filter(i => i.type === 'signal')
  const routes = ir.intents.filter(i => i.type === 'route')
  const plugins = ir.intents.filter(i => i.type === 'plugin_template')
  const backends = ir.intents.filter(i => i.type === 'backend')
  const modifies = ir.intents.filter(i => i.type === 'modify')

  if (ir.operation === 'generate') {
    parts.push('Generated configuration:')

    if (signals.length > 0) {
      const sigList = signals.map(s => {
        const sig = s as { signal_type: string; name: string }
        return `${sig.signal_type}("${sig.name}")`
      })
      parts.push(`- ${signals.length} signal${signals.length > 1 ? 's' : ''}: ${sigList.join(', ')}`)
    }

    if (routes.length > 0) {
      const routeList = routes.map(r => {
        const route = r as { name: string; description?: string }
        return route.description ? `${route.name} (${route.description})` : route.name
      })
      parts.push(`- ${routes.length} route${routes.length > 1 ? 's' : ''}: ${routeList.join(', ')}`)
    }

    if (plugins.length > 0) {
      const pluginList = plugins.map(p => {
        const pl = p as { name: string; plugin_type: string }
        return `${pl.name} (${pl.plugin_type})`
      })
      parts.push(`- ${plugins.length} plugin${plugins.length > 1 ? 's' : ''}: ${pluginList.join(', ')}`)
    }

    if (backends.length > 0) {
      const beList = backends.map(b => {
        const be = b as { name: string; backend_type: string }
        return `${be.name} (${be.backend_type})`
      })
      parts.push(`- ${backends.length} backend${backends.length > 1 ? 's' : ''}: ${beList.join(', ')}`)
    }
  } else {
    parts.push('Modified configuration:')

    for (const m of modifies) {
      const mod = m as { action: string; target_construct: string; target_name: string }
      parts.push(`- ${mod.action} ${mod.target_construct} "${mod.target_name}"`)
    }

    if (signals.length > 0) {
      parts.push(`- Added ${signals.length} new signal${signals.length > 1 ? 's' : ''}`)
    }
    if (plugins.length > 0) {
      parts.push(`- Added ${plugins.length} new plugin${plugins.length > 1 ? 's' : ''}`)
    }
  }

  return parts.join('\n')
}

// ─────────────────────────────────────────────
// Session Management (Multi-turn)
// ─────────────────────────────────────────────

/**
 * Create a new NL session.
 */
export function createSession(): NLSession {
  return {
    id: generateSessionId(),
    turns: [],
    currentDSL: '',
    symbolTable: null,
  }
}

/**
 * Record a turn in the session.
 */
export function addTurn(
  session: NLSession,
  userInput: string,
  intentIR: IntentIR,
  generatedDSL: string,
): NLSession {
  return {
    ...session,
    turns: [
      ...session.turns,
      { userInput, intentIR, generatedDSL, accepted: false },
    ],
  }
}

/**
 * Mark the last turn as accepted and update session state.
 */
export function acceptLastTurn(
  session: NLSession,
  dsl: string,
  symbols: SymbolTable | null,
): NLSession {
  const turns = [...session.turns]
  if (turns.length > 0) {
    turns[turns.length - 1] = { ...turns[turns.length - 1], accepted: true }
  }
  return {
    ...session,
    turns,
    currentDSL: dsl,
    symbolTable: symbols,
  }
}

/**
 * Mark the last turn as rejected.
 */
export function rejectLastTurn(session: NLSession): NLSession {
  const turns = [...session.turns]
  if (turns.length > 0) {
    turns[turns.length - 1] = { ...turns[turns.length - 1], accepted: false }
  }
  return { ...session, turns }
}

function generateSessionId(): string {
  return `nl-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`
}

// ─────────────────────────────────────────────
// Re-exports for convenience
// ─────────────────────────────────────────────

export type { LLMClient } from './nlRepair'
export type { ValidationResult } from './nlValidation'
export { filterErrors, filterWarnings, formatDiagnostics } from './nlValidation'
export { buildSystemPrompt, buildUserPrompt, buildRepairPrompt } from './nlPromptBuilder'
