/**
 * NL Self-Repair — Escalating repair strategies for invalid generated DSL.
 *
 * Three strategies, escalating in cost:
 *   Strategy 1: Deterministic QuickFix (< 5ms, no LLM)
 *   Strategy 2: Targeted LLM repair — fix only the errors (< 1.5s)
 *   Strategy 3: Full LLM regeneration from scratch (< 2s)
 *
 * The repair loop is orchestrated by nlPipeline.ts.
 */

import type { Diagnostic, WasmBridge } from '@/types/dsl'
import type { IntentIR } from '@/types/intentIR'
import { applyQuickFixes, hasQuickFixes, validateGeneratedDSL } from './nlValidation'
import type { ValidationResult } from './nlValidation'
import { buildRepairPrompt } from './nlPromptBuilder'
import type { RepairPromptContext } from './nlPromptBuilder'
import { intentIRToDSL, resolveImplicitDependencies } from './intentToDsl'

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/**
 * LLM client interface — thin abstraction over the actual LLM API call.
 * The pipeline injects this so repair logic is testable without real LLM calls.
 */
export interface LLMClient {
  /**
   * Send a prompt to the LLM and receive the generated Intent IR JSON.
   * The implementation should handle JSON parsing and streaming.
   */
  generateIntentIR(
    systemPrompt: string,
    userPrompt: string,
  ): Promise<IntentIR>
}

export interface RepairResult {
  /** Repaired DSL text */
  dsl: string
  /** Validation result after repair */
  validation: ValidationResult
  /** Which strategy was used */
  strategy: 'quickfix' | 'targeted_llm' | 'full_regen'
}

// ─────────────────────────────────────────────
// Main Repair Dispatcher
// ─────────────────────────────────────────────

/**
 * Attempt to repair invalid DSL using escalating strategies.
 *
 * @param originalNL  - The original natural language input
 * @param failedDSL   - The DSL that failed validation
 * @param diagnostics - Validation diagnostics from the failed DSL
 * @param wasm        - WASM bridge for re-validation
 * @param llmClient   - LLM client for strategies 2 & 3
 * @param systemPrompt - System prompt for LLM calls
 * @param attempt     - 1 = quickfix, 2 = targeted LLM, 3 = full regen
 * @param existingDSL - For modify mode, the original DSL before modifications
 */
export async function repairDSL(
  originalNL: string,
  failedDSL: string,
  diagnostics: Diagnostic[],
  wasm: WasmBridge,
  llmClient: LLMClient,
  systemPrompt: string,
  attempt: number,
  existingDSL?: string,
): Promise<RepairResult> {
  switch (attempt) {
    case 1:
      return repairWithQuickFixes(failedDSL, diagnostics, wasm)
    case 2:
      return repairWithTargetedLLM(originalNL, failedDSL, diagnostics, wasm, llmClient, systemPrompt, existingDSL)
    default:
      return repairWithFullRegeneration(originalNL, failedDSL, diagnostics, wasm, llmClient, systemPrompt, existingDSL)
  }
}

// ─────────────────────────────────────────────
// Strategy 1: Deterministic QuickFix
// ─────────────────────────────────────────────

/**
 * Apply QuickFixes from the WASM validator's diagnostic suggestions.
 * This is the fastest and cheapest repair strategy — no LLM call needed.
 */
function repairWithQuickFixes(
  failedDSL: string,
  diagnostics: Diagnostic[],
  wasm: WasmBridge,
): Promise<RepairResult> {
  let repairedDSL = failedDSL

  if (hasQuickFixes(diagnostics)) {
    repairedDSL = applyQuickFixes(failedDSL, diagnostics)
  }

  const validation = validateGeneratedDSL(repairedDSL, wasm)

  return Promise.resolve({
    dsl: repairedDSL,
    validation,
    strategy: 'quickfix' as const,
  })
}

// ─────────────────────────────────────────────
// Strategy 2: Targeted LLM Repair
// ─────────────────────────────────────────────

/**
 * Send the error list + failed DSL to the LLM for surgical fixes.
 * Only the errors are highlighted — the LLM should fix them without
 * changing other parts of the configuration.
 */
async function repairWithTargetedLLM(
  originalNL: string,
  failedDSL: string,
  diagnostics: Diagnostic[],
  wasm: WasmBridge,
  llmClient: LLMClient,
  systemPrompt: string,
  existingDSL?: string,
): Promise<RepairResult> {
  const repairCtx: RepairPromptContext = {
    originalNL,
    failedDSL,
    diagnostics: diagnostics.map(d => ({
      level: d.level,
      message: d.message,
      line: d.line,
    })),
    attempt: 2,
  }

  const repairPrompt = buildRepairPrompt(repairCtx)

  try {
    let intentIR = await llmClient.generateIntentIR(systemPrompt, repairPrompt)
    intentIR = resolveImplicitDependencies(intentIR)
    const repairedDSL = intentIRToDSL(intentIR, existingDSL)
    const validation = validateGeneratedDSL(repairedDSL, wasm)

    return {
      dsl: repairedDSL,
      validation,
      strategy: 'targeted_llm',
    }
  } catch {
    // If LLM call fails, return original with unchanged validation
    const validation = validateGeneratedDSL(failedDSL, wasm)
    return {
      dsl: failedDSL,
      validation,
      strategy: 'targeted_llm',
    }
  }
}

// ─────────────────────────────────────────────
// Strategy 3: Full Regeneration
// ─────────────────────────────────────────────

/**
 * Full regeneration from scratch with error context as negative examples.
 * The LLM is told what went wrong and asked to regenerate completely.
 */
async function repairWithFullRegeneration(
  originalNL: string,
  failedDSL: string,
  diagnostics: Diagnostic[],
  wasm: WasmBridge,
  llmClient: LLMClient,
  systemPrompt: string,
  existingDSL?: string,
): Promise<RepairResult> {
  const repairCtx: RepairPromptContext = {
    originalNL,
    failedDSL,
    diagnostics: diagnostics.map(d => ({
      level: d.level,
      message: d.message,
      line: d.line,
    })),
    attempt: 3,
  }

  const repairPrompt = buildRepairPrompt(repairCtx)

  try {
    let intentIR = await llmClient.generateIntentIR(systemPrompt, repairPrompt)
    intentIR = resolveImplicitDependencies(intentIR)
    const repairedDSL = intentIRToDSL(intentIR, existingDSL)
    const validation = validateGeneratedDSL(repairedDSL, wasm)

    return {
      dsl: repairedDSL,
      validation,
      strategy: 'full_regen',
    }
  } catch {
    // If LLM call fails, return original
    const validation = validateGeneratedDSL(failedDSL, wasm)
    return {
      dsl: failedDSL,
      validation,
      strategy: 'full_regen',
    }
  }
}
