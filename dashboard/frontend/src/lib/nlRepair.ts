/**
 * NL Self-Repair — Escalating repair strategies for invalid generated DSL.
 *
 * Three strategies, escalating in cost:
 *   Strategy 1: Deterministic QuickFix (< 5ms, no LLM)
 *   Strategy 2: Targeted LLM repair — fix only the errors (< 1.5s)
 *   Strategy 3: Full LLM regeneration from scratch (< 2s)
 *
 * Optimizations:
 *   - Enhanced deterministic fixes for common LLM mistakes
 *   - Signal type correction using fuzzy matching
 *   - Threshold clamping for out-of-range values
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
 *
 * Enhanced with additional deterministic fixes for common LLM mistakes:
 * - Signal type correction (fuzzy matching)
 * - Threshold value clamping
 * - Missing field defaults
 * - Duplicate name deduplication
 */
function repairWithQuickFixes(
  failedDSL: string,
  diagnostics: Diagnostic[],
  wasm: WasmBridge,
): Promise<RepairResult> {
  let repairedDSL = failedDSL

  // Apply WASM-suggested quick fixes first
  if (hasQuickFixes(diagnostics)) {
    repairedDSL = applyQuickFixes(failedDSL, diagnostics)
  }

  // Apply additional deterministic fixes
  repairedDSL = applyDeterministicFixes(repairedDSL, diagnostics)

  const validation = validateGeneratedDSL(repairedDSL, wasm)

  return Promise.resolve({
    dsl: repairedDSL,
    validation,
    strategy: 'quickfix' as const,
  })
}

/**
 * Apply deterministic fixes for common error patterns.
 */
function applyDeterministicFixes(dsl: string, diagnostics: Diagnostic[]): string {
  let result = dsl

  for (const diag of diagnostics) {
    const msg = diag.message.toLowerCase()

    // Fix: Unknown signal type → suggest nearest match
    if (msg.includes('unknown signal type')) {
      result = fixUnknownSignalType(result, diag)
    }

    // Fix: Threshold out of range [0, 1]
    if (msg.includes('threshold') && (msg.includes('range') || msg.includes('must be'))) {
      result = fixThresholdRange(result, diag)
    }

    // Fix: Port out of range [1, 65535]
    if (msg.includes('port') && msg.includes('range')) {
      result = fixPortRange(result, diag)
    }

    // Fix: Missing required field
    if (msg.includes('missing required field')) {
      result = fixMissingRequiredField(result, diag)
    }

    // Fix: Duplicate signal/route name
    if (msg.includes('duplicate') && (msg.includes('signal') || msg.includes('route'))) {
      result = fixDuplicateName(result, diag)
    }
  }

  return result
}

/**
 * Fix unknown signal type by finding the closest valid type.
 */
function fixUnknownSignalType(dsl: string, diag: Diagnostic): string {
  const validSignalTypes = [
    'keyword', 'embedding', 'domain', 'fact_check', 'user_feedback',
    'preference', 'language', 'context', 'complexity', 'modality',
    'authz', 'jailbreak', 'pii',
  ]

  // Extract the unknown type from the error message
  const match = diag.message.match(/unknown signal type[:\s]+["']?(\w+)["']?/i)
  if (!match) return dsl

  const unknownType = match[1].toLowerCase()

  // Find the closest valid type using simple similarity
  let bestMatch = 'keyword'
  let bestScore = 0

  for (const validType of validSignalTypes) {
    const score = stringSimilarity(unknownType, validType)
    if (score > bestScore) {
      bestScore = score
      bestMatch = validType
    }
  }

  // Only replace if we have a reasonable match
  if (bestScore > 0.3) {
    // Replace in the DSL
    const regex = new RegExp(`SIGNAL\\s+${unknownType}\\s*\\(`, 'gi')
    return dsl.replace(regex, `SIGNAL ${bestMatch}(`)
  }

  return dsl
}

/**
 * Clamp threshold values to [0, 1] range.
 */
function fixThresholdRange(dsl: string, diag: Diagnostic): string {
  // Extract line number if available
  if (diag.line) {
    const lines = dsl.split('\n')
    if (diag.line > 0 && diag.line <= lines.length) {
      const line = lines[diag.line - 1]
      // Find threshold value and clamp it
      const thresholdMatch = line.match(/threshold\s*=\s*(-?[\d.]+)/i)
      if (thresholdMatch) {
        const value = parseFloat(thresholdMatch[1])
        const clamped = Math.max(0, Math.min(1, value))
        lines[diag.line - 1] = line.replace(
          /threshold\s*=\s*-?[\d.]+/i,
          `threshold = ${clamped}`
        )
        return lines.join('\n')
      }
    }
  }
  return dsl
}

/**
 * Clamp port values to [1, 65535] range.
 */
function fixPortRange(dsl: string, diag: Diagnostic): string {
  if (diag.line) {
    const lines = dsl.split('\n')
    if (diag.line > 0 && diag.line <= lines.length) {
      const line = lines[diag.line - 1]
      const portMatch = line.match(/port\s*=\s*(-?[\d]+)/i)
      if (portMatch) {
        const value = parseInt(portMatch[1], 10)
        const clamped = Math.max(1, Math.min(65535, value))
        lines[diag.line - 1] = line.replace(
          /port\s*=\s*-?[\d]+/i,
          `port = ${clamped}`
        )
        return lines.join('\n')
      }
    }
  }
  return dsl
}

/**
 * Add default value for missing required field.
 */
function fixMissingRequiredField(dsl: string, diag: Diagnostic): string {
  // Extract field name from error message
  const match = diag.message.match(/missing required field[:\s]+["']?(\w+)["']?/i)
  if (!match) return dsl

  const fieldName = match[1].toLowerCase()

  // Default values for common required fields
  const defaults: Record<string, string> = {
    threshold: '0.7',
    model: '"gpt-4o-mini"',
    name: '"default"',
    provider: '"openai"',
    message: '"Default message"',
    keywords: '["keyword"]',
  }

  if (!defaults[fieldName]) return dsl

  // Insert the field before the closing paren/brace of the block
  // This is a simplified heuristic
  if (diag.line) {
    const lines = dsl.split('\n')
    if (diag.line > 0 && diag.line <= lines.length) {
      const line = lines[diag.line - 1]
      // Add the field if line ends with ) or }
      if (line.match(/[)}]\s*$/)) {
        const indent = line.match(/^\s*/)?.[0] || '  '
        lines.splice(diag.line - 1, 0, `${indent}${fieldName} = ${defaults[fieldName]}`)
        return lines.join('\n')
      }
    }
  }

  return dsl
}

/**
 * Deduplicate signal/route names by adding a suffix.
 */
function fixDuplicateName(dsl: string, diag: Diagnostic): string {
  // Extract the duplicate name
  const match = diag.message.match(/duplicate (?:signal|route)[:\s]+["']?(\w+)["']?/i)
  if (!match) return dsl

  const duplicateName = match[1]

  // Find all occurrences and rename the second one
  let count = 0
  return dsl.replace(
    new RegExp(`(SIGNAL\\s+\\w+\\s*\\()["']?${duplicateName}["']?`, 'gi'),
    (full, prefix) => {
      count++
      if (count > 1) {
        return `${prefix}"${duplicateName}_${count}"`
      }
      return full
    }
  )
}

/**
 * Simple string similarity (Dice coefficient).
 */
function stringSimilarity(s1: string, s2: string): number {
  const bigrams1 = getBigrams(s1)
  const bigrams2 = getBigrams(s2)

  if (bigrams1.size === 0 && bigrams2.size === 0) return 1
  if (bigrams1.size === 0 || bigrams2.size === 0) return 0

  let intersection = 0
  for (const bigram of bigrams1) {
    if (bigrams2.has(bigram)) intersection++
  }

  return (2 * intersection) / (bigrams1.size + bigrams2.size)
}

function getBigrams(str: string): Set<string> {
  const bigrams = new Set<string>()
  for (let i = 0; i < str.length - 1; i++) {
    bigrams.add(str.slice(i, i + 2))
  }
  return bigrams
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
