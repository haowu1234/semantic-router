/**
 * NL Validation — WASM validation wrapper for NL-generated DSL.
 *
 * Wraps the wasmBridge.validate() and wasmBridge.compile() calls into
 * a structured result that the pipeline can consume. Also provides
 * QuickFix application logic for deterministic self-repair (Strategy 1).
 */

import type {
  Diagnostic,
  SymbolTable,
  WasmBridge,
} from '@/types/dsl'

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

export interface ValidationResult {
  /** Whether the DSL is free of errors (warnings are OK) */
  isValid: boolean
  /** The DSL that was validated */
  dsl: string
  /** All diagnostics from validation */
  diagnostics: Diagnostic[]
  /** Number of error-level diagnostics */
  errorCount: number
  /** Number of warning-level diagnostics */
  warningCount: number
  /** Symbol table extracted from the DSL (signals, routes, plugins, etc.) */
  symbolTable: SymbolTable | null
  /** Compiled YAML output — only present if DSL is valid and compilation succeeds */
  yaml?: string
}

// ─────────────────────────────────────────────
// Core Validation
// ─────────────────────────────────────────────

/**
 * Validate generated DSL through the WASM validation stack.
 * If validation passes (no errors), also attempts compilation to YAML.
 */
export function validateGeneratedDSL(dsl: string, wasm: WasmBridge): ValidationResult {
  if (!dsl.trim()) {
    return {
      isValid: true,
      dsl,
      diagnostics: [],
      errorCount: 0,
      warningCount: 0,
      symbolTable: null,
    }
  }

  // Step 1: Validate (diagnostics + symbol table)
  const validateResult = wasm.validate(dsl)

  if (validateResult.error) {
    return {
      isValid: false,
      dsl,
      diagnostics: validateResult.diagnostics || [],
      errorCount: (validateResult.diagnostics || []).filter(d => d.level === 'error').length + 1,
      warningCount: (validateResult.diagnostics || []).filter(d => d.level === 'warning').length,
      symbolTable: validateResult.symbols || null,
    }
  }

  const errorCount = validateResult.errorCount ?? 0
  const warningCount = (validateResult.diagnostics || []).filter(d => d.level === 'warning').length

  // Step 2: If no errors, also compile to get YAML
  let yaml: string | undefined
  if (errorCount === 0) {
    try {
      const compileResult = wasm.compile(dsl)
      const compileErrors = (compileResult.diagnostics || []).filter(d => d.level === 'error')
      if (compileErrors.length === 0 && !compileResult.error) {
        yaml = compileResult.yaml || undefined
      }
    } catch {
      // Compilation failure is non-fatal — DSL is valid but may have semantic issues
    }
  }

  return {
    isValid: errorCount === 0,
    dsl,
    diagnostics: validateResult.diagnostics || [],
    errorCount,
    warningCount,
    symbolTable: validateResult.symbols || null,
    yaml,
  }
}

// ─────────────────────────────────────────────
// QuickFix Application (Strategy 1: Deterministic)
// ─────────────────────────────────────────────

/**
 * Apply all available QuickFixes from diagnostics to the DSL text.
 * Fixes are applied in reverse line order to avoid position shifts.
 *
 * Handles:
 * - Typos in signal/backend/plugin names (Levenshtein → "did you mean X?")
 * - Unknown types → closest valid type
 * - Threshold out-of-range → clamped to [0, 1]
 * - Negative priority → set to 0
 *
 * Returns the patched DSL text. If no fixes are available, returns the original.
 */
export function applyQuickFixes(dsl: string, diagnostics: Diagnostic[]): string {
  // Collect all fixes with their positions
  const fixes = diagnostics
    .flatMap(d =>
      (d.fixes || []).map(f => ({
        line: d.line,
        column: d.column,
        newText: f.newText,
        description: f.description,
      }))
    )
    // Sort by line descending, then column descending — apply from bottom to top
    .sort((a, b) => b.line - a.line || b.column - a.column)

  if (fixes.length === 0) return dsl

  const lines = dsl.split('\n')

  for (const fix of fixes) {
    const lineIdx = fix.line - 1
    if (lineIdx < 0 || lineIdx >= lines.length) continue

    const line = lines[lineIdx]
    if (!line || !fix.newText) continue

    const colIdx = fix.column - 1
    if (colIdx < 0 || colIdx >= line.length) continue

    // Replace the token at the column position
    const before = line.substring(0, colIdx)
    const after = line.substring(colIdx)
    // Replace the first identifier-like token starting at column
    const replaced = after.replace(/[\w\-\.]+/, fix.newText)
    lines[lineIdx] = before + replaced
  }

  return lines.join('\n')
}

// ─────────────────────────────────────────────
// Diagnostic Helpers
// ─────────────────────────────────────────────

/**
 * Filter diagnostics to only error-level ones.
 */
export function filterErrors(diagnostics: Diagnostic[]): Diagnostic[] {
  return diagnostics.filter(d => d.level === 'error')
}

/**
 * Filter diagnostics to only warning-level ones.
 */
export function filterWarnings(diagnostics: Diagnostic[]): Diagnostic[] {
  return diagnostics.filter(d => d.level === 'warning')
}

/**
 * Check if any diagnostics have available QuickFixes.
 */
export function hasQuickFixes(diagnostics: Diagnostic[]): boolean {
  return diagnostics.some(d => d.fixes && d.fixes.length > 0)
}

/**
 * Format diagnostics as a human-readable summary for LLM repair prompts.
 */
export function formatDiagnostics(diagnostics: Diagnostic[]): string {
  return diagnostics
    .map((d, i) => {
      const prefix = `${i + 1}. [${d.level}]`
      const location = d.line ? ` Line ${d.line}` : ''
      const fixHint = d.fixes && d.fixes.length > 0
        ? ` (suggested fix: ${d.fixes[0].description})`
        : ''
      return `${prefix}${location}: ${d.message}${fixHint}`
    })
    .join('\n')
}
