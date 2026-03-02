import { describe, it, expect, vi } from 'vitest'
import {
  validateGeneratedDSL,
  applyQuickFixes,
  filterErrors,
  filterWarnings,
  hasQuickFixes,
  formatDiagnostics,
} from './nlValidation'
import type { ValidationResult } from './nlValidation'
import type { Diagnostic, WasmBridge, ValidateResult, CompileResult } from '@/types/dsl'

// ─── Mock WASM Bridge ────────────────────────

function createMockWasm(overrides?: {
  validateResult?: Partial<ValidateResult>
  compileResult?: Partial<CompileResult>
}): WasmBridge {
  return {
    ready: true,
    init: vi.fn().mockResolvedValue(undefined),
    validate: vi.fn().mockReturnValue({
      diagnostics: [],
      errorCount: 0,
      symbols: {
        signals: [{ name: 'math', type: 'domain' }],
        models: ['gpt-4o'],
        plugins: [],
        backends: [],
        routes: ['math_route'],
      },
      ...overrides?.validateResult,
    }),
    compile: vi.fn().mockReturnValue({
      yaml: 'decisions:\n  - name: test',
      diagnostics: [],
      ...overrides?.compileResult,
    }),
    parseAST: vi.fn().mockReturnValue({ diagnostics: [], errorCount: 0 }),
    decompile: vi.fn().mockReturnValue({ dsl: '' }),
    format: vi.fn().mockReturnValue({ dsl: '' }),
  }
}

// ─── validateGeneratedDSL ────────────────────

describe('validateGeneratedDSL', () => {
  it('should return valid result for empty DSL', () => {
    const wasm = createMockWasm()
    const result = validateGeneratedDSL('', wasm)

    expect(result.isValid).toBe(true)
    expect(result.errorCount).toBe(0)
    expect(result.warningCount).toBe(0)
    expect(result.diagnostics).toEqual([])
    expect(result.symbolTable).toBeNull()
  })

  it('should return valid result for whitespace-only DSL', () => {
    const wasm = createMockWasm()
    const result = validateGeneratedDSL('   \n  \n  ', wasm)

    expect(result.isValid).toBe(true)
    expect(wasm.validate).not.toHaveBeenCalled()
  })

  it('should validate DSL and compile on success', () => {
    const wasm = createMockWasm()
    const dsl = 'SIGNAL domain math { description: "Math" }'
    const result = validateGeneratedDSL(dsl, wasm)

    expect(result.isValid).toBe(true)
    expect(result.errorCount).toBe(0)
    expect(result.symbolTable?.signals).toEqual([{ name: 'math', type: 'domain' }])
    expect(result.yaml).toBe('decisions:\n  - name: test')
    expect(wasm.validate).toHaveBeenCalledWith(dsl)
    expect(wasm.compile).toHaveBeenCalledWith(dsl)
  })

  it('should return invalid result when errors exist', () => {
    const wasm = createMockWasm({
      validateResult: {
        diagnostics: [
          { level: 'error', message: 'Unknown signal type "foo"', line: 1, column: 8 },
        ],
        errorCount: 1,
      },
    })

    const result = validateGeneratedDSL('SIGNAL foo bar {}', wasm)

    expect(result.isValid).toBe(false)
    expect(result.errorCount).toBe(1)
    expect(result.yaml).toBeUndefined()
    expect(wasm.compile).not.toHaveBeenCalled()
  })

  it('should count warnings separately from errors', () => {
    const wasm = createMockWasm({
      validateResult: {
        diagnostics: [
          { level: 'warning', message: 'Unused signal "math"', line: 1, column: 1 },
          { level: 'warning', message: 'Low threshold', line: 2, column: 1 },
        ],
        errorCount: 0,
      },
    })

    const result = validateGeneratedDSL('SIGNAL domain math {}', wasm)

    expect(result.isValid).toBe(true)
    expect(result.errorCount).toBe(0)
    expect(result.warningCount).toBe(2)
  })

  it('should handle WASM validate error', () => {
    const wasm = createMockWasm({
      validateResult: {
        error: 'WASM panic',
        diagnostics: [],
        errorCount: 0,
      },
    })

    const result = validateGeneratedDSL('BAD DSL', wasm)

    expect(result.isValid).toBe(false)
    expect(result.errorCount).toBeGreaterThan(0)
  })

  it('should handle compilation failure gracefully', () => {
    const wasm = createMockWasm({
      compileResult: {
        error: 'Compilation failed',
        yaml: '',
        diagnostics: [{ level: 'error', message: 'internal error', line: 0, column: 0 }],
      },
    })

    const result = validateGeneratedDSL('SIGNAL domain math {}', wasm)

    // Validation passes but YAML is not produced
    expect(result.isValid).toBe(true)
    expect(result.yaml).toBeUndefined()
  })

  it('should handle compile throwing an exception', () => {
    const wasm = createMockWasm()
    ;(wasm.compile as ReturnType<typeof vi.fn>).mockImplementation(() => {
      throw new Error('WASM crash')
    })

    const result = validateGeneratedDSL('SIGNAL domain math {}', wasm)

    expect(result.isValid).toBe(true)
    expect(result.yaml).toBeUndefined()
  })

  it('should pass dsl through in result', () => {
    const wasm = createMockWasm()
    const dsl = 'SIGNAL domain math { description: "test" }'
    const result = validateGeneratedDSL(dsl, wasm)

    expect(result.dsl).toBe(dsl)
  })
})

// ─── applyQuickFixes ─────────────────────────

describe('applyQuickFixes', () => {
  it('should return original DSL when no fixes available', () => {
    const diagnostics: Diagnostic[] = [
      { level: 'error', message: 'Unknown type', line: 1, column: 8 },
    ]
    const dsl = 'SIGNAL foo bar {}'
    expect(applyQuickFixes(dsl, diagnostics)).toBe(dsl)
  })

  it('should return original DSL when diagnostics is empty', () => {
    const dsl = 'SIGNAL domain math {}'
    expect(applyQuickFixes(dsl, [])).toBe(dsl)
  })

  it('should apply single QuickFix', () => {
    const dsl = 'SIGNAL domian math {}'
    const diagnostics: Diagnostic[] = [{
      level: 'error',
      message: 'Unknown signal type "domian" (did you mean "domain"?)',
      line: 1,
      column: 8,
      fixes: [{ description: 'Change to "domain"', newText: 'domain' }],
    }]

    const result = applyQuickFixes(dsl, diagnostics)
    expect(result).toBe('SIGNAL domain math {}')
  })

  it('should apply multiple QuickFixes on different lines', () => {
    const dsl = 'SIGNAL domian math {}\nSIGNAL keywrd urgent {}'
    const diagnostics: Diagnostic[] = [
      {
        level: 'error',
        message: 'Unknown type',
        line: 1,
        column: 8,
        fixes: [{ description: 'Fix', newText: 'domain' }],
      },
      {
        level: 'error',
        message: 'Unknown type',
        line: 2,
        column: 8,
        fixes: [{ description: 'Fix', newText: 'keyword' }],
      },
    ]

    const result = applyQuickFixes(dsl, diagnostics)
    expect(result).toBe('SIGNAL domain math {}\nSIGNAL keyword urgent {}')
  })

  it('should handle out-of-range line numbers gracefully', () => {
    const dsl = 'SIGNAL domain math {}'
    const diagnostics: Diagnostic[] = [{
      level: 'error',
      message: 'test',
      line: 99,
      column: 1,
      fixes: [{ description: 'Fix', newText: 'domain' }],
    }]

    expect(applyQuickFixes(dsl, diagnostics)).toBe(dsl)
  })

  it('should handle out-of-range column numbers gracefully', () => {
    const dsl = 'SIGNAL domain math {}'
    const diagnostics: Diagnostic[] = [{
      level: 'error',
      message: 'test',
      line: 1,
      column: 999,
      fixes: [{ description: 'Fix', newText: 'fixed' }],
    }]

    expect(applyQuickFixes(dsl, diagnostics)).toBe(dsl)
  })

  it('should apply fixes from bottom to top (reverse order)', () => {
    const dsl = 'line1_bad\nline2_bad\nline3_bad'
    const diagnostics: Diagnostic[] = [
      { level: 'error', message: '', line: 1, column: 1, fixes: [{ description: '', newText: 'line1_good' }] },
      { level: 'error', message: '', line: 3, column: 1, fixes: [{ description: '', newText: 'line3_good' }] },
    ]

    const result = applyQuickFixes(dsl, diagnostics)
    expect(result).toBe('line1_good\nline2_bad\nline3_good')
  })
})

// ─── filterErrors / filterWarnings ───────────

describe('filterErrors', () => {
  it('should return only error-level diagnostics', () => {
    const diags: Diagnostic[] = [
      { level: 'error', message: 'err1', line: 1, column: 1 },
      { level: 'warning', message: 'warn1', line: 2, column: 1 },
      { level: 'error', message: 'err2', line: 3, column: 1 },
      { level: 'constraint', message: 'con1', line: 4, column: 1 },
    ]

    const errors = filterErrors(diags)
    expect(errors).toHaveLength(2)
    expect(errors.map(e => e.message)).toEqual(['err1', 'err2'])
  })

  it('should return empty array when no errors', () => {
    const diags: Diagnostic[] = [
      { level: 'warning', message: 'warn1', line: 1, column: 1 },
    ]
    expect(filterErrors(diags)).toHaveLength(0)
  })
})

describe('filterWarnings', () => {
  it('should return only warning-level diagnostics', () => {
    const diags: Diagnostic[] = [
      { level: 'error', message: 'err1', line: 1, column: 1 },
      { level: 'warning', message: 'warn1', line: 2, column: 1 },
      { level: 'warning', message: 'warn2', line: 3, column: 1 },
    ]

    const warnings = filterWarnings(diags)
    expect(warnings).toHaveLength(2)
    expect(warnings.map(w => w.message)).toEqual(['warn1', 'warn2'])
  })
})

// ─── hasQuickFixes ───────────────────────────

describe('hasQuickFixes', () => {
  it('should return true when fixes exist', () => {
    const diags: Diagnostic[] = [{
      level: 'error',
      message: 'test',
      line: 1,
      column: 1,
      fixes: [{ description: 'fix it', newText: 'fixed' }],
    }]
    expect(hasQuickFixes(diags)).toBe(true)
  })

  it('should return false when no fixes', () => {
    const diags: Diagnostic[] = [{
      level: 'error',
      message: 'test',
      line: 1,
      column: 1,
    }]
    expect(hasQuickFixes(diags)).toBe(false)
  })

  it('should return false for empty array', () => {
    expect(hasQuickFixes([])).toBe(false)
  })

  it('should return false when fixes array is empty', () => {
    const diags: Diagnostic[] = [{
      level: 'error',
      message: 'test',
      line: 1,
      column: 1,
      fixes: [],
    }]
    expect(hasQuickFixes(diags)).toBe(false)
  })
})

// ─── formatDiagnostics ──────────────────────

describe('formatDiagnostics', () => {
  it('should format diagnostics with line numbers', () => {
    const diags: Diagnostic[] = [
      { level: 'error', message: 'Unknown signal', line: 5, column: 1 },
      { level: 'warning', message: 'Unused', line: 10, column: 1 },
    ]

    const formatted = formatDiagnostics(diags)
    expect(formatted).toContain('1. [error] Line 5: Unknown signal')
    expect(formatted).toContain('2. [warning] Line 10: Unused')
  })

  it('should include QuickFix hints', () => {
    const diags: Diagnostic[] = [{
      level: 'error',
      message: 'Unknown type',
      line: 1,
      column: 8,
      fixes: [{ description: 'Change to "domain"', newText: 'domain' }],
    }]

    const formatted = formatDiagnostics(diags)
    expect(formatted).toContain('(suggested fix: Change to "domain")')
  })

  it('should handle empty diagnostics', () => {
    expect(formatDiagnostics([])).toBe('')
  })

  it('should handle diagnostics without line numbers', () => {
    const diags: Diagnostic[] = [{
      level: 'error',
      message: 'Global error',
      line: 0,
      column: 0,
    }]

    const formatted = formatDiagnostics(diags)
    // line 0 is treated as no line
    expect(formatted).toContain('[error]')
    expect(formatted).toContain('Global error')
  })
})
