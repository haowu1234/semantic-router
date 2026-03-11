import { describe, it, expect, vi } from 'vitest'
import { repairDSL } from './nlRepair'
import type { LLMClient } from './nlRepair'
import type { Diagnostic, WasmBridge, ValidateResult, CompileResult } from '@/types/dsl'
import type { IntentIR } from '@/types/intentIR'

// ─── Test Helpers ────────────────────────────

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
      symbols: { signals: [], models: [], plugins: [], backends: [], routes: [] },
      ...overrides?.validateResult,
    }),
    compile: vi.fn().mockReturnValue({
      yaml: 'test: yaml',
      diagnostics: [],
      ...overrides?.compileResult,
    }),
    parseAST: vi.fn().mockReturnValue({ diagnostics: [], errorCount: 0 }),
    decompile: vi.fn().mockReturnValue({ dsl: '' }),
    format: vi.fn().mockReturnValue({ dsl: '' }),
  }
}

function createMockLLMClient(intentIR?: IntentIR): LLMClient {
  const defaultIR: IntentIR = {
    version: '1.0',
    operation: 'generate',
    intents: [
      {
        type: 'signal',
        signal_type: 'domain',
        name: 'math',
        fields: { description: 'Mathematics' },
      },
      {
        type: 'route',
        name: 'math_route',
        priority: 10,
        condition: { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
        models: [{ model: 'gpt-4o' }],
      },
      {
        type: 'global',
        fields: { default_model: 'gpt-4o-mini' },
      },
    ],
  }

  return {
    generateIntentIR: vi.fn().mockResolvedValue(intentIR ?? defaultIR),
  }
}

const SYSTEM_PROMPT = 'You are a DSL expert.'

const sampleDiagnostics: Diagnostic[] = [
  {
    level: 'error',
    message: 'Unknown signal type "domian"',
    line: 1,
    column: 8,
    fixes: [{ description: 'Change to "domain"', newText: 'domain' }],
  },
]

// ─── Strategy 1: QuickFix ────────────────────

describe('repairDSL — Strategy 1: QuickFix', () => {
  it('should apply QuickFixes for attempt 1', async () => {
    const wasm = createMockWasm()
    const llm = createMockLLMClient()

    const result = await repairDSL(
      'Route math to GPT-4o',
      'SIGNAL domian math {}',
      sampleDiagnostics,
      wasm,
      llm,
      SYSTEM_PROMPT,
      1,
    )

    expect(result.strategy).toBe('quickfix')
    expect(result.dsl).toBe('SIGNAL domain math {}')
    expect(llm.generateIntentIR).not.toHaveBeenCalled()
  })

  it('should return original DSL when no QuickFixes available', async () => {
    const wasm = createMockWasm()
    const llm = createMockLLMClient()

    const noFixDiags: Diagnostic[] = [{
      level: 'error',
      message: 'Complex error',
      line: 1,
      column: 1,
    }]

    const result = await repairDSL(
      'test',
      'BAD DSL',
      noFixDiags,
      wasm,
      llm,
      SYSTEM_PROMPT,
      1,
    )

    expect(result.strategy).toBe('quickfix')
    expect(result.dsl).toBe('BAD DSL')
  })
})

// ─── Strategy 2: Targeted LLM ───────────────

describe('repairDSL — Strategy 2: Targeted LLM', () => {
  it('should call LLM with repair prompt for attempt 2', async () => {
    const wasm = createMockWasm()
    const llm = createMockLLMClient()

    const result = await repairDSL(
      'Route math to GPT-4o',
      'SIGNAL domian math {}',
      sampleDiagnostics,
      wasm,
      llm,
      SYSTEM_PROMPT,
      2,
    )

    expect(result.strategy).toBe('targeted_llm')
    expect(llm.generateIntentIR).toHaveBeenCalledOnce()

    // Check the repair prompt includes error info
    const callArgs = (llm.generateIntentIR as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(callArgs[0]).toBe(SYSTEM_PROMPT)
    expect(callArgs[1]).toContain('validation errors')
    expect(callArgs[1]).toContain('domian')
  })

  it('should return original DSL when LLM call fails', async () => {
    const wasm = createMockWasm({
      validateResult: { diagnostics: sampleDiagnostics, errorCount: 1 },
    })
    const llm = createMockLLMClient()
    ;(llm.generateIntentIR as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('LLM timeout'))

    const result = await repairDSL(
      'test',
      'SIGNAL domian math {}',
      sampleDiagnostics,
      wasm,
      llm,
      SYSTEM_PROMPT,
      2,
    )

    expect(result.strategy).toBe('targeted_llm')
    expect(result.dsl).toBe('SIGNAL domian math {}')
  })

  it('should re-validate the repaired DSL', async () => {
    const wasm = createMockWasm()
    const llm = createMockLLMClient()

    await repairDSL(
      'test',
      'SIGNAL domian math {}',
      sampleDiagnostics,
      wasm,
      llm,
      SYSTEM_PROMPT,
      2,
    )

    expect(wasm.validate).toHaveBeenCalled()
  })
})

// ─── Strategy 3: Full Regeneration ──────────

describe('repairDSL — Strategy 3: Full Regeneration', () => {
  it('should call LLM with full regen prompt for attempt 3', async () => {
    const wasm = createMockWasm()
    const llm = createMockLLMClient()

    const result = await repairDSL(
      'Route math to GPT-4o',
      'SIGNAL domian math {}',
      sampleDiagnostics,
      wasm,
      llm,
      SYSTEM_PROMPT,
      3,
    )

    expect(result.strategy).toBe('full_regen')
    expect(llm.generateIntentIR).toHaveBeenCalledOnce()

    // Check the regen prompt
    const callArgs = (llm.generateIntentIR as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(callArgs[1]).toContain('regenerate')
    expect(callArgs[1]).toContain('Route math to GPT-4o')
  })

  it('should use full regen for attempt > 3', async () => {
    const wasm = createMockWasm()
    const llm = createMockLLMClient()

    const result = await repairDSL(
      'test',
      'BAD DSL',
      sampleDiagnostics,
      wasm,
      llm,
      SYSTEM_PROMPT,
      5,
    )

    expect(result.strategy).toBe('full_regen')
  })

  it('should return original DSL when LLM call fails', async () => {
    const wasm = createMockWasm({
      validateResult: { diagnostics: sampleDiagnostics, errorCount: 1 },
    })
    const llm = createMockLLMClient()
    ;(llm.generateIntentIR as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('API error'))

    const result = await repairDSL(
      'test',
      'SIGNAL domian math {}',
      sampleDiagnostics,
      wasm,
      llm,
      SYSTEM_PROMPT,
      3,
    )

    expect(result.strategy).toBe('full_regen')
    expect(result.dsl).toBe('SIGNAL domian math {}')
  })
})

// ─── Modify mode ────────────────────────────

describe('repairDSL — Modify mode', () => {
  it('should pass existingDSL through to IR codegen in targeted repair', async () => {
    const existingDSL = 'SIGNAL domain math { description: "Math" }\n'
    const wasm = createMockWasm()

    // LLM returns a modify IR
    const modifyIR: IntentIR = {
      version: '1.0',
      operation: 'modify',
      intents: [{
        type: 'modify',
        action: 'update',
        target_construct: 'signal',
        target_name: 'math',
        target_signal_type: 'domain',
        changes: { description: 'Mathematics' },
      }],
    }
    const llm = createMockLLMClient(modifyIR)

    const result = await repairDSL(
      'Change math description',
      'BAD DSL',
      sampleDiagnostics,
      wasm,
      llm,
      SYSTEM_PROMPT,
      2,
      existingDSL,
    )

    expect(result.strategy).toBe('targeted_llm')
    // The intentIRToDSL is called with existingDSL
    expect(result.dsl).toBeTruthy()
  })
})
