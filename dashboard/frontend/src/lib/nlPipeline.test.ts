import { describe, it, expect, vi } from 'vitest'
import {
  nlToDSL,
  classifyOperation,
  computeConfidence,
  generateExplanation,
  createSession,
  addTurn,
  acceptLastTurn,
  rejectLastTurn,
  MAX_RETRIES,
} from './nlPipeline'
import type { NLContext, NLProgressStep } from './nlPipeline'
import type { LLMClient } from './nlRepair'
import type { WasmBridge, ValidateResult, CompileResult, SymbolTable } from '@/types/dsl'
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
      yaml: 'decisions:\n  - name: test',
      diagnostics: [],
      ...overrides?.compileResult,
    }),
    parseAST: vi.fn().mockReturnValue({ diagnostics: [], errorCount: 0 }),
    decompile: vi.fn().mockReturnValue({ dsl: '' }),
    format: vi.fn().mockReturnValue({ dsl: '' }),
  }
}

const basicIR: IntentIR = {
  version: '1.0',
  operation: 'generate',
  intents: [
    { type: 'signal', signal_type: 'domain', name: 'math', fields: { description: 'Mathematics' } },
    {
      type: 'route',
      name: 'math_route',
      description: 'Math questions',
      priority: 10,
      condition: { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
      models: [{ model: 'gpt-4o' }],
    },
    { type: 'global', fields: { default_model: 'gpt-4o-mini' } },
  ],
}

function createMockLLM(intentIR?: IntentIR): LLMClient {
  return {
    generateIntentIR: vi.fn().mockResolvedValue(intentIR ?? basicIR),
  }
}

// ─── classifyOperation ──────────────────────

describe('classifyOperation', () => {
  it('should return "generate" when no existing DSL', () => {
    expect(classifyOperation('Create a routing config', {})).toBe('generate')
    expect(classifyOperation('Route math to GPT-4o', { currentDSL: '' })).toBe('generate')
    expect(classifyOperation('Route math to GPT-4o', { currentDSL: '   ' })).toBe('generate')
  })

  it('should return "modify" when modification language detected', () => {
    const ctx: NLContext = { currentDSL: 'SIGNAL domain math {}' }

    expect(classifyOperation('Add PII to the math route', ctx)).toBe('modify')
    expect(classifyOperation('Remove the jailbreak signal', ctx)).toBe('modify')
    expect(classifyOperation('Change the threshold to 0.8', ctx)).toBe('modify')
    expect(classifyOperation('Increase priority to 50', ctx)).toBe('modify')
    expect(classifyOperation('Enable semantic caching', ctx)).toBe('modify')
    expect(classifyOperation('Replace the model with Claude', ctx)).toBe('modify')
    expect(classifyOperation('Fix the validation errors', ctx)).toBe('modify')
    expect(classifyOperation('Delete the test route', ctx)).toBe('modify')
    expect(classifyOperation('Update the backend address', ctx)).toBe('modify')
  })

  it('should return "modify" when existing entity names are referenced', () => {
    const ctx: NLContext = {
      currentDSL: 'SIGNAL domain math {}',
      symbols: {
        signals: [{ name: 'math', type: 'domain' }],
        models: [],
        plugins: [],
        backends: [],
        routes: ['math_route'],
      },
    }

    expect(classifyOperation('Make math_route priority 50', ctx)).toBe('modify')
    expect(classifyOperation('Use math signal for routing', ctx)).toBe('modify')
  })

  it('should return "generate" when no modification hints', () => {
    const ctx: NLContext = { currentDSL: 'SIGNAL domain math {}' }
    expect(classifyOperation('Create a new routing config for science', ctx)).toBe('generate')
  })
})

// ─── computeConfidence ──────────────────────

describe('computeConfidence', () => {
  it('should return 1.0 for perfect validation with no retries', () => {
    const validation = {
      isValid: true,
      dsl: '',
      diagnostics: [],
      errorCount: 0,
      warningCount: 0,
      symbolTable: null,
    }
    expect(computeConfidence(validation, 0)).toBe(1.0)
  })

  it('should deduct 0.5 for invalid validation', () => {
    const validation = {
      isValid: false,
      dsl: '',
      diagnostics: [],
      errorCount: 1,
      warningCount: 0,
      symbolTable: null,
    }
    expect(computeConfidence(validation, 0)).toBe(0.5)
  })

  it('should deduct 0.1 per retry', () => {
    const validation = {
      isValid: true,
      dsl: '',
      diagnostics: [],
      errorCount: 0,
      warningCount: 0,
      symbolTable: null,
    }
    expect(computeConfidence(validation, 1)).toBeCloseTo(0.9)
    expect(computeConfidence(validation, 2)).toBeCloseTo(0.8)
    expect(computeConfidence(validation, 3)).toBeCloseTo(0.7)
  })

  it('should deduct 0.05 per warning', () => {
    const validation = {
      isValid: true,
      dsl: '',
      diagnostics: [],
      errorCount: 0,
      warningCount: 2,
      symbolTable: null,
    }
    expect(computeConfidence(validation, 0)).toBeCloseTo(0.9)
  })

  it('should clamp to [0, 1]', () => {
    const validation = {
      isValid: false,
      dsl: '',
      diagnostics: [],
      errorCount: 5,
      warningCount: 20,
      symbolTable: null,
    }
    expect(computeConfidence(validation, 10)).toBe(0)
  })
})

// ─── generateExplanation ────────────────────

describe('generateExplanation', () => {
  it('should explain generate operation', () => {
    const explanation = generateExplanation(basicIR)

    expect(explanation).toContain('Generated configuration')
    expect(explanation).toContain('1 signal')
    expect(explanation).toContain('domain("math")')
    expect(explanation).toContain('1 route')
    expect(explanation).toContain('math_route')
  })

  it('should explain modify operation', () => {
    const modifyIR: IntentIR = {
      version: '1.0',
      operation: 'modify',
      intents: [
        {
          type: 'modify',
          action: 'update',
          target_construct: 'route',
          target_name: 'math_route',
          changes: { priority: 50 },
        },
        {
          type: 'plugin_template',
          name: 'pii_filter',
          plugin_type: 'pii',
          fields: { enabled: true },
        },
      ],
    }

    const explanation = generateExplanation(modifyIR)

    expect(explanation).toContain('Modified configuration')
    expect(explanation).toContain('update route "math_route"')
    expect(explanation).toContain('1 new plugin')
  })

  it('should handle empty intents', () => {
    const ir: IntentIR = { version: '1.0', operation: 'generate', intents: [] }
    const explanation = generateExplanation(ir)
    expect(explanation).toContain('Generated configuration')
  })

  it('should pluralize correctly', () => {
    const ir: IntentIR = {
      version: '1.0',
      operation: 'generate',
      intents: [
        { type: 'signal', signal_type: 'domain', name: 'math', fields: {} },
        { type: 'signal', signal_type: 'domain', name: 'physics', fields: {} },
        {
          type: 'route',
          name: 'r1',
          priority: 10,
          condition: { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
          models: [{ model: 'gpt-4o' }],
        },
      ],
    }

    const explanation = generateExplanation(ir)
    expect(explanation).toContain('2 signals')
    expect(explanation).toContain('1 route:')
  })
})

// ─── Session Management ─────────────────────

describe('Session Management', () => {
  it('should create a new session with unique ID', () => {
    const s1 = createSession()
    const s2 = createSession()

    expect(s1.id).toMatch(/^nl-\d+-[a-z0-9]+$/)
    expect(s1.id).not.toBe(s2.id)
    expect(s1.turns).toEqual([])
    expect(s1.currentDSL).toBe('')
    expect(s1.symbolTable).toBeNull()
  })

  it('should add a turn immutably', () => {
    const session = createSession()
    const ir: IntentIR = { version: '1.0', operation: 'generate', intents: [] }

    const updated = addTurn(session, 'test input', ir, 'SIGNAL domain math {}')

    expect(session.turns).toHaveLength(0) // original unchanged
    expect(updated.turns).toHaveLength(1)
    expect(updated.turns[0].userInput).toBe('test input')
    expect(updated.turns[0].accepted).toBe(false)
  })

  it('should accept last turn and update DSL + symbols', () => {
    const session = createSession()
    const ir: IntentIR = { version: '1.0', operation: 'generate', intents: [] }
    const withTurn = addTurn(session, 'test', ir, 'DSL text')

    const symbols: SymbolTable = {
      signals: [{ name: 'math', type: 'domain' }],
      models: ['gpt-4o'],
      plugins: [],
      backends: [],
      routes: ['math_route'],
    }

    const accepted = acceptLastTurn(withTurn, 'DSL text', symbols)

    expect(accepted.turns[0].accepted).toBe(true)
    expect(accepted.currentDSL).toBe('DSL text')
    expect(accepted.symbolTable).toBe(symbols)
  })

  it('should reject last turn', () => {
    const session = createSession()
    const ir: IntentIR = { version: '1.0', operation: 'generate', intents: [] }
    const withTurn = addTurn(session, 'test', ir, 'DSL')

    const rejected = rejectLastTurn(withTurn)

    expect(rejected.turns[0].accepted).toBe(false)
  })

  it('should handle accept/reject on empty session gracefully', () => {
    const session = createSession()
    expect(() => acceptLastTurn(session, '', null)).not.toThrow()
    expect(() => rejectLastTurn(session)).not.toThrow()
  })
})

// ─── nlToDSL (Integration) ──────────────────

describe('nlToDSL', () => {
  it('should run the full happy path pipeline', async () => {
    const wasm = createMockWasm()
    const llm = createMockLLM()

    const result = await nlToDSL(
      'Route math questions to GPT-4o',
      {},
      wasm,
      llm,
    )

    expect(result.isValid).toBe(true)
    expect(result.dsl).toContain('SIGNAL domain math')
    expect(result.dsl).toContain('ROUTE math_route')
    expect(result.dsl).toContain('GLOBAL')
    expect(result.yaml).toBeTruthy()
    expect(result.confidence).toBe(1.0)
    expect(result.retries).toBe(0)
    expect(result.explanation).toContain('Generated configuration')
    expect(result.intentIR).toEqual(basicIR)
  })

  it('should report progress via callback', async () => {
    const wasm = createMockWasm()
    const llm = createMockLLM()
    const steps: NLProgressStep[] = []

    await nlToDSL(
      'Route math to GPT-4o',
      {},
      wasm,
      llm,
      (step) => steps.push(step),
    )

    const stages = steps.map(s => s.stage)
    expect(stages).toContain('classifying')
    expect(stages).toContain('generating')
    expect(stages).toContain('validating')
    expect(stages).toContain('done')
  })

  it('should trigger repair loop when validation fails then succeeds', async () => {
    let callCount = 0
    const wasm = createMockWasm()
    // First validate fails, second succeeds
    ;(wasm.validate as ReturnType<typeof vi.fn>).mockImplementation(() => {
      callCount++
      if (callCount === 1) {
        return {
          diagnostics: [{
            level: 'error',
            message: 'Unknown type',
            line: 1,
            column: 8,
            fixes: [{ description: 'Fix', newText: 'domain' }],
          }],
          errorCount: 1,
          symbols: { signals: [], models: [], plugins: [], backends: [], routes: [] },
        }
      }
      return { diagnostics: [], errorCount: 0, symbols: { signals: [], models: [], plugins: [], backends: [], routes: [] } }
    })

    const llm = createMockLLM()
    const result = await nlToDSL('Route math to GPT-4o', {}, wasm, llm)

    expect(result.retries).toBe(1)
    expect(result.confidence).toBeCloseTo(0.9) // 1.0 - 0.1 for 1 retry
  })

  it('should stop after MAX_RETRIES', async () => {
    const wasm = createMockWasm({
      validateResult: {
        diagnostics: [{ level: 'error', message: 'Persistent error', line: 1, column: 1 }],
        errorCount: 1,
      },
    })

    const llm = createMockLLM()
    const result = await nlToDSL('test', {}, wasm, llm)

    expect(result.retries).toBe(MAX_RETRIES)
    expect(result.isValid).toBe(false)
    expect(result.confidence).toBeLessThan(0.5)
  })

  it('should throw when LLM generation fails', async () => {
    const wasm = createMockWasm()
    const llm = createMockLLM()
    ;(llm.generateIntentIR as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('API down'))

    await expect(
      nlToDSL('test', {}, wasm, llm),
    ).rejects.toThrow('LLM generation failed')
  })

  it('should use modify mode when context has DSL and input suggests modification', async () => {
    const wasm = createMockWasm()

    const modifyIR: IntentIR = {
      version: '1.0',
      operation: 'modify',
      intents: [{
        type: 'modify',
        action: 'update',
        target_construct: 'signal',
        target_name: 'math',
        target_signal_type: 'domain',
        changes: { description: 'Advanced Math' },
      }],
    }
    const llm = createMockLLM(modifyIR)

    const context: NLContext = {
      currentDSL: 'SIGNAL domain math { description: "Math" }',
      symbols: {
        signals: [{ name: 'math', type: 'domain' }],
        models: [],
        plugins: [],
        backends: [],
        routes: [],
      },
    }

    const result = await nlToDSL(
      'Change the math description to Advanced Math',
      context,
      wasm,
      llm,
    )

    // LLM was called with the user prompt (which should include context)
    const callArgs = (llm.generateIntentIR as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(callArgs[1]).toContain('modify')
    expect(result.dsl).toBeTruthy()
  })

  it('should report error progress on LLM failure', async () => {
    const wasm = createMockWasm()
    const llm = createMockLLM()
    ;(llm.generateIntentIR as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('timeout'))

    const steps: NLProgressStep[] = []
    await nlToDSL('test', {}, wasm, llm, s => steps.push(s)).catch(() => {})

    const errorStep = steps.find(s => s.stage === 'error')
    expect(errorStep).toBeDefined()
  })
})
