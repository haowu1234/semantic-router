import { describe, it, expect } from 'vitest'
import {
  buildSystemPrompt,
  buildUserPrompt,
  buildRepairPrompt,
  type NLPromptContext,
  type RepairPromptContext,
} from './nlPromptBuilder'
import { NLSchemaRegistry } from './nlSchemaRegistry'

// ─────────────────────────────────────────────
// System Prompt
// ─────────────────────────────────────────────

describe('buildSystemPrompt', () => {
  it('should contain the role definition', () => {
    const prompt = buildSystemPrompt()
    expect(prompt).toContain('Signal DSL configuration expert')
    expect(prompt).toContain('Intent IR')
  })

  it('should contain the DSL type system section', () => {
    const prompt = buildSystemPrompt()
    expect(prompt).toContain('## DSL Type System')
    expect(prompt).toContain('Signal Types')
    expect(prompt).toContain('Plugin Types')
    expect(prompt).toContain('Algorithm Types')
    expect(prompt).toContain('Backend Types')
  })

  it('should contain signal types from registry', () => {
    const prompt = buildSystemPrompt()
    expect(prompt).toContain('keyword')
    expect(prompt).toContain('embedding')
    expect(prompt).toContain('domain')
    expect(prompt).toContain('jailbreak')
    expect(prompt).toContain('pii')
  })

  it('should contain plugin types from registry', () => {
    const prompt = buildSystemPrompt()
    expect(prompt).toContain('semantic_cache')
    expect(prompt).toContain('memory')
    expect(prompt).toContain('system_prompt')
    expect(prompt).toContain('rag')
  })

  it('should contain algorithm types from registry', () => {
    const prompt = buildSystemPrompt()
    expect(prompt).toContain('confidence')
    expect(prompt).toContain('elo')
    expect(prompt).toContain('rl_driven')
    expect(prompt).toContain('remom')
  })

  it('should contain the Intent IR JSON schema', () => {
    const prompt = buildSystemPrompt()
    expect(prompt).toContain('## Intent IR JSON Schema')
    expect(prompt).toContain('IntentIR')
    expect(prompt).toContain('SignalIntent')
    expect(prompt).toContain('RouteIntent')
    expect(prompt).toContain('ConditionNode')
    expect(prompt).toContain('ModelIntent')
    expect(prompt).toContain('PluginTemplateIntent')
    expect(prompt).toContain('BackendIntent')
    expect(prompt).toContain('GlobalIntent')
    expect(prompt).toContain('ModifyIntent')
  })

  it('should contain rules', () => {
    const prompt = buildSystemPrompt()
    expect(prompt).toContain('## Rules')
    expect(prompt).toContain('EVERY signal referenced')
    expect(prompt).toContain('thresholds: 0-1')
    expect(prompt).toContain('valid JSON')
  })

  it('should work with a custom registry', () => {
    const registry = new NLSchemaRegistry()
    registry.register({
      construct: 'signal',
      type_name: 'custom_signal',
      nl_triggers: ['custom'],
      nl_description: 'A custom signal for testing',
      nl_examples: ['test example'],
      fields: [{ key: 'value', label: 'Value', type: 'string', required: true }],
    })
    const prompt = buildSystemPrompt(registry)
    expect(prompt).toContain('custom_signal')
    expect(prompt).toContain('A custom signal for testing')
    // Should NOT contain default types since we used a fresh registry
    expect(prompt).not.toContain('embedding')
  })
})

// ─────────────────────────────────────────────
// User Prompt
// ─────────────────────────────────────────────

describe('buildUserPrompt', () => {
  it('should include user input and mode', () => {
    const ctx: NLPromptContext = {
      userInput: 'Route math questions to GPT-4o',
      mode: 'generate',
    }
    const prompt = buildUserPrompt(ctx)
    expect(prompt).toContain('Route math questions to GPT-4o')
    expect(prompt).toContain('Operation mode: generate')
    expect(prompt).toContain('Generate the Intent IR JSON')
  })

  it('should include few-shot examples', () => {
    const ctx: NLPromptContext = {
      userInput: 'test',
      mode: 'generate',
    }
    const prompt = buildUserPrompt(ctx)
    expect(prompt).toContain('## Examples')
    expect(prompt).toContain('Example 1')
    expect(prompt).toContain('Example 2')
  })

  it('should include context for modify mode', () => {
    const ctx: NLPromptContext = {
      userInput: 'Add PII protection to math route',
      mode: 'modify',
      symbols: {
        signals: [{ name: 'math', type: 'domain' }, { name: 'urgent', type: 'keyword' }],
        routes: ['math_route', 'urgent_route'],
        plugins: ['safe_pii'],
        backends: [{ name: 'ollama', type: 'vllm_endpoint' }],
        models: ['gpt-4o', 'qwen2.5:3b'],
      },
    }
    const prompt = buildUserPrompt(ctx)
    expect(prompt).toContain('Current configuration context')
    expect(prompt).toContain('domain("math")')
    expect(prompt).toContain('keyword("urgent")')
    expect(prompt).toContain('math_route')
    expect(prompt).toContain('safe_pii')
    expect(prompt).toContain('vllm_endpoint("ollama")')
    expect(prompt).toContain('gpt-4o')
  })

  it('should not include context for generate mode even if symbols provided', () => {
    const ctx: NLPromptContext = {
      userInput: 'Create a new config',
      mode: 'generate',
      symbols: {
        signals: [{ name: 'test', type: 'keyword' }],
        routes: ['r1'],
        plugins: [],
        backends: [],
        models: [],
      },
    }
    const prompt = buildUserPrompt(ctx)
    expect(prompt).not.toContain('Current configuration context')
  })

  it('should include diagnostics if provided', () => {
    const ctx: NLPromptContext = {
      userInput: 'Fix the errors',
      mode: 'modify',
      diagnostics: [
        { level: 'error', message: 'signal "math" not defined' },
        { level: 'warning', message: 'threshold out of range' },
      ],
    }
    const prompt = buildUserPrompt(ctx)
    expect(prompt).toContain('Current compilation errors')
    expect(prompt).toContain('[error] signal "math" not defined')
    expect(prompt).toContain('[warning] threshold out of range')
  })

  it('should not include diagnostics section if none provided', () => {
    const ctx: NLPromptContext = {
      userInput: 'test',
      mode: 'generate',
    }
    const prompt = buildUserPrompt(ctx)
    expect(prompt).not.toContain('compilation errors')
  })
})

// ─────────────────────────────────────────────
// Repair Prompt
// ─────────────────────────────────────────────

describe('buildRepairPrompt', () => {
  const baseCtx: RepairPromptContext = {
    originalNL: 'Route math to GPT-4o',
    failedDSL: 'SIGNAL domain mathmatics {\n  description: "Math"\n}',
    diagnostics: [
      { level: 'error', message: 'signal "mathmatics" not defined', line: 1 },
      { level: 'error', message: 'threshold 1.5 out of range [0.0, 1.0]', line: 5 },
    ],
    attempt: 2,
  }

  describe('targeted repair (attempt 2)', () => {
    it('should include error list with line numbers', () => {
      const prompt = buildRepairPrompt({ ...baseCtx, attempt: 2 })
      expect(prompt).toContain('Fix ONLY the errors')
      expect(prompt).toContain('Line 1: signal "mathmatics" not defined')
      expect(prompt).toContain('Line 5: threshold 1.5 out of range')
    })

    it('should include the failed DSL', () => {
      const prompt = buildRepairPrompt({ ...baseCtx, attempt: 2 })
      expect(prompt).toContain('SIGNAL domain mathmatics')
    })

    it('should include original NL', () => {
      const prompt = buildRepairPrompt({ ...baseCtx, attempt: 2 })
      expect(prompt).toContain('Route math to GPT-4o')
    })
  })

  describe('full regeneration (attempt 3)', () => {
    it('should ask for regeneration from scratch', () => {
      const prompt = buildRepairPrompt({ ...baseCtx, attempt: 3 })
      expect(prompt).toContain('regenerate the Intent IR from scratch')
      expect(prompt).toContain('avoiding these mistakes')
    })

    it('should include error summary', () => {
      const prompt = buildRepairPrompt({ ...baseCtx, attempt: 3 })
      expect(prompt).toContain('signal "mathmatics" not defined')
      expect(prompt).toContain('threshold 1.5 out of range')
    })

    it('should include original NL', () => {
      const prompt = buildRepairPrompt({ ...baseCtx, attempt: 3 })
      expect(prompt).toContain('Route math to GPT-4o')
    })

    it('should not include the failed DSL in full regen', () => {
      const prompt = buildRepairPrompt({ ...baseCtx, attempt: 3 })
      expect(prompt).not.toContain('Current DSL')
    })
  })

  it('should handle diagnostics without line numbers', () => {
    const ctx: RepairPromptContext = {
      ...baseCtx,
      diagnostics: [{ level: 'error', message: 'general error' }],
      attempt: 2,
    }
    const prompt = buildRepairPrompt(ctx)
    expect(prompt).toContain('1. general error')
    expect(prompt).not.toContain('Line')
  })
})

// ─────────────────────────────────────────────
// Prompt quality checks
// ─────────────────────────────────────────────

describe('prompt quality', () => {
  it('system prompt should be under 10000 chars (not too long for LLM context)', () => {
    const prompt = buildSystemPrompt()
    // The prompt should be comprehensive but not excessively long
    expect(prompt.length).toBeGreaterThan(1000) // Not trivially short
    expect(prompt.length).toBeLessThan(10000)   // Not excessively long
  })

  it('few-shot examples should cover generate, modify, multi-condition patterns', () => {
    const ctx: NLPromptContext = { userInput: 'test', mode: 'generate' }
    const prompt = buildUserPrompt(ctx)
    // Generate examples
    expect(prompt).toContain('"operation": "generate"')
    // Modify example
    expect(prompt).toContain('"operation": "modify"')
    // AND/OR conditions
    expect(prompt).toContain('"op": "AND"')
    expect(prompt).toContain('"op": "OR"')
    // Signal references
    expect(prompt).toContain('"op": "SIGNAL_REF"')
  })

  it('system prompt should mention NOT to output raw DSL', () => {
    const prompt = buildSystemPrompt()
    expect(prompt).toContain('NOT output raw DSL')
  })
})
