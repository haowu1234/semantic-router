import { describe, it, expect } from 'vitest'
import {
  intentIRToDSL,
  emitCondition,
  sanitizeName,
  collectSignalRefs,
  resolveImplicitDependencies,
} from './intentToDsl'
import type {
  IntentIR,
  SignalIntent,
  RouteIntent,
  PluginTemplateIntent,
  BackendIntent,
  GlobalIntent,
  ModifyIntent,
  ConditionNode,
} from '../types/intentIR'

// ─────────────────────────────────────────────
// Helper: build minimal IR
// ─────────────────────────────────────────────

function makeIR(intents: IntentIR['intents'], operation: IntentIR['operation'] = 'generate'): IntentIR {
  return { version: '1.0', operation, intents }
}

// ─────────────────────────────────────────────
// sanitizeName
// ─────────────────────────────────────────────

describe('sanitizeName', () => {
  it('should keep valid identifiers unchanged', () => {
    expect(sanitizeName('my_signal')).toBe('my_signal')
    expect(sanitizeName('route-1')).toBe('route-1')
  })

  it('should replace spaces and special chars with _', () => {
    expect(sanitizeName('my signal')).toBe('my_signal')
    expect(sanitizeName('hello@world!')).toBe('hello_world_')
  })

  it('should handle empty string', () => {
    expect(sanitizeName('')).toBe('')
  })
})

// ─────────────────────────────────────────────
// emitCondition
// ─────────────────────────────────────────────

describe('emitCondition', () => {
  it('should emit SIGNAL_REF', () => {
    const node: ConditionNode = { op: 'SIGNAL_REF', signal_type: 'keyword', signal_name: 'urgent' }
    expect(emitCondition(node)).toBe('keyword("urgent")')
  })

  it('should emit AND', () => {
    const node: ConditionNode = {
      op: 'AND',
      operands: [
        { op: 'SIGNAL_REF', signal_type: 'keyword', signal_name: 'a' },
        { op: 'SIGNAL_REF', signal_type: 'embedding', signal_name: 'b' },
      ],
    }
    expect(emitCondition(node)).toBe('keyword("a") AND embedding("b")')
  })

  it('should emit OR', () => {
    const node: ConditionNode = {
      op: 'OR',
      operands: [
        { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
        { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'code' },
      ],
    }
    expect(emitCondition(node)).toBe('domain("math") OR domain("code")')
  })

  it('should emit NOT', () => {
    const node: ConditionNode = {
      op: 'NOT',
      operand: { op: 'SIGNAL_REF', signal_type: 'complexity', signal_name: 'simple' },
    }
    expect(emitCondition(node)).toBe('NOT complexity("simple")')
  })

  it('should wrap OR inside AND with parens', () => {
    const node: ConditionNode = {
      op: 'AND',
      operands: [
        {
          op: 'OR',
          operands: [
            { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
            { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'physics' },
          ],
        },
        { op: 'SIGNAL_REF', signal_type: 'complexity', signal_name: 'hard' },
      ],
    }
    expect(emitCondition(node)).toBe('(domain("math") OR domain("physics")) AND complexity("hard")')
  })

  it('should wrap compound expression after NOT with parens', () => {
    const node: ConditionNode = {
      op: 'NOT',
      operand: {
        op: 'AND',
        operands: [
          { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'a' },
          { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'b' },
        ],
      },
    }
    expect(emitCondition(node)).toBe('NOT (domain("a") AND domain("b"))')
  })

  it('should handle deeply nested expressions', () => {
    // WHEN (A OR B) AND NOT C
    const node: ConditionNode = {
      op: 'AND',
      operands: [
        {
          op: 'OR',
          operands: [
            { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
            { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'physics' },
          ],
        },
        {
          op: 'NOT',
          operand: { op: 'SIGNAL_REF', signal_type: 'language', signal_name: 'chinese' },
        },
      ],
    }
    expect(emitCondition(node)).toBe(
      '(domain("math") OR domain("physics")) AND NOT language("chinese")'
    )
  })
})

// ─────────────────────────────────────────────
// collectSignalRefs
// ─────────────────────────────────────────────

describe('collectSignalRefs', () => {
  it('should collect refs from simple condition', () => {
    const node: ConditionNode = { op: 'SIGNAL_REF', signal_type: 'keyword', signal_name: 'test' }
    expect(collectSignalRefs(node)).toEqual([{ signal_type: 'keyword', signal_name: 'test' }])
  })

  it('should collect refs from nested AND/OR/NOT', () => {
    const node: ConditionNode = {
      op: 'AND',
      operands: [
        {
          op: 'OR',
          operands: [
            { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
            { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'code' },
          ],
        },
        {
          op: 'NOT',
          operand: { op: 'SIGNAL_REF', signal_type: 'complexity', signal_name: 'simple' },
        },
      ],
    }
    const refs = collectSignalRefs(node)
    expect(refs).toHaveLength(3)
    expect(refs).toContainEqual({ signal_type: 'domain', signal_name: 'math' })
    expect(refs).toContainEqual({ signal_type: 'domain', signal_name: 'code' })
    expect(refs).toContainEqual({ signal_type: 'complexity', signal_name: 'simple' })
  })
})

// ─────────────────────────────────────────────
// Signal emission
// ─────────────────────────────────────────────

describe('emitSignal (via intentIRToDSL)', () => {
  it('should emit a keyword signal', () => {
    const ir = makeIR([
      {
        type: 'signal',
        signal_type: 'keyword',
        name: 'urgent',
        fields: { operator: 'any', keywords: ['urgent', 'asap'], method: 'regex', case_sensitive: false },
      } as SignalIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('SIGNAL keyword urgent {')
    expect(dsl).toContain('operator: "any"')
    expect(dsl).toContain('keywords: ["urgent", "asap"]')
    expect(dsl).toContain('method: "regex"')
    expect(dsl).toContain('case_sensitive: false')
    expect(dsl).toContain('}')
  })

  it('should emit an embedding signal', () => {
    const ir = makeIR([
      {
        type: 'signal',
        signal_type: 'embedding',
        name: 'ai_topics',
        fields: { threshold: 0.75, candidates: ['ML', 'DL'], aggregation_method: 'max' },
      } as SignalIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('SIGNAL embedding ai_topics {')
    expect(dsl).toContain('threshold: 0.75')
    expect(dsl).toContain('candidates: ["ML", "DL"]')
  })

  it('should sanitize signal names with spaces', () => {
    const ir = makeIR([
      {
        type: 'signal',
        signal_type: 'domain',
        name: 'my signal name',
        fields: { description: 'test' },
      } as SignalIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('SIGNAL domain my_signal_name {')
  })
})

// ─────────────────────────────────────────────
// Plugin template emission
// ─────────────────────────────────────────────

describe('emitPluginTemplate (via intentIRToDSL)', () => {
  it('should emit PLUGIN <name> <type> (name before type)', () => {
    const ir = makeIR([
      {
        type: 'plugin_template',
        name: 'safe_pii',
        plugin_type: 'pii',
        fields: { enabled: true, pii_types_allowed: [] },
      } as PluginTemplateIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('PLUGIN safe_pii pii {')
    expect(dsl).toContain('enabled: true')
    expect(dsl).toContain('pii_types_allowed: []')
  })
})

// ─────────────────────────────────────────────
// Route emission
// ─────────────────────────────────────────────

describe('emitRoute (via intentIRToDSL)', () => {
  it('should emit a basic route with all clauses', () => {
    const ir = makeIR([
      {
        type: 'route',
        name: 'math_route',
        description: 'Math questions',
        priority: 100,
        condition: { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
        models: [{ model: 'gpt-4o', reasoning: false }],
        algorithm: { algo_type: 'confidence', params: { threshold: 0.8, confidence_method: 'hybrid' } },
        plugins: [{ name: 'safe_pii' }],
      } as RouteIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('ROUTE math_route (description = "Math questions") {')
    expect(dsl).toContain('PRIORITY 100')
    expect(dsl).toContain('WHEN domain("math")')
    expect(dsl).toContain('MODEL "gpt-4o" (reasoning = false)')
    expect(dsl).toContain('ALGORITHM confidence {')
    expect(dsl).toContain('threshold: 0.8')
    expect(dsl).toContain('confidence_method: "hybrid"')
    expect(dsl).toContain('PLUGIN safe_pii')
  })

  it('should emit multi-model references with comma separation', () => {
    const ir = makeIR([
      {
        type: 'route',
        name: 'multi_model',
        priority: 10,
        condition: { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'test' },
        models: [
          { model: 'qwen3:70b', reasoning: true, effort: 'high' },
          { model: 'qwen2.5:3b', reasoning: false },
        ],
      } as RouteIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('"qwen3:70b" (reasoning = true, effort = "high")')
    expect(dsl).toContain('"qwen2.5:3b" (reasoning = false)')
  })

  it('should emit route without description', () => {
    const ir = makeIR([
      {
        type: 'route',
        name: 'no_desc',
        priority: 5,
        condition: { op: 'SIGNAL_REF', signal_type: 'keyword', signal_name: 'test' },
        models: [{ model: 'gpt-4o' }],
      } as RouteIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('ROUTE no_desc {')
    expect(dsl).not.toContain('description')
  })

  it('should emit plugin with overrides', () => {
    const ir = makeIR([
      {
        type: 'route',
        name: 'test_route',
        priority: 10,
        condition: { op: 'SIGNAL_REF', signal_type: 'keyword', signal_name: 'a' },
        models: [{ model: 'gpt-4o' }],
        plugins: [{ name: 'my_cache', overrides: { enabled: true, max_entries: 5000 } }],
      } as RouteIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('PLUGIN my_cache {')
    expect(dsl).toContain('enabled: true')
    expect(dsl).toContain('max_entries: 5000')
  })

  it('should emit algorithm with no params as empty braces', () => {
    const ir = makeIR([
      {
        type: 'route',
        name: 'static_route',
        priority: 10,
        condition: { op: 'SIGNAL_REF', signal_type: 'keyword', signal_name: 'a' },
        models: [{ model: 'gpt-4o' }],
        algorithm: { algo_type: 'static', params: {} },
      } as RouteIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('ALGORITHM static {}')
  })

  it('should emit model with all attributes', () => {
    const ir = makeIR([
      {
        type: 'route',
        name: 'full_model',
        priority: 10,
        condition: { op: 'SIGNAL_REF', signal_type: 'keyword', signal_name: 'a' },
        models: [{
          model: 'qwen3:70b',
          reasoning: true,
          effort: 'high',
          lora: 'my-adapter',
          param_size: '70b',
          weight: 0.8,
          reasoning_family: 'qwen3',
        }],
      } as RouteIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('reasoning = true')
    expect(dsl).toContain('effort = "high"')
    expect(dsl).toContain('lora = "my-adapter"')
    expect(dsl).toContain('param_size = "70b"')
    expect(dsl).toContain('weight = 0.8')
    expect(dsl).toContain('reasoning_family = "qwen3"')
  })
})

// ─────────────────────────────────────────────
// Backend emission
// ─────────────────────────────────────────────

describe('emitBackend (via intentIRToDSL)', () => {
  it('should emit BACKEND <type> <name> (type before name)', () => {
    const ir = makeIR([
      {
        type: 'backend',
        backend_type: 'vllm_endpoint',
        name: 'ollama',
        fields: { address: '127.0.0.1', port: 11434, weight: 1, type: 'ollama' },
      } as BackendIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('BACKEND vllm_endpoint ollama {')
    expect(dsl).toContain('address: "127.0.0.1"')
    expect(dsl).toContain('port: 11434')
  })
})

// ─────────────────────────────────────────────
// Global emission
// ─────────────────────────────────────────────

describe('emitGlobal (via intentIRToDSL)', () => {
  it('should emit GLOBAL block', () => {
    const ir = makeIR([
      {
        type: 'global',
        fields: { default_model: 'gpt-4o', strategy: 'priority', default_reasoning_effort: 'low' },
      } as GlobalIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('GLOBAL {')
    expect(dsl).toContain('default_model: "gpt-4o"')
    expect(dsl).toContain('strategy: "priority"')
    expect(dsl).toContain('default_reasoning_effort: "low"')
  })
})

// ─────────────────────────────────────────────
// Full DSL generation — section ordering
// ─────────────────────────────────────────────

describe('generateFullDSL section ordering', () => {
  it('should output sections in order: SIGNALS, PLUGINS, ROUTES, BACKENDS, GLOBAL', () => {
    const ir = makeIR([
      { type: 'global', fields: { default_model: 'gpt-4o' } } as GlobalIntent,
      { type: 'backend', backend_type: 'vllm_endpoint', name: 'ep', fields: { port: 8080 } } as BackendIntent,
      { type: 'route', name: 'r1', priority: 10, condition: { op: 'SIGNAL_REF', signal_type: 'keyword', signal_name: 's1' }, models: [{ model: 'gpt-4o' }] } as RouteIntent,
      { type: 'plugin_template', name: 'p1', plugin_type: 'pii', fields: { enabled: true } } as PluginTemplateIntent,
      { type: 'signal', signal_type: 'keyword', name: 's1', fields: { keywords: ['test'] } } as SignalIntent,
    ])
    const dsl = intentIRToDSL(ir)

    const signalIdx = dsl.indexOf('# SIGNALS')
    const pluginIdx = dsl.indexOf('# PLUGINS')
    const routeIdx = dsl.indexOf('# ROUTES')
    const backendIdx = dsl.indexOf('# BACKENDS')
    const globalIdx = dsl.indexOf('# GLOBAL')

    expect(signalIdx).toBeLessThan(pluginIdx)
    expect(pluginIdx).toBeLessThan(routeIdx)
    expect(routeIdx).toBeLessThan(backendIdx)
    expect(backendIdx).toBeLessThan(globalIdx)
  })

  it('should omit empty sections', () => {
    const ir = makeIR([
      { type: 'signal', signal_type: 'keyword', name: 'test', fields: { keywords: ['x'] } } as SignalIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('# SIGNALS')
    expect(dsl).not.toContain('# PLUGINS')
    expect(dsl).not.toContain('# ROUTES')
    expect(dsl).not.toContain('# BACKENDS')
    expect(dsl).not.toContain('# GLOBAL')
  })
})

// ─────────────────────────────────────────────
// Value serialization
// ─────────────────────────────────────────────

describe('value serialization', () => {
  it('should serialize nested objects', () => {
    const ir = makeIR([
      {
        type: 'signal',
        signal_type: 'complexity',
        name: 'hard',
        fields: {
          threshold: 0.7,
          hard: { candidates: ['prove', 'derive'] },
        },
      } as SignalIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('threshold: 0.7')
    // Nested object should be serialized inline (≤3 entries)
    expect(dsl).toContain('candidates: ["prove", "derive"]')
  })

  it('should serialize empty arrays', () => {
    const ir = makeIR([
      {
        type: 'plugin_template',
        name: 'pii_plugin',
        plugin_type: 'pii',
        fields: { pii_types_allowed: [] },
      } as PluginTemplateIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('pii_types_allowed: []')
  })

  it('should skip null/undefined fields', () => {
    const ir = makeIR([
      {
        type: 'signal',
        signal_type: 'keyword',
        name: 'test',
        fields: { keywords: ['a'], method: null, description: undefined },
      } as unknown as SignalIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('keywords: ["a"]')
    expect(dsl).not.toContain('method')
    expect(dsl).not.toContain('description')
  })

  it('should escape strings with quotes', () => {
    const ir = makeIR([
      {
        type: 'signal',
        signal_type: 'domain',
        name: 'test',
        fields: { description: 'He said "hello"' },
      } as SignalIntent,
    ])
    const dsl = intentIRToDSL(ir)
    expect(dsl).toContain('description: "He said \\"hello\\""')
  })
})

// ─────────────────────────────────────────────
// Modification mode
// ─────────────────────────────────────────────

describe('modification mode', () => {
  const existingDSL = `SIGNAL keyword urgent {
  operator: "any"
  keywords: ["urgent"]
}

ROUTE test_route (description = "Test") {
  PRIORITY 10

  WHEN keyword("urgent")

  MODEL "gpt-4o" (reasoning = false)
}

GLOBAL {
  default_model: "gpt-4o"
  strategy: "priority"
}
`

  it('should add a new signal to existing DSL', () => {
    const ir = makeIR(
      [
        {
          type: 'signal',
          signal_type: 'domain',
          name: 'math',
          fields: { description: 'Math' },
        } as SignalIntent,
      ],
      'modify'
    )
    const result = intentIRToDSL(ir, existingDSL)
    expect(result).toContain('SIGNAL keyword urgent')
    expect(result).toContain('SIGNAL domain math')
    expect(result).toContain('ROUTE test_route')
  })

  it('should update signal via ModifyIntent', () => {
    const ir = makeIR(
      [
        {
          type: 'modify',
          action: 'update',
          target_construct: 'signal',
          target_name: 'urgent',
          target_signal_type: 'keyword',
          changes: { operator: 'all', keywords: ['urgent', 'asap'] },
        } as ModifyIntent,
      ],
      'modify'
    )
    const result = intentIRToDSL(ir, existingDSL)
    expect(result).toContain('operator: "all"')
    expect(result).toContain('keywords: ["urgent", "asap"]')
  })

  it('should delete a signal via ModifyIntent', () => {
    const ir = makeIR(
      [
        {
          type: 'modify',
          action: 'delete',
          target_construct: 'signal',
          target_name: 'urgent',
          target_signal_type: 'keyword',
        } as ModifyIntent,
      ],
      'modify'
    )
    const result = intentIRToDSL(ir, existingDSL)
    expect(result).not.toContain('SIGNAL keyword urgent')
    expect(result).toContain('ROUTE test_route')
  })

  it('should delete a route via ModifyIntent', () => {
    const ir = makeIR(
      [
        {
          type: 'modify',
          action: 'delete',
          target_construct: 'route',
          target_name: 'test_route',
        } as ModifyIntent,
      ],
      'modify'
    )
    const result = intentIRToDSL(ir, existingDSL)
    expect(result).toContain('SIGNAL keyword urgent')
    expect(result).not.toContain('ROUTE test_route')
  })

  it('should update global via ModifyIntent', () => {
    const ir = makeIR(
      [
        {
          type: 'modify',
          action: 'update',
          target_construct: 'global',
          target_name: 'global',
          changes: { default_model: 'qwen3:70b', strategy: 'weighted' },
        } as ModifyIntent,
      ],
      'modify'
    )
    const result = intentIRToDSL(ir, existingDSL)
    expect(result).toContain('default_model: "qwen3:70b"')
    expect(result).toContain('strategy: "weighted"')
  })

  it('should fall back to generate mode if no existing DSL', () => {
    const ir = makeIR(
      [
        { type: 'signal', signal_type: 'keyword', name: 'test', fields: { keywords: ['a'] } } as SignalIntent,
      ],
      'modify'
    )
    // No existingDSL → generate mode
    const result = intentIRToDSL(ir)
    expect(result).toContain('SIGNAL keyword test')
  })
})

// ─────────────────────────────────────────────
// resolveImplicitDependencies
// ─────────────────────────────────────────────

describe('resolveImplicitDependencies', () => {
  it('should not add signals that already exist', () => {
    const ir = makeIR([
      { type: 'signal', signal_type: 'keyword', name: 'test', fields: { keywords: ['x'] } } as SignalIntent,
      {
        type: 'route', name: 'r', priority: 10,
        condition: { op: 'SIGNAL_REF', signal_type: 'keyword', signal_name: 'test' },
        models: [{ model: 'gpt-4o' }],
      } as RouteIntent,
    ])
    const resolved = resolveImplicitDependencies(ir)
    const signals = resolved.intents.filter(i => i.type === 'signal')
    expect(signals).toHaveLength(1)
  })

  it('should auto-create missing signal referenced in condition', () => {
    const ir = makeIR([
      {
        type: 'route', name: 'r', priority: 10,
        condition: { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
        models: [{ model: 'gpt-4o' }],
      } as RouteIntent,
    ])
    const resolved = resolveImplicitDependencies(ir)
    const signals = resolved.intents.filter(i => i.type === 'signal') as SignalIntent[]
    expect(signals).toHaveLength(1)
    expect(signals[0].signal_type).toBe('domain')
    expect(signals[0].name).toBe('math')
  })

  it('should auto-create feedback signal for elo algorithm', () => {
    const ir = makeIR([
      {
        type: 'route', name: 'r', priority: 10,
        condition: { op: 'SIGNAL_REF', signal_type: 'keyword', signal_name: 'test' },
        models: [{ model: 'gpt-4o' }],
        algorithm: { algo_type: 'elo', params: { initial_rating: 1500 } },
      } as RouteIntent,
    ])
    const resolved = resolveImplicitDependencies(ir)
    const signals = resolved.intents.filter(i => i.type === 'signal') as SignalIntent[]
    // Should create both: 'keyword:test' and 'user_feedback:feedback'
    expect(signals).toHaveLength(2)
    const types = signals.map(s => s.signal_type)
    expect(types).toContain('keyword')
    expect(types).toContain('user_feedback')
  })

  it('should handle complex nested conditions', () => {
    const ir = makeIR([
      {
        type: 'route', name: 'r', priority: 10,
        condition: {
          op: 'AND',
          operands: [
            {
              op: 'OR',
              operands: [
                { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
                { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'physics' },
              ],
            },
            { op: 'NOT', operand: { op: 'SIGNAL_REF', signal_type: 'complexity', signal_name: 'easy' } },
          ],
        },
        models: [{ model: 'gpt-4o' }],
      } as RouteIntent,
    ])
    const resolved = resolveImplicitDependencies(ir)
    const signals = resolved.intents.filter(i => i.type === 'signal') as SignalIntent[]
    expect(signals).toHaveLength(3)
    const names = signals.map(s => s.name)
    expect(names).toContain('math')
    expect(names).toContain('physics')
    expect(names).toContain('easy')
  })
})

// ─────────────────────────────────────────────
// Full end-to-end DSL generation
// ─────────────────────────────────────────────

describe('end-to-end full DSL generation', () => {
  it('should generate a complete DSL from a realistic IR', () => {
    const ir: IntentIR = {
      version: '1.0',
      operation: 'generate',
      intents: [
        {
          type: 'signal',
          signal_type: 'keyword',
          name: 'urgent_request',
          fields: { operator: 'any', keywords: ['urgent', 'asap', 'emergency'], method: 'regex', case_sensitive: false },
        },
        {
          type: 'signal',
          signal_type: 'embedding',
          name: 'ai_topics',
          fields: { threshold: 0.75, candidates: ['machine learning', 'neural network'], aggregation_method: 'max' },
        },
        {
          type: 'signal',
          signal_type: 'domain',
          name: 'math',
          fields: { description: 'Mathematics', mmlu_categories: ['math'] },
        },
        {
          type: 'plugin_template',
          name: 'safe_pii',
          plugin_type: 'pii',
          fields: { enabled: true, pii_types_allowed: [] },
        },
        {
          type: 'route',
          name: 'ai_route',
          description: 'AI-related queries',
          priority: 100,
          condition: {
            op: 'AND',
            operands: [
              { op: 'SIGNAL_REF', signal_type: 'keyword', signal_name: 'urgent_request' },
              { op: 'SIGNAL_REF', signal_type: 'embedding', signal_name: 'ai_topics' },
            ],
          },
          models: [{ model: 'qwen2.5:3b', reasoning: false }],
          algorithm: { algo_type: 'confidence', params: { confidence_method: 'hybrid', threshold: 0.5, on_error: 'skip' } },
          plugins: [{ name: 'safe_pii' }],
        },
        {
          type: 'route',
          name: 'math_route',
          description: 'Math questions',
          priority: 50,
          condition: { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
          models: [
            { model: 'qwen3:70b', reasoning: true, effort: 'high' },
            { model: 'qwen2.5:3b', reasoning: false },
          ],
        },
        {
          type: 'backend',
          backend_type: 'vllm_endpoint',
          name: 'ollama',
          fields: { address: '127.0.0.1', port: 11434, weight: 1, type: 'ollama' },
        },
        {
          type: 'global',
          fields: { default_model: 'qwen2.5:3b', strategy: 'priority', default_reasoning_effort: 'low' },
        },
      ],
    }

    const dsl = intentIRToDSL(ir)

    // Section ordering
    expect(dsl.indexOf('# SIGNALS')).toBeLessThan(dsl.indexOf('# PLUGINS'))
    expect(dsl.indexOf('# PLUGINS')).toBeLessThan(dsl.indexOf('# ROUTES'))
    expect(dsl.indexOf('# ROUTES')).toBeLessThan(dsl.indexOf('# BACKENDS'))
    expect(dsl.indexOf('# BACKENDS')).toBeLessThan(dsl.indexOf('# GLOBAL'))

    // Signal blocks
    expect(dsl).toContain('SIGNAL keyword urgent_request {')
    expect(dsl).toContain('SIGNAL embedding ai_topics {')
    expect(dsl).toContain('SIGNAL domain math {')

    // Plugin template
    expect(dsl).toContain('PLUGIN safe_pii pii {')

    // Routes
    expect(dsl).toContain('ROUTE ai_route (description = "AI-related queries") {')
    expect(dsl).toContain('WHEN keyword("urgent_request") AND embedding("ai_topics")')
    expect(dsl).toContain('ALGORITHM confidence {')
    expect(dsl).toContain('ROUTE math_route (description = "Math questions") {')

    // Multi-model
    expect(dsl).toContain('"qwen3:70b" (reasoning = true, effort = "high")')
    expect(dsl).toContain('"qwen2.5:3b" (reasoning = false)')

    // Backend (type before name)
    expect(dsl).toContain('BACKEND vllm_endpoint ollama {')

    // Global
    expect(dsl).toContain('GLOBAL {')
    expect(dsl).toContain('default_model: "qwen2.5:3b"')
  })
})
