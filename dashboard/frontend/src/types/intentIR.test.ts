/**
 * Unit tests for Intent IR type definitions.
 *
 * Since intentIR.ts is purely type definitions, these tests validate that
 * the type system correctly constrains valid IR structures at runtime
 * (structure shape, required fields, discriminated unions).
 */

import { describe, it, expect } from 'vitest'
import type {
  IntentIR,
  Intent,
  SignalIntent,
  RouteIntent,
  PluginTemplateIntent,
  BackendIntent,
  GlobalIntent,
  ModifyIntent,
  ConditionNode,
  ModelIntent,
  AlgorithmIntent,
  PluginRefIntent,
  NLGenerateResult,
  NLGenerateRequest,
  NLSession,
} from './intentIR'

describe('IntentIR type structures', () => {
  describe('IntentIR top-level', () => {
    it('should create a valid generate IR', () => {
      const ir: IntentIR = {
        version: '1.0',
        operation: 'generate',
        intents: [],
      }
      expect(ir.version).toBe('1.0')
      expect(ir.operation).toBe('generate')
      expect(ir.intents).toEqual([])
    })

    it('should support all operation modes', () => {
      const ops: IntentIR['operation'][] = ['generate', 'modify', 'fix']
      for (const op of ops) {
        const ir: IntentIR = { version: '1.0', operation: op, intents: [] }
        expect(ir.operation).toBe(op)
      }
    })
  })

  describe('SignalIntent', () => {
    it('should create a keyword signal intent', () => {
      const signal: SignalIntent = {
        type: 'signal',
        signal_type: 'keyword',
        name: 'urgent_request',
        fields: {
          operator: 'any',
          keywords: ['urgent', 'asap', 'emergency'],
        },
      }
      expect(signal.type).toBe('signal')
      expect(signal.signal_type).toBe('keyword')
      expect(signal.name).toBe('urgent_request')
      expect(signal.fields.keywords).toEqual(['urgent', 'asap', 'emergency'])
    })

    it('should create all 13 signal types', () => {
      const signalTypes = [
        'keyword', 'embedding', 'domain', 'fact_check', 'user_feedback',
        'preference', 'language', 'context', 'complexity', 'modality',
        'authz', 'jailbreak', 'pii',
      ] as const

      for (const st of signalTypes) {
        const signal: SignalIntent = {
          type: 'signal',
          signal_type: st,
          name: `test_${st}`,
          fields: {},
        }
        expect(signal.signal_type).toBe(st)
      }
    })

    it('should handle complex field values', () => {
      const signal: SignalIntent = {
        type: 'signal',
        signal_type: 'authz',
        name: 'admin_access',
        fields: {
          subjects: [{ kind: 'Group', name: 'admins' }],
          role: 'admin',
        },
      }
      expect(signal.fields.subjects).toHaveLength(1)
      expect(signal.fields.role).toBe('admin')
    })
  })

  describe('RouteIntent', () => {
    it('should create a route with simple signal ref condition', () => {
      const route: RouteIntent = {
        type: 'route',
        name: 'math_decision',
        description: 'Route math queries',
        priority: 10,
        condition: {
          op: 'SIGNAL_REF',
          signal_type: 'domain',
          signal_name: 'math',
        },
        models: [{ model: 'gpt-4o' }],
      }
      expect(route.type).toBe('route')
      expect(route.condition.op).toBe('SIGNAL_REF')
      expect(route.models).toHaveLength(1)
    })

    it('should create a route with AND condition', () => {
      const route: RouteIntent = {
        type: 'route',
        name: 'complex_math',
        condition: {
          op: 'AND',
          operands: [
            { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
            { op: 'SIGNAL_REF', signal_type: 'complexity', signal_name: 'hard' },
          ],
        },
        models: [{ model: 'deepseek-r1', reasoning: true, effort: 'high' }],
      }
      expect(route.condition.op).toBe('AND')
      if (route.condition.op === 'AND') {
        expect(route.condition.operands).toHaveLength(2)
      }
    })

    it('should create a route with nested boolean expression', () => {
      const condition: ConditionNode = {
        op: 'OR',
        operands: [
          {
            op: 'AND',
            operands: [
              { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
              { op: 'NOT', operand: { op: 'SIGNAL_REF', signal_type: 'complexity', signal_name: 'easy' } },
            ],
          },
          { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'coding' },
        ],
      }
      expect(condition.op).toBe('OR')
      if (condition.op === 'OR') {
        expect(condition.operands).toHaveLength(2)
        expect(condition.operands[0].op).toBe('AND')
      }
    })

    it('should support multiple models with attributes', () => {
      const models: ModelIntent[] = [
        { model: 'gpt-4o-mini', weight: 0.7 },
        { model: 'gpt-4o', weight: 0.3, reasoning: true, effort: 'medium' },
      ]
      expect(models).toHaveLength(2)
      expect(models[0].weight).toBe(0.7)
      expect(models[1].reasoning).toBe(true)
    })

    it('should support algorithm with params', () => {
      const algo: AlgorithmIntent = {
        algo_type: 'confidence',
        params: { threshold: 0.8, confidence_method: 'avg_logprob' },
      }
      expect(algo.algo_type).toBe('confidence')
      expect(algo.params.threshold).toBe(0.8)
    })

    it('should support plugin references', () => {
      const plugins: PluginRefIntent[] = [
        { name: 'my_cache' },
        { name: 'my_prompt', overrides: { system_prompt: 'Custom' } },
      ]
      expect(plugins).toHaveLength(2)
      expect(plugins[1].overrides?.system_prompt).toBe('Custom')
    })
  })

  describe('PluginTemplateIntent', () => {
    it('should create a plugin template', () => {
      const plugin: PluginTemplateIntent = {
        type: 'plugin_template',
        name: 'my_cache',
        plugin_type: 'semantic_cache',
        fields: { enabled: true, similarity_threshold: 0.95 },
      }
      expect(plugin.type).toBe('plugin_template')
      expect(plugin.plugin_type).toBe('semantic_cache')
    })
  })

  describe('BackendIntent', () => {
    it('should create a backend intent', () => {
      const backend: BackendIntent = {
        type: 'backend',
        backend_type: 'vllm_endpoint',
        name: 'my_vllm',
        fields: { host: 'localhost', port: 8000 },
      }
      expect(backend.type).toBe('backend')
      expect(backend.backend_type).toBe('vllm_endpoint')
    })
  })

  describe('GlobalIntent', () => {
    it('should create a global intent', () => {
      const global: GlobalIntent = {
        type: 'global',
        fields: { log_level: 'debug', max_retries: 3 },
      }
      expect(global.type).toBe('global')
      expect(global.fields.log_level).toBe('debug')
    })
  })

  describe('ModifyIntent', () => {
    it('should create an add action', () => {
      const modify: ModifyIntent = {
        type: 'modify',
        action: 'add',
        target_construct: 'signal',
        target_name: 'new_signal',
        target_signal_type: 'keyword',
        changes: { keywords: ['test'] },
      }
      expect(modify.action).toBe('add')
      expect(modify.target_construct).toBe('signal')
    })

    it('should create a delete action', () => {
      const modify: ModifyIntent = {
        type: 'modify',
        action: 'delete',
        target_construct: 'route',
        target_name: 'old_route',
      }
      expect(modify.action).toBe('delete')
      expect(modify.changes).toBeUndefined()
    })

    it('should support all target constructs', () => {
      const constructs: ModifyIntent['target_construct'][] = [
        'signal', 'route', 'plugin', 'backend', 'global',
      ]
      for (const c of constructs) {
        const m: ModifyIntent = {
          type: 'modify',
          action: 'update',
          target_construct: c,
          target_name: 'test',
          changes: {},
        }
        expect(m.target_construct).toBe(c)
      }
    })
  })

  describe('Intent discriminated union', () => {
    it('should correctly identify intent types via type field', () => {
      const intents: Intent[] = [
        { type: 'signal', signal_type: 'domain', name: 'math', fields: {} },
        {
          type: 'route',
          name: 'math_route',
          condition: { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
          models: [{ model: 'gpt-4o' }],
        },
        { type: 'plugin_template', name: 'cache', plugin_type: 'semantic_cache', fields: {} },
        { type: 'backend', backend_type: 'vllm_endpoint', name: 'ep1', fields: {} },
        { type: 'global', fields: {} },
        { type: 'modify', action: 'add', target_construct: 'signal', target_name: 'x' },
      ]

      const types = intents.map(i => i.type)
      expect(types).toEqual([
        'signal', 'route', 'plugin_template', 'backend', 'global', 'modify',
      ])
    })
  })

  describe('Full IntentIR example', () => {
    it('should compose a complete multi-route configuration', () => {
      const ir: IntentIR = {
        version: '1.0',
        operation: 'generate',
        intents: [
          {
            type: 'signal',
            signal_type: 'domain',
            name: 'math',
            fields: { description: 'Mathematics and calculations' },
          },
          {
            type: 'signal',
            signal_type: 'domain',
            name: 'coding',
            fields: { description: 'Programming and software development' },
          },
          {
            type: 'signal',
            signal_type: 'pii',
            name: 'sensitive_data',
            fields: { threshold: 0.8, pii_types_allowed: ['email'] },
          },
          {
            type: 'plugin_template',
            name: 'pii_guard',
            plugin_type: 'hallucination',
            fields: { enabled: true },
          },
          {
            type: 'route',
            name: 'math_decision',
            description: 'Route math queries to reasoning model',
            priority: 10,
            condition: { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'math' },
            models: [{ model: 'deepseek-r1', reasoning: true, effort: 'high' }],
            algorithm: { algo_type: 'confidence', params: { threshold: 0.8 } },
            plugins: [{ name: 'pii_guard' }],
          },
          {
            type: 'route',
            name: 'code_decision',
            priority: 10,
            condition: { op: 'SIGNAL_REF', signal_type: 'domain', signal_name: 'coding' },
            models: [{ model: 'gpt-4o' }],
            plugins: [{ name: 'pii_guard' }],
          },
        ],
      }

      expect(ir.intents).toHaveLength(6)

      const signals = ir.intents.filter(i => i.type === 'signal')
      const routes = ir.intents.filter(i => i.type === 'route')
      const plugins = ir.intents.filter(i => i.type === 'plugin_template')

      expect(signals).toHaveLength(3)
      expect(routes).toHaveLength(2)
      expect(plugins).toHaveLength(1)
    })
  })

  describe('NL pipeline types', () => {
    it('should create a valid NLGenerateRequest', () => {
      const req: NLGenerateRequest = {
        prompt: 'Route math questions to GPT-4o',
        mode: 'generate',
      }
      expect(req.prompt).toBeTruthy()
      expect(req.mode).toBe('generate')
      expect(req.current_dsl).toBeUndefined()
    })

    it('should create a modify request with context', () => {
      const req: NLGenerateRequest = {
        prompt: 'Add PII protection to all routes',
        current_dsl: 'SIGNAL domain math {\n  description: "math"\n}\n',
        mode: 'modify',
        symbols: {
          signals: [{ name: 'math', type: 'domain' }],
          models: [],
          plugins: [],
          backends: [],
          routes: [],
        },
      }
      expect(req.mode).toBe('modify')
      expect(req.current_dsl).toBeTruthy()
      expect(req.symbols?.signals).toHaveLength(1)
    })

    it('should create a valid NLGenerateResult', () => {
      const result: NLGenerateResult = {
        dsl: 'SIGNAL domain math {\n  description: "Mathematics"\n}\n',
        intentIR: { version: '1.0', operation: 'generate', intents: [] },
        repaired: false,
        repairAttempts: 0,
        diagnostics: [],
      }
      expect(result.dsl).toBeTruthy()
      expect(result.repaired).toBe(false)
      expect(result.diagnostics).toEqual([])
    })

    it('should create a valid NLSession with turns', () => {
      const session: NLSession = {
        id: 'sess_123',
        turns: [
          {
            input: 'Route math to GPT-4o',
            dsl: 'SIGNAL domain math {}',
            accepted: true,
            timestamp: Date.now(),
          },
          {
            input: 'Add caching',
            error: 'Failed to generate',
            timestamp: Date.now(),
          },
        ],
        created_at: Date.now(),
      }
      expect(session.turns).toHaveLength(2)
      expect(session.turns[0].accepted).toBe(true)
      expect(session.turns[1].error).toBeTruthy()
    })
  })
})
