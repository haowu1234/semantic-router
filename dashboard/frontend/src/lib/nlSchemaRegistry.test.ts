/**
 * Unit tests for NL Schema Registry.
 *
 * Tests cover:
 * - Registry CRUD operations (register, get, getByConstruct)
 * - Trigger-based entity matching (findByTriggers)
 * - System prompt generation (buildSystemPromptSection)
 * - Default registry completeness (all 44 entries)
 * - Edge cases (unknown types, empty queries)
 */

import { describe, it, expect, beforeEach } from 'vitest'
import {
  NLSchemaRegistry,
  defaultRegistry,
  type NLSchemaEntry,
} from './nlSchemaRegistry'
import {
  SIGNAL_TYPES,
  PLUGIN_TYPES,
  ALGORITHM_TYPES,
  BACKEND_TYPES,
} from './dslMutations'

describe('NLSchemaRegistry', () => {
  let registry: NLSchemaRegistry

  beforeEach(() => {
    registry = new NLSchemaRegistry()
  })

  // ─── Basic CRUD ───

  describe('register and get', () => {
    const entry: NLSchemaEntry = {
      construct: 'signal',
      type_name: 'test_signal',
      nl_triggers: ['test', 'testing'],
      nl_description: 'A test signal',
      nl_examples: ['Route test queries'],
      fields: [{ key: 'threshold', label: 'Threshold', type: 'number' }],
    }

    it('should register and retrieve an entry', () => {
      registry.register(entry)
      const result = registry.get('signal', 'test_signal')
      expect(result).toBeDefined()
      expect(result!.type_name).toBe('test_signal')
      expect(result!.nl_triggers).toEqual(['test', 'testing'])
    })

    it('should return undefined for non-existent entry', () => {
      expect(registry.get('signal', 'nonexistent')).toBeUndefined()
    })

    it('should report has() correctly', () => {
      registry.register(entry)
      expect(registry.has('signal', 'test_signal')).toBe(true)
      expect(registry.has('signal', 'nonexistent')).toBe(false)
    })

    it('should overwrite on duplicate register', () => {
      registry.register(entry)
      const updated = { ...entry, nl_description: 'Updated description' }
      registry.register(updated)
      expect(registry.get('signal', 'test_signal')!.nl_description).toBe('Updated description')
    })
  })

  describe('registerAll', () => {
    it('should register multiple entries', () => {
      const entries: NLSchemaEntry[] = [
        {
          construct: 'signal',
          type_name: 'a',
          nl_triggers: ['a'],
          nl_description: 'Signal A',
          nl_examples: [],
          fields: [],
        },
        {
          construct: 'plugin',
          type_name: 'b',
          nl_triggers: ['b'],
          nl_description: 'Plugin B',
          nl_examples: [],
          fields: [],
        },
      ]
      registry.registerAll(entries)
      expect(registry.has('signal', 'a')).toBe(true)
      expect(registry.has('plugin', 'b')).toBe(true)
    })
  })

  describe('getByConstruct', () => {
    it('should filter by construct type', () => {
      registry.registerAll([
        { construct: 'signal', type_name: 's1', nl_triggers: [], nl_description: '', nl_examples: [], fields: [] },
        { construct: 'signal', type_name: 's2', nl_triggers: [], nl_description: '', nl_examples: [], fields: [] },
        { construct: 'plugin', type_name: 'p1', nl_triggers: [], nl_description: '', nl_examples: [], fields: [] },
        { construct: 'algorithm', type_name: 'a1', nl_triggers: [], nl_description: '', nl_examples: [], fields: [] },
      ])

      expect(registry.getByConstruct('signal')).toHaveLength(2)
      expect(registry.getByConstruct('plugin')).toHaveLength(1)
      expect(registry.getByConstruct('algorithm')).toHaveLength(1)
      expect(registry.getByConstruct('backend')).toHaveLength(0)
    })
  })

  describe('getAll', () => {
    it('should return all entries', () => {
      registry.registerAll([
        { construct: 'signal', type_name: 's1', nl_triggers: [], nl_description: '', nl_examples: [], fields: [] },
        { construct: 'plugin', type_name: 'p1', nl_triggers: [], nl_description: '', nl_examples: [], fields: [] },
      ])
      expect(registry.getAll()).toHaveLength(2)
    })

    it('should return empty array for empty registry', () => {
      expect(registry.getAll()).toHaveLength(0)
    })
  })

  // ─── Trigger Matching ───

  describe('findByTriggers', () => {
    beforeEach(() => {
      registry.registerAll([
        {
          construct: 'signal',
          type_name: 'keyword',
          nl_triggers: ['keyword', 'pattern', 'match'],
          nl_description: 'Keyword matching',
          nl_examples: [],
          fields: [],
        },
        {
          construct: 'signal',
          type_name: 'domain',
          nl_triggers: ['domain', 'topic', 'category'],
          nl_description: 'Domain classification',
          nl_examples: [],
          fields: [],
        },
        {
          construct: 'plugin',
          type_name: 'semantic_cache',
          nl_triggers: ['cache', 'caching', 'semantic cache'],
          nl_description: 'Semantic caching',
          nl_examples: [],
          fields: [],
        },
      ])
    })

    it('should find entries matching trigger words', () => {
      const results = registry.findByTriggers(['domain'])
      expect(results).toHaveLength(1)
      expect(results[0].type_name).toBe('domain')
    })

    it('should match partial words (substring)', () => {
      const results = registry.findByTriggers(['keywords'])
      expect(results.length).toBeGreaterThanOrEqual(1)
      expect(results[0].type_name).toBe('keyword')
    })

    it('should be case-insensitive', () => {
      const results = registry.findByTriggers(['CACHE'])
      expect(results.length).toBeGreaterThanOrEqual(1)
      expect(results[0].type_name).toBe('semantic_cache')
    })

    it('should return multiple matches sorted by score', () => {
      const results = registry.findByTriggers(['keyword', 'cache', 'pattern'])
      expect(results.length).toBeGreaterThanOrEqual(2)
      // keyword should score highest (matches 'keyword' + 'pattern')
      expect(results[0].type_name).toBe('keyword')
    })

    it('should return empty for no matches', () => {
      const results = registry.findByTriggers(['quantum', 'teleportation'])
      expect(results).toHaveLength(0)
    })

    it('should handle empty input', () => {
      const results = registry.findByTriggers([])
      expect(results).toHaveLength(0)
    })
  })

  // ─── System Prompt Generation ───

  describe('buildSystemPromptSection', () => {
    it('should generate prompt sections for all construct types', () => {
      registry.registerAll([
        {
          construct: 'signal',
          type_name: 'keyword',
          nl_triggers: ['keyword'],
          nl_description: 'Keyword matching signal',
          nl_examples: [],
          fields: [
            { key: 'patterns', label: 'Patterns', type: 'string[]', required: true },
            { key: 'threshold', label: 'Threshold', type: 'number' },
          ],
        },
        {
          construct: 'plugin',
          type_name: 'semantic_cache',
          nl_triggers: ['cache'],
          nl_description: 'Response caching plugin',
          nl_examples: [],
          fields: [
            { key: 'enabled', label: 'Enabled', type: 'boolean' },
          ],
          requires_backend: ['semantic_cache'],
        },
        {
          construct: 'algorithm',
          type_name: 'confidence',
          nl_triggers: ['confidence'],
          nl_description: 'Confidence cascade algorithm',
          nl_examples: [],
          fields: [
            { key: 'threshold', label: 'Threshold', type: 'number' },
          ],
        },
        {
          construct: 'backend',
          type_name: 'vllm_endpoint',
          nl_triggers: ['vllm'],
          nl_description: 'vLLM inference endpoint',
          nl_examples: [],
          fields: [],
        },
      ])

      const prompt = registry.buildSystemPromptSection()

      // Should contain section headers
      expect(prompt).toContain('### Signal Types')
      expect(prompt).toContain('### Plugin Types')
      expect(prompt).toContain('### Algorithm Types')
      expect(prompt).toContain('### Backend Types')

      // Should contain type names
      expect(prompt).toContain('**keyword**')
      expect(prompt).toContain('**semantic_cache**')
      expect(prompt).toContain('**confidence**')
      expect(prompt).toContain('**vllm_endpoint**')

      // Should contain descriptions
      expect(prompt).toContain('Keyword matching signal')
      expect(prompt).toContain('Response caching plugin')

      // Should list required fields
      expect(prompt).toContain('Required: patterns')

      // Should list backend dependencies
      expect(prompt).toContain('Requires backend: semantic_cache')
    })

    it('should handle empty registry', () => {
      const prompt = registry.buildSystemPromptSection()
      expect(prompt).toBe('')
    })

    it('should handle registry with only one construct type', () => {
      registry.register({
        construct: 'signal',
        type_name: 'domain',
        nl_triggers: ['domain'],
        nl_description: 'Domain classification',
        nl_examples: [],
        fields: [],
      })

      const prompt = registry.buildSystemPromptSection()
      expect(prompt).toContain('### Signal Types')
      expect(prompt).not.toContain('### Plugin Types')
    })
  })
})

// ─── Default Registry Tests ───

describe('defaultRegistry', () => {
  describe('completeness', () => {
    it('should have all 13 signal types registered', () => {
      const signals = defaultRegistry.getByConstruct('signal')
      const registeredTypes = signals.map(s => s.type_name).sort()
      const expectedTypes = [...SIGNAL_TYPES].sort()
      expect(registeredTypes).toEqual(expectedTypes)
    })

    it('should have all 9 plugin types registered', () => {
      const plugins = defaultRegistry.getByConstruct('plugin')
      const registeredTypes = plugins.map(p => p.type_name).sort()
      const expectedTypes = [...PLUGIN_TYPES].sort()
      expect(registeredTypes).toEqual(expectedTypes)
    })

    it('should have all 14 algorithm types registered', () => {
      const algorithms = defaultRegistry.getByConstruct('algorithm')
      const registeredTypes = algorithms.map(a => a.type_name).sort()
      const expectedTypes = [...ALGORITHM_TYPES].sort()
      expect(registeredTypes).toEqual(expectedTypes)
    })

    it('should have all 8 backend types registered', () => {
      const backends = defaultRegistry.getByConstruct('backend')
      const registeredTypes = backends.map(b => b.type_name).sort()
      const expectedTypes = [...BACKEND_TYPES].sort()
      expect(registeredTypes).toEqual(expectedTypes)
    })

    it('should have 44 total entries (13+9+14+8)', () => {
      expect(defaultRegistry.getAll()).toHaveLength(44)
    })
  })

  describe('trigger quality', () => {
    it('every entry should have at least 1 trigger word', () => {
      for (const entry of defaultRegistry.getAll()) {
        expect(
          entry.nl_triggers.length,
          `${entry.construct}:${entry.type_name} should have triggers`,
        ).toBeGreaterThanOrEqual(1)
      }
    })

    it('every entry should have a non-empty description', () => {
      for (const entry of defaultRegistry.getAll()) {
        expect(
          entry.nl_description.length,
          `${entry.construct}:${entry.type_name} should have description`,
        ).toBeGreaterThan(0)
      }
    })

    it('every entry should have at least 1 example', () => {
      for (const entry of defaultRegistry.getAll()) {
        expect(
          entry.nl_examples.length,
          `${entry.construct}:${entry.type_name} should have examples`,
        ).toBeGreaterThanOrEqual(1)
      }
    })
  })

  describe('field schema consistency', () => {
    it('signal entries should have field schemas from getSignalFieldSchema', () => {
      for (const signal of defaultRegistry.getByConstruct('signal')) {
        expect(
          Array.isArray(signal.fields),
          `signal:${signal.type_name} fields should be an array`,
        ).toBe(true)
      }
    })

    it('keyword signal should have operator and keywords fields', () => {
      const keyword = defaultRegistry.get('signal', 'keyword')!
      const fieldKeys = keyword.fields.map(f => f.key)
      expect(fieldKeys).toContain('operator')
      expect(fieldKeys).toContain('keywords')
    })

    it('embedding signal should have threshold and candidates', () => {
      const embedding = defaultRegistry.get('signal', 'embedding')!
      const fieldKeys = embedding.fields.map(f => f.key)
      expect(fieldKeys).toContain('threshold')
      expect(fieldKeys).toContain('candidates')
    })

    it('confidence algorithm should have threshold field', () => {
      const confidence = defaultRegistry.get('algorithm', 'confidence')!
      const fieldKeys = confidence.fields.map(f => f.key)
      expect(fieldKeys).toContain('threshold')
    })

    it('semantic_cache plugin should have similarity_threshold field', () => {
      const cache = defaultRegistry.get('plugin', 'semantic_cache')!
      const fieldKeys = cache.fields.map(f => f.key)
      expect(fieldKeys).toContain('similarity_threshold')
    })
  })

  describe('dependency declarations', () => {
    it('semantic_cache plugin should require semantic_cache backend', () => {
      const cache = defaultRegistry.get('plugin', 'semantic_cache')!
      expect(cache.requires_backend).toContain('semantic_cache')
    })

    it('memory plugin should require memory backend', () => {
      const memory = defaultRegistry.get('plugin', 'memory')!
      expect(memory.requires_backend).toContain('memory')
    })

    it('rag plugin should require vector_store backend', () => {
      const rag = defaultRegistry.get('plugin', 'rag')!
      expect(rag.requires_backend).toContain('vector_store')
    })

    it('image_gen plugin should require image_gen_backend', () => {
      const imageGen = defaultRegistry.get('plugin', 'image_gen')!
      expect(imageGen.requires_backend).toContain('image_gen_backend')
    })

    it('elo algorithm should require user_feedback signal', () => {
      const elo = defaultRegistry.get('algorithm', 'elo')!
      expect(elo.requires_signal).toContain('user_feedback')
    })

    it('rl_driven algorithm should require user_feedback signal', () => {
      const rl = defaultRegistry.get('algorithm', 'rl_driven')!
      expect(rl.requires_signal).toContain('user_feedback')
    })
  })

  describe('NL trigger matching on default registry', () => {
    it('should find domain signal for "数学题"', () => {
      const results = defaultRegistry.findByTriggers(['domain', '领域'])
      const types = results.map(r => r.type_name)
      expect(types).toContain('domain')
    })

    it('should find cache plugin for "缓存"', () => {
      const results = defaultRegistry.findByTriggers(['缓存'])
      const types = results.map(r => r.type_name)
      expect(types).toContain('semantic_cache')
    })

    it('should find pii signal for "隐私"', () => {
      const results = defaultRegistry.findByTriggers(['隐私'])
      const types = results.map(r => r.type_name)
      expect(types).toContain('pii')
    })

    it('should find jailbreak signal for "安全"', () => {
      const results = defaultRegistry.findByTriggers(['安全'])
      const types = results.map(r => r.type_name)
      expect(types).toContain('jailbreak')
    })

    it('should find confidence algorithm for "置信度"', () => {
      const results = defaultRegistry.findByTriggers(['置信度'])
      const types = results.map(r => r.type_name)
      expect(types).toContain('confidence')
    })

    it('should find latency_aware algorithm for "延迟"', () => {
      const results = defaultRegistry.findByTriggers(['延迟'])
      const types = results.map(r => r.type_name)
      expect(types).toContain('latency_aware')
    })

    it('should find multiple results for "embedding similarity"', () => {
      const results = defaultRegistry.findByTriggers(['embedding', 'similarity'])
      expect(results.length).toBeGreaterThanOrEqual(2)
      const types = results.map(r => r.type_name)
      expect(types).toContain('embedding')
      expect(types).toContain('embedding_model')
    })
  })

  describe('system prompt generation', () => {
    it('should generate a non-empty prompt', () => {
      const prompt = defaultRegistry.buildSystemPromptSection()
      expect(prompt.length).toBeGreaterThan(500)
    })

    it('should contain all 4 section headers', () => {
      const prompt = defaultRegistry.buildSystemPromptSection()
      expect(prompt).toContain('### Signal Types')
      expect(prompt).toContain('### Plugin Types')
      expect(prompt).toContain('### Algorithm Types')
      expect(prompt).toContain('### Backend Types')
    })

    it('should list all signal types', () => {
      const prompt = defaultRegistry.buildSystemPromptSection()
      for (const st of SIGNAL_TYPES) {
        expect(prompt).toContain(`**${st}**`)
      }
    })

    it('should list required fields for keyword signal', () => {
      const prompt = defaultRegistry.buildSystemPromptSection()
      expect(prompt).toContain('Required: operator')
    })
  })
})
