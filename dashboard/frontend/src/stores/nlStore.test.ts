/**
 * Tests for nlStore — NL Mode Zustand store.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useNLStore } from './nlStore'
import type { NLStore } from './nlStore'

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

function getStore(): NLStore {
  return useNLStore.getState()
}

function resetStore() {
  useNLStore.setState({
    ...useNLStore.getState(),
    session: { id: 'test', turns: [], currentDSL: '', symbolTable: null },
    messages: [],
    progress: null,
    isGenerating: false,
    inputText: '',
    pendingResult: null,
    apiEndpoint: '/api/nl/generate',
    apiKey: '',
    modelName: 'qwen3-32b',
    showSettings: false,
  })
}

// Minimal mock WASM bridge
function createMockWasm() {
  return {
    init: vi.fn().mockResolvedValue(undefined),
    compile: vi.fn().mockReturnValue({
      yaml: 'mock: yaml',
      crd: '',
      diagnostics: [],
      ast: null,
      error: null,
    }),
    validate: vi.fn().mockReturnValue({
      diagnostics: [],
      symbols: { signals: [], routes: [], plugins: [], backends: [], models: [] },
      error: null,
    }),
    parseAST: vi.fn().mockReturnValue({
      ast: null,
      diagnostics: [],
      symbols: null,
      error: null,
    }),
    decompile: vi.fn().mockReturnValue({ dsl: '', error: null }),
    format: vi.fn().mockReturnValue({ dsl: '', error: null }),
  }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

describe('nlStore', () => {
  beforeEach(() => {
    resetStore()
  })

  describe('initial state', () => {
    it('starts with empty messages and session', () => {
      const s = getStore()
      expect(s.messages).toEqual([])
      expect(s.isGenerating).toBe(false)
      expect(s.pendingResult).toBeNull()
      expect(s.progress).toBeNull()
    })

    it('has default LLM settings', () => {
      const s = getStore()
      expect(s.apiEndpoint).toBe('/api/nl/generate')
      expect(s.apiKey).toBe('')
      expect(s.modelName).toBe('qwen3-32b')
      expect(s.showSettings).toBe(false)
    })
  })

  describe('setInputText', () => {
    it('updates input text', () => {
      getStore().setInputText('hello world')
      expect(getStore().inputText).toBe('hello world')
    })

    it('allows empty string', () => {
      getStore().setInputText('something')
      getStore().setInputText('')
      expect(getStore().inputText).toBe('')
    })
  })

  describe('LLM settings', () => {
    it('setApiEndpoint updates endpoint', () => {
      getStore().setApiEndpoint('https://api.example.com/v1/chat')
      expect(getStore().apiEndpoint).toBe('https://api.example.com/v1/chat')
    })

    it('setApiKey updates key', () => {
      getStore().setApiKey('sk-test-key')
      expect(getStore().apiKey).toBe('sk-test-key')
    })

    it('setModelName updates model', () => {
      getStore().setModelName('gpt-4o')
      expect(getStore().modelName).toBe('gpt-4o')
    })

    it('setShowSettings toggles', () => {
      getStore().setShowSettings(true)
      expect(getStore().showSettings).toBe(true)
      getStore().setShowSettings(false)
      expect(getStore().showSettings).toBe(false)
    })
  })

  describe('resetSession', () => {
    it('clears messages and session', () => {
      // Pollute state
      useNLStore.setState({
        messages: [
          { id: '1', role: 'user', content: 'test', timestamp: 1 },
        ],
        inputText: 'pending input',
        isGenerating: false,
        pendingResult: { dsl: 'x', intentIR: { version: '1.0', operation: 'generate', intents: [] }, confidence: 0.9, isValid: true, retries: 0, explanation: '', diagnostics: [] },
      })

      getStore().resetSession()

      expect(getStore().messages).toEqual([])
      expect(getStore().inputText).toBe('')
      expect(getStore().pendingResult).toBeNull()
      expect(getStore().isGenerating).toBe(false)
      expect(getStore().progress).toBeNull()
    })
  })

  describe('acceptResult', () => {
    it('returns null when no pending result', () => {
      const result = getStore().acceptResult()
      expect(result).toBeNull()
    })

    it('returns DSL and IR when pending result exists', () => {
      const pendingResult = {
        dsl: 'SIGNAL keyword test {\n  keywords: ["hello"]\n}',
        intentIR: {
          version: '1.0' as const,
          operation: 'generate' as const,
          intents: [
            {
              type: 'signal' as const,
              signal_type: 'keyword' as any,
              name: 'test',
              fields: { keywords: ['hello'] },
            },
          ],
        },
        confidence: 0.9,
        isValid: true,
        retries: 0,
        explanation: 'Generated 1 signal',
        diagnostics: [],
      }

      useNLStore.setState({
        pendingResult,
        messages: [
          { id: '1', role: 'user', content: 'test', timestamp: 1 },
          { id: '2', role: 'assistant', content: 'Generated 1 signal', timestamp: 2, result: pendingResult },
        ],
      })

      const result = getStore().acceptResult()
      expect(result).not.toBeNull()
      expect(result!.dsl).toBe(pendingResult.dsl)
      expect(result!.intentIR).toEqual(pendingResult.intentIR)
    })

    it('marks the last assistant message as accepted', () => {
      const pendingResult = {
        dsl: 'test',
        intentIR: { version: '1.0' as const, operation: 'generate' as const, intents: [] },
        confidence: 1,
        isValid: true,
        retries: 0,
        explanation: 'done',
        diagnostics: [],
      }

      useNLStore.setState({
        pendingResult,
        messages: [
          { id: '1', role: 'user', content: 'x', timestamp: 1 },
          { id: '2', role: 'assistant', content: 'done', timestamp: 2, result: pendingResult },
        ],
      })

      getStore().acceptResult()

      const msgs = getStore().messages
      expect(msgs[1].accepted).toBe(true)
      expect(getStore().pendingResult).toBeNull()
    })
  })

  describe('rejectResult', () => {
    it('marks the last assistant message as rejected', () => {
      const pendingResult = {
        dsl: 'test',
        intentIR: { version: '1.0' as const, operation: 'generate' as const, intents: [] },
        confidence: 1,
        isValid: true,
        retries: 0,
        explanation: 'done',
        diagnostics: [],
      }

      useNLStore.setState({
        pendingResult,
        messages: [
          { id: '1', role: 'user', content: 'x', timestamp: 1 },
          { id: '2', role: 'assistant', content: 'done', timestamp: 2, result: pendingResult },
        ],
      })

      getStore().rejectResult()

      const msgs = getStore().messages
      expect(msgs[1].accepted).toBe(false)
      expect(getStore().pendingResult).toBeNull()
    })
  })

  describe('generate', () => {
    it('adds user message and sets isGenerating on call', async () => {
      const mockWasm = createMockWasm()

      // Mock fetch to return a valid LLM response
      const mockIntentIR = {
        version: '1.0',
        operation: 'generate',
        intents: [
          { type: 'signal', signal_type: 'keyword', name: 'test', fields: { keywords: ['hello'] } },
        ],
      }

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ message: { content: JSON.stringify(mockIntentIR) } }],
        }),
      })

      // The generate call will fail at WASM validation since mockWasm returns
      // no diagnostics but the pipeline expects a real WasmBridge.
      // We just verify the user message is added.
      const promise = getStore().generate(
        'route math to gpt-4o',
        mockWasm as any,
        '',
        null,
        [],
      )

      // User message should be added immediately
      const msgs = getStore().messages
      expect(msgs.length).toBeGreaterThanOrEqual(1)
      expect(msgs[0].role).toBe('user')
      expect(msgs[0].content).toBe('route math to gpt-4o')
      expect(getStore().isGenerating).toBe(true)

      // Wait for completion
      await promise

      // After completion, isGenerating should be false
      expect(getStore().isGenerating).toBe(false)
    })

    it('handles LLM API failure gracefully', async () => {
      const mockWasm = createMockWasm()

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        text: () => Promise.resolve('Internal Server Error'),
      })

      const result = await getStore().generate(
        'test input',
        mockWasm as any,
        '',
        null,
        [],
      )

      expect(result).toBeNull()
      expect(getStore().isGenerating).toBe(false)

      // Should have a user message + error message
      const msgs = getStore().messages
      expect(msgs.length).toBe(2)
      expect(msgs[0].role).toBe('user')
      expect(msgs[1].role).toBe('system')
      expect(msgs[1].error).toBe(true)
    })

    it('handles network error gracefully', async () => {
      const mockWasm = createMockWasm()

      globalThis.fetch = vi.fn().mockRejectedValue(new Error('Network error'))

      const result = await getStore().generate(
        'test input',
        mockWasm as any,
        '',
        null,
        [],
      )

      expect(result).toBeNull()
      expect(getStore().isGenerating).toBe(false)
      expect(getStore().messages[1].error).toBe(true)
    })

    it('clears input text on send', async () => {
      const mockWasm = createMockWasm()
      globalThis.fetch = vi.fn().mockRejectedValue(new Error('fail'))

      getStore().setInputText('my input')
      await getStore().generate('my input', mockWasm as any, '', null, [])

      expect(getStore().inputText).toBe('')
    })
  })

  describe('session management', () => {
    it('session has a valid ID', () => {
      const s = getStore()
      expect(s.session.id).toBeTruthy()
      expect(typeof s.session.id).toBe('string')
    })

    it('resetSession creates new session ID', () => {
      const oldId = getStore().session.id
      getStore().resetSession()
      const newId = getStore().session.id
      // New session should have a different ID (probabilistic but extremely unlikely to collide)
      expect(newId).not.toBe(oldId)
    })
  })
})
