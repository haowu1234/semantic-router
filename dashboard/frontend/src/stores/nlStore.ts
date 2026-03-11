/**
 * Zustand store for NL Mode session state.
 *
 * Manages:
 * - Multi-turn NL conversation (input → IR → DSL)
 * - Pipeline progress tracking (classifying / generating / validating / repairing / done)
 * - Result accept/reject with DSL Store integration
 * - Session lifecycle (create / reset)
 * - LLM client configuration
 */

import { create } from 'zustand'
import type { Diagnostic, SymbolTable, WasmBridge } from '@/types/dsl'
import type { IntentIR } from '@/types/intentIR'
import {
  nlToDSL,
  createSession,
  addTurn,
  acceptLastTurn,
  rejectLastTurn,
} from '@/lib/nlPipeline'
import type {
  NLSession,
  NLProgressStep,
  NLGenerateResult,
  LLMClient,
  NLContext,
} from '@/lib/nlPipeline'

// ─────────────────────────────────────────────
// Store State
// ─────────────────────────────────────────────

export interface NLMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  /** If assistant, the generation result */
  result?: NLGenerateResult
  /** If assistant, accept/reject status */
  accepted?: boolean
  /** If system, the error message */
  error?: boolean
}

interface NLState {
  /** Multi-turn session */
  session: NLSession
  /** Chat messages for UI rendering */
  messages: NLMessage[]
  /** Current pipeline progress */
  progress: NLProgressStep | null
  /** Whether the pipeline is currently running */
  isGenerating: boolean
  /** Current input text (controlled) */
  inputText: string
  /** Last generation result (pending accept/reject) */
  pendingResult: NLGenerateResult | null
  /** API endpoint for LLM (configurable) */
  apiEndpoint: string
  /** API key (stored in memory only) */
  apiKey: string
  /** Model name for LLM */
  modelName: string
  /** Whether LLM settings panel is open */
  showSettings: boolean
  /** Available models from backend config (auto-detected from config.yaml) */
  availableModels: string[]
  /** Whether server-side endpoint is configured (no manual config needed) */
  hasServerEndpoint: boolean
  /** Whether server-side key is configured */
  hasServerKey: boolean
  /** Whether config has been fetched from backend */
  configLoaded: boolean
}

// ─────────────────────────────────────────────
// Store Actions
// ─────────────────────────────────────────────

interface NLActions {
  /** Set input text */
  setInputText(text: string): void

  /** Run the NL → DSL pipeline for the given input */
  generate(
    nlInput: string,
    wasm: WasmBridge,
    currentDSL: string,
    symbols: SymbolTable | null,
    diagnostics: Diagnostic[],
  ): Promise<NLGenerateResult | null>

  /** Accept the pending result → push DSL to editor */
  acceptResult(): { dsl: string; intentIR: IntentIR } | null

  /** Reject the pending result */
  rejectResult(): void

  /** Reset the session (new conversation) */
  resetSession(): void

  /** Update LLM settings */
  setApiEndpoint(endpoint: string): void
  setApiKey(key: string): void
  setModelName(model: string): void
  setShowSettings(show: boolean): void

  /** Fetch NL config from backend (auto-detect endpoint/model from config.yaml) */
  fetchConfig(): Promise<void>
}

export type NLStore = NLState & NLActions

// ─────────────────────────────────────────────
// LLM Client Factory
// ─────────────────────────────────────────────

function createLLMClient(
  apiEndpoint: string,
  apiKey: string,
  modelName: string,
): LLMClient {
  return {
    async generateIntentIR(systemPrompt: string, userPrompt: string): Promise<IntentIR> {
      const resp = await fetch(apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
        },
        body: JSON.stringify({
          model: modelName,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
          ],
          temperature: 0.1,
          response_format: { type: 'json_object' },
        }),
      })

      if (!resp.ok) {
        const text = await resp.text().catch(() => '')
        throw new Error(`LLM API error ${resp.status}: ${text.slice(0, 200)}`)
      }

      const data = await resp.json()

      // OpenAI-compatible response format
      const content = data.choices?.[0]?.message?.content
      if (!content) {
        throw new Error('LLM response missing choices[0].message.content')
      }

      return parseIntentIRFromLLM(content)
    },
  }
}

/**
 * Robust JSON extraction from LLM output.
 * Handles: raw JSON, markdown code blocks, JSON with surrounding text.
 */
function parseIntentIRFromLLM(output: string): IntentIR {
  // 1. Try direct parse
  try {
    return JSON.parse(output)
  } catch { /* continue */ }

  // 2. Extract from markdown code block
  const codeBlockMatch = output.match(/```(?:json)?\s*\n([\s\S]*?)\n```/)
  if (codeBlockMatch) {
    try {
      return JSON.parse(codeBlockMatch[1])
    } catch { /* continue */ }
  }

  // 3. Find first { ... } block
  const braceMatch = output.match(/\{[\s\S]*\}/)
  if (braceMatch) {
    try {
      return JSON.parse(braceMatch[0])
    } catch { /* continue */ }
  }

  throw new Error('Failed to parse Intent IR from LLM output')
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

function createMessageId(): string {
  return `msg-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`
}

// ─────────────────────────────────────────────
// Initial State
// ─────────────────────────────────────────────

const initialState: NLState = {
  session: createSession(),
  messages: [],
  progress: null,
  isGenerating: false,
  inputText: '',
  pendingResult: null,
  apiEndpoint: '/api/nl/generate',
  apiKey: '',
  modelName: '',
  showSettings: false,
  availableModels: [],
  hasServerEndpoint: false,
  hasServerKey: false,
  configLoaded: false,
}

// ─────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────

export const useNLStore = create<NLStore>((set, get) => ({
  ...initialState,

  setInputText(text: string) {
    set({ inputText: text })
  },

  async generate(
    nlInput: string,
    wasm: WasmBridge,
    currentDSL: string,
    symbols: SymbolTable | null,
    diagnostics: Diagnostic[],
  ): Promise<NLGenerateResult | null> {
    const { apiEndpoint, apiKey, modelName } = get()

    // Add user message
    const userMsg: NLMessage = {
      id: createMessageId(),
      role: 'user',
      content: nlInput,
      timestamp: Date.now(),
    }
    set(s => ({
      messages: [...s.messages, userMsg],
      inputText: '',
      isGenerating: true,
      pendingResult: null,
      progress: { stage: 'classifying' },
    }))

    const context: NLContext = {
      currentDSL: currentDSL || undefined,
      symbols: symbols || undefined,
      diagnostics: diagnostics.length > 0 ? diagnostics : undefined,
    }

    const llmClient = createLLMClient(apiEndpoint, apiKey, modelName)

    const onProgress = (step: NLProgressStep) => {
      set({ progress: step })
    }

    try {
      const result = await nlToDSL(nlInput, context, wasm, llmClient, onProgress)

      // Update session
      const session = addTurn(get().session, nlInput, result.intentIR, result.dsl)

      // Create assistant message
      const assistantMsg: NLMessage = {
        id: createMessageId(),
        role: 'assistant',
        content: result.explanation,
        timestamp: Date.now(),
        result,
      }

      set({
        session,
        messages: [...get().messages, assistantMsg],
        isGenerating: false,
        progress: null,
        pendingResult: result,
      })

      return result
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)

      const errorMsg: NLMessage = {
        id: createMessageId(),
        role: 'system',
        content: `Generation failed: ${msg}`,
        timestamp: Date.now(),
        error: true,
      }

      set({
        messages: [...get().messages, errorMsg],
        isGenerating: false,
        progress: null,
        pendingResult: null,
      })

      return null
    }
  },

  acceptResult(): { dsl: string; intentIR: IntentIR } | null {
    const { pendingResult, session } = get()
    if (!pendingResult) return null

    const updatedSession = acceptLastTurn(session, pendingResult.dsl, null)

    // Mark last assistant message as accepted
    const messages = [...get().messages]
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant' && messages[i].result) {
        messages[i] = { ...messages[i], accepted: true }
        break
      }
    }

    set({
      session: updatedSession,
      messages,
      pendingResult: null,
    })

    return { dsl: pendingResult.dsl, intentIR: pendingResult.intentIR }
  },

  rejectResult() {
    const { session } = get()
    const updatedSession = rejectLastTurn(session)

    // Mark last assistant message as rejected
    const messages = [...get().messages]
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant' && messages[i].result) {
        messages[i] = { ...messages[i], accepted: false }
        break
      }
    }

    set({
      session: updatedSession,
      messages,
      pendingResult: null,
    })
  },

  resetSession() {
    set({
      session: createSession(),
      messages: [],
      progress: null,
      isGenerating: false,
      inputText: '',
      pendingResult: null,
    })
  },

  setApiEndpoint(endpoint: string) {
    set({ apiEndpoint: endpoint })
  },

  setApiKey(key: string) {
    set({ apiKey: key })
  },

  setModelName(model: string) {
    set({ modelName: model })
  },

  setShowSettings(show: boolean) {
    set({ showSettings: show })
  },

  async fetchConfig() {
    try {
      const resp = await fetch('/api/nl/config')
      if (!resp.ok) return
      const data = await resp.json()
      const updates: Partial<NLState> = {
        configLoaded: true,
        hasServerEndpoint: !!data.has_server_endpoint,
        hasServerKey: !!data.has_server_key,
      }
      if (data.available_models?.length > 0) {
        updates.availableModels = data.available_models
      }
      // Only set model if user hasn't manually changed it
      if (!get().modelName && data.default_model) {
        updates.modelName = data.default_model
      }
      set(updates)
    } catch {
      // Silently fail — config fetch is best-effort
    }
  },
}))
