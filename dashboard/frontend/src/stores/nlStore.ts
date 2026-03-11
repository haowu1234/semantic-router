/**
 * Zustand store for NL Mode session state.
 *
 * Manages:
 * - Multi-turn NL conversation (input → IR → DSL)
 * - Pipeline progress tracking (classifying / generating / validating / repairing / done)
 * - Result accept/reject with DSL Store integration
 * - Session lifecycle (create / reset)
 * - LLM client configuration
 * - Streaming support for real-time generation feedback
 */

import { create } from 'zustand'
import type { Diagnostic, SymbolTable, WasmBridge } from '@/types/dsl'
import type { IntentIR, Intent } from '@/types/intentIR'
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
  ConversationTurn,
} from '@/lib/nlPipeline'
import {
  createPartialAcceptState,
  updatePartialAcceptState,
  toggleSelection,
  setAllSelections,
  autoSelectDependencies,
  updateIntentEdit,
  revertIntentEdit,
  buildPartialIntentIR,
  generatePartialPreview,
  selectByCategory,
  invertSelections,
  getSelectionStats,
} from '@/lib/nlPartialAccept'
import type {
  PartialAcceptState,
  IntentSelectionState,
} from '@/lib/nlPartialAccept'
import { intentIRToDSL } from '@/lib/intentToDsl'

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
  /** Streaming text for real-time feedback */
  streamingText: string
  /** Whether streaming is enabled */
  streamingEnabled: boolean
  /** Conversation history for multi-turn context */
  conversationHistory: ConversationTurn[]
  /** Partial accept state (when in partial accept mode) */
  partialAcceptState: PartialAcceptState | null
  /** Whether partial accept mode is active */
  isPartialAcceptMode: boolean
  /** WASM bridge reference for partial accept preview */
  wasmRef: WasmBridge | null
  /** Current DSL reference for partial accept */
  currentDSLRef: string
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
  setStreamingEnabled(enabled: boolean): void

  /** Fetch NL config from backend (auto-detect endpoint/model from config.yaml) */
  fetchConfig(): Promise<void>

  /** Clear streaming text */
  clearStreamingText(): void

  // ─────────────────────────────────────────────
  // Partial Accept Actions
  // ─────────────────────────────────────────────

  /** Enter partial accept mode for the pending result */
  enterPartialAcceptMode(wasm: WasmBridge, currentDSL: string, symbols: SymbolTable | null): void

  /** Exit partial accept mode without applying changes */
  exitPartialAcceptMode(): void

  /** Toggle selection of a single intent */
  toggleIntentSelection(index: number): void

  /** Select or deselect all intents */
  selectAllIntents(selected: boolean): void

  /** Auto-select all dependencies for currently selected intents */
  autoSelectDependencies(): void

  /** Select/deselect all intents of a specific category */
  selectCategory(category: string, selected: boolean): void

  /** Invert all selections */
  invertSelections(): void

  /** Edit a single intent */
  editIntent(index: number, editedIntent: Intent): void

  /** Revert an intent to its original state */
  revertIntentEdit(index: number): void

  /** Refresh the preview DSL based on current selection */
  refreshPartialPreview(): Promise<void>

  /** Apply partial accept - only accept selected intents */
  applyPartialAccept(): { dsl: string; intentIR: IntentIR } | null
}

export type NLStore = NLState & NLActions

// ─────────────────────────────────────────────
// LLM Client Factory
// ─────────────────────────────────────────────

function createLLMClient(
  apiEndpoint: string,
  apiKey: string,
  modelName: string,
  onStreamingUpdate?: (text: string) => void,
): LLMClient {
  return {
    async generateIntentIR(systemPrompt: string, userPrompt: string): Promise<IntentIR> {
      const useStreaming = !!onStreamingUpdate

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
          stream: useStreaming,
        }),
      })

      if (!resp.ok) {
        const text = await resp.text().catch(() => '')
        throw new Error(`LLM API error ${resp.status}: ${text.slice(0, 200)}`)
      }

      if (useStreaming && resp.body) {
        // Handle streaming response
        return handleStreamingResponse(resp.body, onStreamingUpdate!)
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
 * Handle SSE streaming response from LLM API.
 */
async function handleStreamingResponse(
  body: ReadableStream<Uint8Array>,
  onUpdate: (text: string) => void,
): Promise<IntentIR> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let fullContent = ''
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // Process SSE lines
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim()

          if (data === '[DONE]') continue

          try {
            const parsed = JSON.parse(data)
            const delta = parsed.choices?.[0]?.delta?.content
            if (delta) {
              fullContent += delta
              onUpdate(fullContent)
            }
          } catch {
            // Ignore parse errors for partial data
          }
        }
      }
    }
  } finally {
    reader.releaseLock()
  }

  return parseIntentIRFromLLM(fullContent)
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
  streamingText: '',
  streamingEnabled: true,
  conversationHistory: [],
  partialAcceptState: null,
  isPartialAcceptMode: false,
  wasmRef: null,
  currentDSLRef: '',
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
    const { apiEndpoint, apiKey, modelName, streamingEnabled, conversationHistory } = get()

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
      streamingText: '',
    }))

    const context: NLContext = {
      currentDSL: currentDSL || undefined,
      symbols: symbols || undefined,
      diagnostics: diagnostics.length > 0 ? diagnostics : undefined,
      conversationHistory: conversationHistory.length > 0 ? conversationHistory : undefined,
      useDynamicPrompt: true, // Enable token savings
    }

    // Create LLM client with optional streaming callback
    const onStreamingUpdate = streamingEnabled
      ? (text: string) => set({ streamingText: text })
      : undefined
    const llmClient = createLLMClient(apiEndpoint, apiKey, modelName, onStreamingUpdate)

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
        streamingText: '',
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
        streamingText: '',
      })

      return null
    }
  },

  acceptResult(): { dsl: string; intentIR: IntentIR } | null {
    const { pendingResult, session, messages } = get()
    if (!pendingResult) return null

    const updatedSession = acceptLastTurn(session, pendingResult.dsl, null)

    // Mark last assistant message as accepted
    const updatedMessages = [...messages]
    for (let i = updatedMessages.length - 1; i >= 0; i--) {
      if (updatedMessages[i].role === 'assistant' && updatedMessages[i].result) {
        updatedMessages[i] = { ...updatedMessages[i], accepted: true }
        break
      }
    }

    // Add to conversation history for multi-turn context
    const lastUserMsg = messages.filter(m => m.role === 'user').pop()
    const newHistoryEntry: ConversationTurn = {
      userInput: lastUserMsg?.content || '',
      summary: pendingResult.explanation,
      accepted: true,
    }

    set({
      session: updatedSession,
      messages: updatedMessages,
      pendingResult: null,
      conversationHistory: [...get().conversationHistory, newHistoryEntry],
    })

    return { dsl: pendingResult.dsl, intentIR: pendingResult.intentIR }
  },

  rejectResult() {
    const { session, messages } = get()
    const updatedSession = rejectLastTurn(session)

    // Mark last assistant message as rejected
    const updatedMessages = [...messages]
    for (let i = updatedMessages.length - 1; i >= 0; i--) {
      if (updatedMessages[i].role === 'assistant' && updatedMessages[i].result) {
        updatedMessages[i] = { ...updatedMessages[i], accepted: false }
        break
      }
    }

    // Add to conversation history as rejected
    const lastUserMsg = messages.filter(m => m.role === 'user').pop()
    const pendingResult = get().pendingResult
    if (pendingResult) {
      const newHistoryEntry: ConversationTurn = {
        userInput: lastUserMsg?.content || '',
        summary: pendingResult.explanation,
        accepted: false,
      }

      set({
        session: updatedSession,
        messages: updatedMessages,
        pendingResult: null,
        conversationHistory: [...get().conversationHistory, newHistoryEntry],
      })
    } else {
      set({
        session: updatedSession,
        messages: updatedMessages,
        pendingResult: null,
      })
    }
  },

  resetSession() {
    set({
      session: createSession(),
      messages: [],
      progress: null,
      isGenerating: false,
      inputText: '',
      pendingResult: null,
      streamingText: '',
      conversationHistory: [],
      partialAcceptState: null,
      isPartialAcceptMode: false,
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

  setStreamingEnabled(enabled: boolean) {
    set({ streamingEnabled: enabled })
  },

  clearStreamingText() {
    set({ streamingText: '' })
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

  // ─────────────────────────────────────────────
  // Partial Accept Actions
  // ─────────────────────────────────────────────

  enterPartialAcceptMode(wasm: WasmBridge, currentDSL: string, symbols: SymbolTable | null) {
    const { pendingResult } = get()
    if (!pendingResult) return

    const partialState = createPartialAcceptState(
      pendingResult.intentIR,
      currentDSL,
      symbols ?? undefined,
    )

    set({
      isPartialAcceptMode: true,
      partialAcceptState: partialState,
      wasmRef: wasm,
      currentDSLRef: currentDSL,
    })

    // Generate initial preview
    get().refreshPartialPreview()
  },

  exitPartialAcceptMode() {
    set({
      isPartialAcceptMode: false,
      partialAcceptState: null,
    })
  },

  toggleIntentSelection(index: number) {
    const { partialAcceptState } = get()
    if (!partialAcceptState) return

    const newSelections = toggleSelection(partialAcceptState.selections, index)
    const newState = updatePartialAcceptState(partialAcceptState, newSelections)

    set({ partialAcceptState: newState })

    // Debounced preview refresh
    get().refreshPartialPreview()
  },

  selectAllIntents(selected: boolean) {
    const { partialAcceptState } = get()
    if (!partialAcceptState) return

    const newSelections = setAllSelections(partialAcceptState.selections, selected)
    const newState = updatePartialAcceptState(partialAcceptState, newSelections)

    set({ partialAcceptState: newState })
    get().refreshPartialPreview()
  },

  autoSelectDependencies() {
    const { partialAcceptState } = get()
    if (!partialAcceptState) return

    const newSelections = autoSelectDependencies(
      partialAcceptState.selections,
      partialAcceptState.dependencyGraph,
    )
    const newState = updatePartialAcceptState(partialAcceptState, newSelections)

    set({ partialAcceptState: newState })
    get().refreshPartialPreview()
  },

  selectCategory(category: string, selected: boolean) {
    const { partialAcceptState } = get()
    if (!partialAcceptState) return

    const newSelections = selectByCategory(
      partialAcceptState.originalIR,
      partialAcceptState.selections,
      category,
      selected,
    )
    const newState = updatePartialAcceptState(partialAcceptState, newSelections)

    set({ partialAcceptState: newState })
    get().refreshPartialPreview()
  },

  invertSelections() {
    const { partialAcceptState } = get()
    if (!partialAcceptState) return

    const newSelections = invertSelections(partialAcceptState.selections)
    const newState = updatePartialAcceptState(partialAcceptState, newSelections)

    set({ partialAcceptState: newState })
    get().refreshPartialPreview()
  },

  editIntent(index: number, editedIntent: Intent) {
    const { partialAcceptState } = get()
    if (!partialAcceptState) return

    const newSelections = updateIntentEdit(
      partialAcceptState.selections,
      index,
      editedIntent,
    )
    const newState = updatePartialAcceptState(partialAcceptState, newSelections)

    set({ partialAcceptState: newState })
    get().refreshPartialPreview()
  },

  revertIntentEdit(index: number) {
    const { partialAcceptState } = get()
    if (!partialAcceptState) return

    const newSelections = revertIntentEdit(partialAcceptState.selections, index)
    const newState = updatePartialAcceptState(partialAcceptState, newSelections)

    set({ partialAcceptState: newState })
    get().refreshPartialPreview()
  },

  async refreshPartialPreview() {
    const { partialAcceptState, wasmRef, currentDSLRef } = get()
    if (!partialAcceptState || !wasmRef) return

    try {
      const { dsl, validation } = await generatePartialPreview(
        partialAcceptState.originalIR,
        partialAcceptState.selections,
        currentDSLRef || undefined,
        wasmRef,
      )

      set({
        partialAcceptState: {
          ...partialAcceptState,
          previewDSL: dsl,
          previewValidation: validation,
        },
      })
    } catch (err) {
      console.error('[NL] Failed to generate partial preview:', err)
    }
  },

  applyPartialAccept(): { dsl: string; intentIR: IntentIR } | null {
    const { partialAcceptState, pendingResult, session, messages, currentDSLRef } = get()
    if (!partialAcceptState || !pendingResult) return null

    // Build the partial IntentIR with only selected intents
    const partialIR = buildPartialIntentIR(
      partialAcceptState.originalIR,
      partialAcceptState.selections,
    )

    // Use the preview DSL if available, otherwise generate
    const dsl = partialAcceptState.previewDSL || intentIRToDSL(partialIR, currentDSLRef || undefined)

    // Update session
    const updatedSession = acceptLastTurn(session, dsl, null)

    // Mark last assistant message as accepted (partially)
    const updatedMessages = [...messages]
    for (let i = updatedMessages.length - 1; i >= 0; i--) {
      if (updatedMessages[i].role === 'assistant' && updatedMessages[i].result) {
        updatedMessages[i] = { ...updatedMessages[i], accepted: true }
        break
      }
    }

    // Build conversation history summary
    const stats = getSelectionStats(partialAcceptState.selections)
    const lastUserMsg = messages.filter(m => m.role === 'user').pop()
    const newHistoryEntry: ConversationTurn = {
      userInput: lastUserMsg?.content || '',
      summary: `${pendingResult.explanation} (partial: ${stats.selected}/${stats.total} items accepted)`,
      accepted: true,
    }

    set({
      session: updatedSession,
      messages: updatedMessages,
      pendingResult: null,
      conversationHistory: [...get().conversationHistory, newHistoryEntry],
      isPartialAcceptMode: false,
      partialAcceptState: null,
    })

    return { dsl, intentIR: partialIR }
  },
}))
