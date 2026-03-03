/**
 * NLMode — Natural Language chat interface for the Config Builder.
 *
 * Provides a conversational UI where users describe routing configs in natural
 * language. The pipeline (nlPipeline.ts) converts NL → Intent IR → DSL, with
 * WASM validation and self-repair. Users can accept/reject generated DSL
 * before it's applied to the editor.
 *
 * Layout:
 *   ┌────────────────────────────────────────┐
 *   │  Settings bar (endpoint, model, reset) │
 *   ├────────────────────────────────────────┤
 *   │                                        │
 *   │  Chat messages (scrollable)            │
 *   │    - User messages                     │
 *   │    - Assistant responses with DSL      │
 *   │    - Accept/Reject controls            │
 *   │    - Progress indicators               │
 *   │                                        │
 *   ├────────────────────────────────────────┤
 *   │  Input area (textarea + send button)   │
 *   └────────────────────────────────────────┘
 */

import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useNLStore } from '@/stores/nlStore'
import type { NLMessage } from '@/stores/nlStore'
import type { NLProgressStep } from '@/lib/nlPipeline'
import { useDSLStore } from '@/stores/dslStore'
import { wasmBridge } from '@/lib/wasm'
import styles from './NLMode.module.css'

// ─────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────

const NLMode: React.FC = () => {
  const {
    messages,
    progress,
    isGenerating,
    inputText,
    pendingResult,
    apiEndpoint,
    apiKey,
    modelName,
    showSettings,
    availableModels,
    hasServerEndpoint,
    hasServerKey,
    configLoaded,
    setInputText,
    generate,
    acceptResult,
    rejectResult,
    resetSession,
    setApiEndpoint,
    setApiKey,
    setModelName,
    setShowSettings,
    fetchConfig,
  } = useNLStore()

  const {
    dslSource,
    symbols,
    diagnostics,
    wasmReady,
    setDslSource,
    compile,
  } = useDSLStore()

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Fetch NL config from backend on mount (auto-detect endpoint/model)
  useEffect(() => {
    if (!configLoaded) {
      fetchConfig()
    }
  }, [configLoaded, fetchConfig])

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, progress])

  // Auto-resize textarea
  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setInputText(e.target.value)
      const el = e.target
      el.style.height = 'auto'
      el.style.height = `${Math.min(el.scrollHeight, 160)}px`
    },
    [setInputText],
  )

  // Submit on Enter (Shift+Enter for newline)
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [inputText, isGenerating, wasmReady],
  )

  const handleSend = useCallback(() => {
    const trimmed = inputText.trim()
    if (!trimmed || isGenerating || !wasmReady) return

    generate(
      trimmed,
      wasmBridge,
      dslSource,
      symbols,
      diagnostics,
    )
  }, [inputText, isGenerating, wasmReady, generate, dslSource, symbols, diagnostics])

  const handleAccept = useCallback(() => {
    const result = acceptResult()
    if (result) {
      // Push DSL to the editor store — this triggers WASM validation via setDslSource
      setDslSource(result.dsl)
      // Also trigger a full compile so YAML/CRD output is ready
      setTimeout(() => compile(), 50)
    }
  }, [acceptResult, setDslSource, compile])

  const handleReject = useCallback(() => {
    rejectResult()
    // Re-focus input for retry
    textareaRef.current?.focus()
  }, [rejectResult])

  return (
    <div className={styles.container}>
      {/* Settings bar */}
      <div className={styles.settingsBar}>
        <button
          className={styles.settingsToggle}
          onClick={() => setShowSettings(!showSettings)}
          title="LLM Settings"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="8" cy="8" r="3" />
            <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41" strokeLinecap="round" />
          </svg>
        </button>

        <span className={styles.settingsLabel}>
          NL Mode
          {modelName && <span className={styles.modelBadge}>{modelName}</span>}
        </span>

        <div className={styles.settingsActions}>
          <button
            className={styles.resetBtn}
            onClick={resetSession}
            title="New conversation"
            disabled={isGenerating}
          >
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2 8a6 6 0 0111.47-2.47M14 8a6 6 0 01-11.47 2.47" strokeLinecap="round" />
              <path d="M14 2v4h-4M2 14v-4h4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            New
          </button>
        </div>
      </div>

      {/* Settings panel (collapsible) */}
      {showSettings && (
        <div className={styles.settingsPanel}>
          <div className={styles.settingsField}>
            <label>
              API Endpoint
              {hasServerEndpoint && <span className={styles.autoTag}>auto</span>}
            </label>
            <input
              type="text"
              value={apiEndpoint}
              onChange={e => setApiEndpoint(e.target.value)}
              placeholder="/api/nl/generate"
              className={styles.settingsInput}
            />
          </div>
          <div className={styles.settingsField}>
            <label>
              API Key
              {hasServerKey && <span className={styles.autoTag}>server</span>}
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder={hasServerKey ? 'Using server-side key' : 'Optional — for direct LLM access'}
              className={styles.settingsInput}
              disabled={hasServerKey}
            />
          </div>
          <div className={styles.settingsField}>
            <label>Model</label>
            {availableModels.length > 0 ? (
              <select
                value={modelName}
                onChange={e => setModelName(e.target.value)}
                className={styles.settingsInput}
              >
                {!modelName && <option value="">Select a model...</option>}
                {availableModels.map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={modelName}
                onChange={e => setModelName(e.target.value)}
                placeholder="Model name (e.g., qwen3-32b)"
                className={styles.settingsInput}
              />
            )}
          </div>
        </div>
      )}

      {/* Chat messages area */}
      <div className={styles.chatArea}>
        {messages.length === 0 && !progress && (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>
              <svg width="40" height="40" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="6" y="10" width="36" height="28" rx="4" />
                <path d="M6 18h36" />
                <circle cx="14" cy="14" r="1.5" fill="currentColor" stroke="none" />
                <circle cx="20" cy="14" r="1.5" fill="currentColor" stroke="none" />
                <circle cx="26" cy="14" r="1.5" fill="currentColor" stroke="none" />
                <path d="M14 26h8M14 31h16" strokeLinecap="round" />
              </svg>
            </div>
            <div className={styles.emptyTitle}>Natural Language Mode</div>
            <div className={styles.emptyDesc}>
              Describe your routing configuration in plain English.
              The AI will generate DSL code for you.
            </div>
            <div className={styles.emptyExamples}>
              <div className={styles.exampleLabel}>Try something like:</div>
              <button className={styles.exampleBtn} onClick={() => setInputText('Route math questions to GPT-4o and coding questions to DeepSeek-V3')}>
                "Route math questions to GPT-4o and coding questions to DeepSeek-V3"
              </button>
              <button className={styles.exampleBtn} onClick={() => setInputText('Create a config with PII protection and semantic caching')}>
                "Create a config with PII protection and semantic caching"
              </button>
              <button className={styles.exampleBtn} onClick={() => setInputText('Set up three tiers: simple queries to a small model, medium to GPT-4o, and hard problems to DeepSeek-R1')}>
                "Set up three tiers: simple, medium, and hard routing"
              </button>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* Pending accept/reject for last result */}
        {pendingResult && !isGenerating && (
          <div className={styles.acceptRejectBar}>
            <div className={styles.acceptRejectInfo}>
              <span className={pendingResult.isValid ? styles.validBadge : styles.invalidBadge}>
                {pendingResult.isValid ? '✓ Valid' : '⚠ Has issues'}
              </span>
              <span className={styles.confidenceBadge}>
                Confidence: {Math.round(pendingResult.confidence * 100)}%
              </span>
              {pendingResult.retries > 0 && (
                <span className={styles.retriesBadge}>
                  {pendingResult.retries} repair{pendingResult.retries > 1 ? 's' : ''}
                </span>
              )}
            </div>
            <div className={styles.acceptRejectBtns}>
              <button className={styles.acceptBtn} onClick={handleAccept}>
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Apply to Editor
              </button>
              <button className={styles.rejectBtn} onClick={handleReject}>
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
                </svg>
                Discard
              </button>
            </div>
          </div>
        )}

        {/* Progress indicator */}
        {isGenerating && progress && (
          <ProgressIndicator step={progress} />
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className={styles.inputArea}>
        <div className={styles.inputRow}>
          <textarea
            ref={textareaRef}
            className={styles.textarea}
            value={inputText}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={
              !wasmReady
                ? 'Waiting for WASM compiler...'
                : dslSource.trim()
                  ? 'Describe changes to make... (e.g., "Add PII protection to all routes")'
                  : 'Describe your routing config... (e.g., "Route math questions to GPT-4o")'
            }
            disabled={!wasmReady || isGenerating}
            rows={1}
          />
          <button
            className={styles.sendBtn}
            onClick={handleSend}
            disabled={!inputText.trim() || isGenerating || !wasmReady}
            title="Send (Enter)"
          >
            {isGenerating ? (
              <span className={styles.spinner} />
            ) : (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M2.5 2.5l11 5.5-11 5.5V9.25L9 8 2.5 6.75V2.5z" />
              </svg>
            )}
          </button>
        </div>
        <div className={styles.inputHint}>
          {dslSource.trim()
            ? 'Modify mode — changes will be applied to current DSL'
            : 'Generate mode — will create new DSL from scratch'}
          <span className={styles.shortcutHint}>Enter to send, Shift+Enter for newline</span>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// Message Bubble
// ─────────────────────────────────────────────

const MessageBubble: React.FC<{ message: NLMessage }> = ({ message }) => {
  const [showDSL, setShowDSL] = useState(false)
  const [showIR, setShowIR] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCopyDSL = useCallback(() => {
    if (message.result?.dsl) {
      navigator.clipboard.writeText(message.result.dsl).then(() => {
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      })
    }
  }, [message.result])

  if (message.role === 'user') {
    return (
      <div className={styles.messageBubbleUser}>
        <div className={styles.messageContent}>{message.content}</div>
      </div>
    )
  }

  if (message.role === 'system' && message.error) {
    return (
      <div className={styles.messageBubbleError}>
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="8" cy="8" r="6" />
          <path d="M8 5v4M8 11h.01" strokeLinecap="round" />
        </svg>
        <div className={styles.messageContent}>{message.content}</div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className={`${styles.messageBubbleAssistant} ${message.accepted === true ? styles.accepted : message.accepted === false ? styles.rejected : ''}`}>
      <div className={styles.messageContent}>
        {message.content.split('\n').map((line, i) => (
          <div key={i}>{line || '\u00A0'}</div>
        ))}
      </div>

      {message.result && (
        <div className={styles.resultActions}>
          <button
            className={styles.toggleBtn}
            onClick={() => setShowDSL(!showDSL)}
          >
            {showDSL ? '▾ Hide DSL' : '▸ Show DSL'}
            {message.result.dsl && (
              <span className={styles.dslLineCount}>
                {message.result.dsl.split('\n').length} lines
              </span>
            )}
          </button>
          <button
            className={styles.toggleBtn}
            onClick={() => setShowIR(!showIR)}
          >
            {showIR ? '▾ Hide IR' : '▸ Show IR'}
          </button>
          {showDSL && (
            <button className={styles.copyBtn} onClick={handleCopyDSL}>
              {copied ? 'Copied!' : 'Copy DSL'}
            </button>
          )}
        </div>
      )}

      {showDSL && message.result?.dsl && (
        <pre className={styles.dslPreview}>{message.result.dsl}</pre>
      )}

      {showIR && message.result?.intentIR && (
        <pre className={styles.irPreview}>
          {JSON.stringify(message.result.intentIR, null, 2)}
        </pre>
      )}

      {message.result?.diagnostics && message.result.diagnostics.length > 0 && (
        <div className={styles.diagnosticsList}>
          {message.result.diagnostics.map((d, i) => (
            <div key={i} className={d.level === 'error' ? styles.diagError : styles.diagWarning}>
              {d.level === 'error' ? '✕' : '⚠'} {d.message}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────
// Progress Indicator
// ─────────────────────────────────────────────

const ProgressIndicator: React.FC<{ step: NLProgressStep }> = ({ step }) => {
  const labels: Record<string, string> = {
    classifying: 'Analyzing your request...',
    generating: 'message' in step ? (step as { message: string }).message : 'Generating...',
    validating: 'Validating generated DSL...',
    repairing: 'attempt' in step
      ? `Repairing (${(step as { attempt: number; strategy: string }).strategy}, attempt ${(step as { attempt: number }).attempt}/${(step as { maxRetries: number }).maxRetries})...`
      : 'Repairing...',
    done: 'isValid' in step && (step as { isValid: boolean }).isValid ? 'Done — valid DSL generated!' : 'Done — review the result below.',
    error: 'message' in step ? (step as { message: string }).message : 'An error occurred.',
  }

  return (
    <div className={styles.progressIndicator}>
      <span className={step.stage === 'error' ? styles.progressError : styles.progressDot}>
        {step.stage === 'done' ? (
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : step.stage === 'error' ? (
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
          </svg>
        ) : (
          <span className={styles.spinner} />
        )}
      </span>
      <span className={styles.progressLabel}>
        {labels[step.stage] || 'Processing...'}
      </span>
    </div>
  )
}

export default NLMode
