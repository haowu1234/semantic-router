import { useState, useRef, useEffect, useCallback } from 'react'
import styles from './ChatComponent.module.css'
import HeaderDisplay from './HeaderDisplay'
import MarkdownRenderer from './MarkdownRenderer'
import ThinkingAnimation from './ThinkingAnimation'
import HeaderReveal from './HeaderReveal'
import ToolCallDisplay from './ToolCallDisplay'
import { useToolCalling } from '../hooks/useToolCalling'
import { ToolCall, ToolCallResult } from '../types/tools'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  timestamp: Date
  isStreaming?: boolean
  headers?: Record<string, string>
  // Tool calling fields
  tool_calls?: ToolCall[]
  tool_call_id?: string
  toolResults?: Map<string, ToolCallResult>
}

interface ChatComponentProps {
  endpoint?: string
  defaultModel?: string
  defaultSystemPrompt?: string
  isFullscreenMode?: boolean
  enableToolCalling?: boolean
}

const ChatComponent = ({
  endpoint = '/api/router/v1/chat/completions',
  defaultModel = 'MoM',
  defaultSystemPrompt = 'You are a helpful assistant.',
  isFullscreenMode = false,
  enableToolCalling = true,
}: ChatComponentProps) => {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [model, setModel] = useState(defaultModel)
  const [systemPrompt, setSystemPrompt] = useState(defaultSystemPrompt)
  const [showSettings, setShowSettings] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showThinking, setShowThinking] = useState(false)
  const [showHeaderReveal, setShowHeaderReveal] = useState(false)
  const [pendingHeaders, setPendingHeaders] = useState<Record<string, string> | null>(null)
  const [isFullscreen] = useState(isFullscreenMode)
  const [toolCallingEnabled, setToolCallingEnabled] = useState(enableToolCalling)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  
  // Tool calling hook
  const {
    isExecutingTools,
    fetchEnabledTools,
    getToolsForRequest,
    executeTools,
    shouldContinue,
    reset: resetToolCalling,
  } = useToolCalling({
    onError: (err) => setError(err),
  })

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // When headers arrive, show HeaderReveal
  useEffect(() => {
    if (pendingHeaders && Object.keys(pendingHeaders).length > 0) {
      setShowHeaderReveal(true)
    }
  }, [pendingHeaders])

  // Toggle fullscreen mode by adding/removing class to body
  useEffect(() => {
    if (isFullscreen) {
      document.body.classList.add('playground-fullscreen')
    } else {
      document.body.classList.remove('playground-fullscreen')
    }

    return () => {
      document.body.classList.remove('playground-fullscreen')
    }
  }, [isFullscreen])

  const generateId = () => `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

  const handleThinkingComplete = useCallback(() => {
    // Thinking animation will be hidden when headers arrive
    // This callback is kept for ThinkingAnimation component compatibility
  }, [])

  const handleHeaderRevealComplete = useCallback(() => {
    setShowHeaderReveal(false)
    setPendingHeaders(null)
  }, [])

  // Build messages array for API request (including tool messages)
  const buildChatMessages = useCallback((
    currentMessages: Message[],
    newUserContent?: string
  ) => {
    const chatMessages: Array<{
      role: string
      content?: string
      tool_calls?: ToolCall[]
      tool_call_id?: string
    }> = [{ role: 'system', content: systemPrompt }]

    for (const m of currentMessages) {
      if (m.role === 'user') {
        chatMessages.push({ role: 'user', content: m.content })
      } else if (m.role === 'assistant') {
        if (m.tool_calls && m.tool_calls.length > 0) {
          // Assistant message with tool calls (no content)
          chatMessages.push({ role: 'assistant', tool_calls: m.tool_calls })
        } else if (m.content) {
          chatMessages.push({ role: 'assistant', content: m.content })
        }
      } else if (m.role === 'tool' && m.tool_call_id) {
        chatMessages.push({
          role: 'tool',
          tool_call_id: m.tool_call_id,
          content: m.content,
        })
      }
    }

    if (newUserContent) {
      chatMessages.push({ role: 'user', content: newUserContent })
    }

    return chatMessages
  }, [systemPrompt])

  // Make API request (supports both streaming and non-streaming for tool calls)
  const makeApiRequest = useCallback(async (
    chatMessages: Array<Record<string, unknown>>,
    tools?: Array<Record<string, unknown>>,
    stream = true
  ) => {
    const body: Record<string, unknown> = {
      model,
      messages: chatMessages,
      stream,
    }

    if (tools && tools.length > 0 && toolCallingEnabled) {
      body.tools = tools
      body.tool_choice = 'auto'
    }

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: abortControllerRef.current?.signal,
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`API error: ${response.status} - ${errorText}`)
    }

    return response
  }, [model, endpoint, toolCallingEnabled])

  // Extract response headers
  const extractHeaders = useCallback((response: Response) => {
    const responseHeaders: Record<string, string> = {}
    const headerKeys = [
      'x-vsr-selected-model',
      'x-vsr-selected-decision',
      'x-vsr-cache-hit',
      'x-vsr-selected-reasoning',
      'x-vsr-jailbreak-blocked',
      'x-vsr-pii-violation',
      'x-vsr-hallucination-detected',
      'x-vsr-fact-check-needed',
      'x-vsr-matched-keywords',
      'x-vsr-matched-embeddings',
      'x-vsr-matched-domains',
      'x-vsr-matched-fact-check',
      'x-vsr-matched-user-feedback',
      'x-vsr-matched-preference',
    ]

    headerKeys.forEach(key => {
      const value = response.headers.get(key)
      if (value) responseHeaders[key] = value
    })

    return responseHeaders
  }, [])

  // Process streaming response
  const processStreamingResponse = useCallback(async (
    response: Response,
    assistantMessageId: string
  ): Promise<{ content: string; toolCalls: ToolCall[] | null; headers: Record<string, string> }> => {
    const responseHeaders = extractHeaders(response)

    if (Object.keys(responseHeaders).length > 0) {
      setPendingHeaders(responseHeaders)
      setShowThinking(false)
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body')

    const decoder = new TextDecoder()
    let accumulatedContent = ''
    const chunks: Array<Record<string, unknown>> = []
    let detectedToolCalls: ToolCall[] | null = null

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })
      const lines = chunk.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim()
          if (data === '[DONE]') continue

          try {
            const parsed = JSON.parse(data)
            chunks.push(parsed)

            // Check for tool_calls in delta
            const delta = parsed.choices?.[0]?.delta
            const finishReason = parsed.choices?.[0]?.finish_reason

            if (delta?.content) {
              accumulatedContent += delta.content
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantMessageId
                    ? { ...m, content: accumulatedContent }
                    : m
                )
              )
            }

            // Accumulate tool_calls from delta
            if (delta?.tool_calls) {
              if (!detectedToolCalls) detectedToolCalls = []
              for (const tc of delta.tool_calls) {
                const idx = tc.index
                if (!detectedToolCalls[idx]) {
                  detectedToolCalls[idx] = {
                    id: tc.id || `call_${idx}`,
                    type: 'function',
                    function: { name: tc.function?.name || '', arguments: '' },
                  }
                }
                if (tc.function?.name) {
                  detectedToolCalls[idx].function.name = tc.function.name
                }
                if (tc.function?.arguments) {
                  detectedToolCalls[idx].function.arguments += tc.function.arguments
                }
              }
            }

            // Check finish reason
            if (finishReason === 'tool_calls' && detectedToolCalls) {
              // Tool calls detected, stop content streaming
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }

    // Filter out undefined entries from tool calls array
    if (detectedToolCalls) {
      detectedToolCalls = detectedToolCalls.filter(Boolean)
      if (detectedToolCalls.length === 0) detectedToolCalls = null
    }

    return { content: accumulatedContent, toolCalls: detectedToolCalls, headers: responseHeaders }
  }, [extractHeaders])

  // Handle tool calling loop
  const handleToolCallingLoop = useCallback(async (
    toolCalls: ToolCall[],
    currentMessages: Message[],
    assistantMessageId: string,
    tools: Array<Record<string, unknown>>
  ) => {
    let iterationMessages = [...currentMessages]
    let currentToolCalls = toolCalls

    while (currentToolCalls && shouldContinue()) {
      // Update assistant message with tool_calls
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMessageId
            ? { ...m, tool_calls: currentToolCalls, isStreaming: false }
            : m
        )
      )

      // Execute tools
      const results = await executeTools(currentToolCalls)

      // Create tool result messages
      const toolMessages: Message[] = results.map(result => ({
        id: generateId(),
        role: 'tool' as const,
        content: result.content,
        tool_call_id: result.tool_call_id,
        timestamp: new Date(),
      }))

      // Update messages state
      setMessages(prev => [...prev, ...toolMessages])

      // Update iteration messages for next API call
      iterationMessages = [
        ...iterationMessages.map(m =>
          m.id === assistantMessageId
            ? { ...m, tool_calls: currentToolCalls }
            : m
        ),
        ...toolMessages,
      ]

      // Create new assistant message for continuation
      const newAssistantId = generateId()
      const newAssistantMessage: Message = {
        id: newAssistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
      }
      setMessages(prev => [...prev, newAssistantMessage])

      // Continue conversation with tool results
      const chatMessages = buildChatMessages(iterationMessages)
      const response = await makeApiRequest(chatMessages, tools, true)
      const result = await processStreamingResponse(response, newAssistantId)

      // Update the new assistant message
      setMessages(prev =>
        prev.map(m =>
          m.id === newAssistantId
            ? {
                ...m,
                content: result.content,
                isStreaming: false,
                headers: Object.keys(result.headers).length > 0 ? result.headers : undefined,
              }
            : m
        )
      )

      // Check if there are more tool calls
      currentToolCalls = result.toolCalls!
      if (!currentToolCalls) break

      // Update for next iteration
      iterationMessages = [
        ...iterationMessages,
        { ...newAssistantMessage, content: result.content },
      ]
    }
  }, [shouldContinue, executeTools, buildChatMessages, makeApiRequest, processStreamingResponse])

  const handleSend = async () => {
    const trimmedInput = inputValue.trim()
    if (!trimmedInput || isLoading) return

    setError(null)
    resetToolCalling()

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    setPendingHeaders(null)
    setShowHeaderReveal(false)
    setShowThinking(true)

    const assistantMessageId = generateId()
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
    }
    setMessages(prev => [...prev, assistantMessage])

    try {
      abortControllerRef.current = new AbortController()

      // Fetch enabled tools if tool calling is enabled
      let tools: Array<Record<string, unknown>> = []
      if (toolCallingEnabled) {
        await fetchEnabledTools()
        tools = getToolsForRequest()
      }

      // Build initial messages
      const currentMessages = [...messages, userMessage]
      const chatMessages = buildChatMessages(messages, trimmedInput)

      // Make API request
      const response = await makeApiRequest(chatMessages, tools, true)
      const result = await processStreamingResponse(response, assistantMessageId)

      // Update assistant message
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMessageId
            ? {
                ...m,
                content: result.content,
                tool_calls: result.toolCalls || undefined,
                isStreaming: false,
                headers: Object.keys(result.headers).length > 0 ? result.headers : undefined,
              }
            : m
        )
      )

      // Handle tool calls if present
      if (result.toolCalls && result.toolCalls.length > 0) {
        await handleToolCallingLoop(
          result.toolCalls,
          [...currentMessages, { ...assistantMessage, content: result.content }],
          assistantMessageId,
          tools
        )
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        return
      }
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      setMessages(prev => prev.filter(m => m.id !== assistantMessageId))
    } finally {
      setIsLoading(false)
      setShowThinking(false)
      abortControllerRef.current = null
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleStop = () => {
    abortControllerRef.current?.abort()
    setIsLoading(false)
  }

  const handleClear = () => {
    setMessages([])
    setError(null)
  }

  return (
    <>
      {/* Thinking Animation */}
      {showThinking && (
        <ThinkingAnimation onComplete={handleThinkingComplete} />
      )}

      {/* Header Reveal */}
      {showHeaderReveal && pendingHeaders && (
        <HeaderReveal
          headers={pendingHeaders}
          onComplete={handleHeaderRevealComplete}
          displayDuration={2000}
        />
      )}

      <div className={`${styles.container} ${isFullscreen ? styles.fullscreen : ''}`}>
      {showSettings && (
        <div className={styles.settings}>
          <div className={styles.settingsHeader}>
            <span className={styles.settingsTitle}>Settings</span>
            <button
              className={styles.iconButton}
              onClick={() => setShowSettings(false)}
              title="Close settings"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M1 1l12 12M13 1L1 13" strokeLinecap="round"/>
              </svg>
            </button>
          </div>
          <div className={styles.settingRow}>
            <label className={styles.settingLabel}>Model:</label>
            <input
              type="text"
              value={model}
              onChange={e => setModel(e.target.value)}
              className={styles.settingInput}
              placeholder="auto, gpt-4, etc."
            />
          </div>
          <div className={styles.settingRow}>
            <label className={styles.settingLabel}>System Prompt:</label>
            <textarea
              value={systemPrompt}
              onChange={e => setSystemPrompt(e.target.value)}
              className={styles.settingTextarea}
              rows={3}
              placeholder="You are a helpful assistant."
            />
          </div>
          <div className={styles.settingRow}>
            <label className={styles.settingLabel}>
              <input
                type="checkbox"
                checked={toolCallingEnabled}
                onChange={e => setToolCallingEnabled(e.target.checked)}
                className={styles.settingCheckbox}
              />
              Enable Tool Calling
            </label>
          </div>
        </div>
      )}

      {error && (
        <div className={styles.error}>
          <span className={styles.errorIcon}>⚠️</span>
          <span>{error}</span>
          <button
            className={styles.errorDismiss}
            onClick={() => setError(null)}
          >
            ×
          </button>
        </div>
      )}

      <div className={styles.messagesContainer}>
        {messages.length === 0 ? (
          <div className={styles.emptyState}>
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" opacity="0.3">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <h3>Start a conversation</h3>
            <p>Send a message to begin chatting with the mixture of models.</p>
          </div>
        ) : (
          <div className={styles.messages}>
            {messages.map(message => {
              // Skip rendering tool messages separately - they're shown in ToolCallDisplay
              if (message.role === 'tool') return null

              return (
                <div
                  key={message.id}
                  className={`${styles.message} ${styles[message.role]}`}
                >
                  <div className={styles.messageAvatar}>
                    {message.role === 'user' ? (
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    ) : (
                      <img src="/vllm.png" alt="vLLM SR" className={styles.avatarImage} />
                    )}
                  </div>
                  <div className={styles.messageContent}>
                    <div className={styles.messageRole}>
                      {message.role === 'user' ? 'You' : 'vLLM SR'}
                    </div>
                    
                    {/* Tool Calls Display */}
                    {message.role === 'assistant' && message.tool_calls && message.tool_calls.length > 0 && (
                      <div className={styles.toolCalls}>
                        {message.tool_calls.map(toolCall => {
                          // Find corresponding tool result
                          const toolResult = messages.find(
                            m => m.role === 'tool' && m.tool_call_id === toolCall.id
                          )
                          return (
                            <ToolCallDisplay
                              key={toolCall.id}
                              toolCall={toolCall}
                              result={toolResult ? {
                                tool_call_id: toolResult.tool_call_id!,
                                role: 'tool',
                                content: toolResult.content,
                                success: !toolResult.content.startsWith('Error'),
                              } : undefined}
                              isExecuting={isExecutingTools && !toolResult}
                            />
                          )
                        })}
                      </div>
                    )}

                    <div className={styles.messageText}>
                      {message.role === 'assistant' && message.content ? (
                        <>
                          <MarkdownRenderer content={message.content} />
                          {message.isStreaming && (
                            <span className={styles.cursor}>▊</span>
                          )}
                        </>
                      ) : (
                        <>
                          {message.content || (message.isStreaming && (
                            <span className={styles.cursor}>▊</span>
                          ))}
                          {message.isStreaming && message.content && (
                            <span className={styles.cursor}>▊</span>
                          )}
                        </>
                      )}
                    </div>
                    {message.role === 'assistant' && message.headers && (
                      <HeaderDisplay headers={message.headers} />
                    )}
                  </div>
                </div>
              )
            })}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className={styles.inputContainer}>
        <div className={styles.inputActions}>
          <button
            className={styles.inputActionButton}
            onClick={() => setShowSettings(!showSettings)}
            title="Settings"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="8" cy="8" r="2.5"/>
              <path d="M8 1v2M8 13v2M15 8h-2M3 8H1M13.5 2.5l-1.4 1.4M3.9 12.1l-1.4 1.4M13.5 13.5l-1.4-1.4M3.9 3.9L2.5 2.5" strokeLinecap="round"/>
            </svg>
          </button>
          <button
            className={styles.inputActionButton}
            onClick={handleClear}
            title="Clear chat"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2 4h12M5.5 4V2.5h5V4M13 4v9.5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4M6.5 7v4M9.5 7v4" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (Enter to send)"
          className={styles.input}
          rows={1}
          disabled={isLoading}
        />
        {isLoading ? (
          <button
            className={`${styles.sendButton} ${styles.stopButton}`}
            onClick={handleStop}
            title="Stop generating"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="6" width="12" height="12" rx="1"/>
            </svg>
          </button>
        ) : (
          <button
            className={styles.sendButton}
            onClick={handleSend}
            disabled={!inputValue.trim()}
            title="Send message"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        )}
      </div>
    </div>
    </>
  )
}

export default ChatComponent

