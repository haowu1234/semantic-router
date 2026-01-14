import React, { useState } from 'react'
import { ToolCall, ToolCallResult } from '../types/tools'
import styles from './ToolCallDisplay.module.css'

interface ToolCallDisplayProps {
  toolCall: ToolCall
  result?: ToolCallResult
  isExecuting?: boolean
}

const ToolCallDisplay: React.FC<ToolCallDisplayProps> = ({
  toolCall,
  result,
  isExecuting,
}) => {
  const [expanded, setExpanded] = useState(true)

  const getStatusIcon = () => {
    if (isExecuting) return '⏳'
    if (!result) return '🔄'
    return result.success ? '✅' : '❌'
  }

  const getStatusText = () => {
    if (isExecuting) return 'Executing...'
    if (!result) return 'Pending'
    return result.success ? 'Done' : 'Failed'
  }

  let parsedArgs: Record<string, any> = {}
  try {
    parsedArgs = JSON.parse(toolCall.function.arguments)
  } catch {
    parsedArgs = { raw: toolCall.function.arguments }
  }

  let parsedResult: any = null
  if (result?.content) {
    try {
      parsedResult = JSON.parse(result.content)
    } catch {
      parsedResult = result.content
    }
  }

  const getToolIcon = () => {
    const name = toolCall.function.name.toLowerCase()
    if (name.includes('calculator') || name.includes('math')) return '📐'
    if (name.includes('search') || name.includes('web')) return '🔍'
    if (name.includes('weather')) return '🌤️'
    if (name.includes('file')) return '📁'
    return '🔧'
  }

  return (
    <div className={`${styles.container} ${result?.success === false ? styles.error : ''}`}>
      <div className={styles.header} onClick={() => setExpanded(!expanded)}>
        <span className={styles.icon}>{getToolIcon()}</span>
        <span className={styles.name}>Tool: {toolCall.function.name}</span>
        <span className={`${styles.status} ${isExecuting ? styles.executing : ''}`}>
          {getStatusIcon()} {getStatusText()}
        </span>
        <span className={styles.expandIcon}>{expanded ? '▼' : '▶'}</span>
      </div>

      {expanded && (
        <div className={styles.body}>
          <div className={styles.section}>
            <label>Input:</label>
            <pre>{JSON.stringify(parsedArgs, null, 2)}</pre>
          </div>

          {result && (
            <div className={styles.section}>
              <label>Output:</label>
              <pre className={result.success ? '' : styles.errorText}>
                {typeof parsedResult === 'string' 
                  ? parsedResult 
                  : JSON.stringify(parsedResult, null, 2)
                }
              </pre>
            </div>
          )}

          {result?.duration_ms !== undefined && (
            <div className={styles.latency}>
              Latency: {result.duration_ms}ms
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ToolCallDisplay
