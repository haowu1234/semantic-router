import React, { useState, useEffect } from 'react'
import { Tool, ToolExecutionResult } from '../types/tools'
import styles from './ToolsPage.module.css'

type FilterType = 'all' | 'builtin' | 'mcp';

const ToolsPage: React.FC = () => {
  const [tools, setTools] = useState<Tool[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<FilterType>('all')
  const [search, setSearch] = useState('')
  const [testingTool, setTestingTool] = useState<Tool | null>(null)
  const [testArgs, setTestArgs] = useState<Record<string, any>>({})
  const [testResult, setTestResult] = useState<ToolExecutionResult | null>(null)
  const [testLoading, setTestLoading] = useState(false)

  useEffect(() => {
    fetchTools()
  }, [])

  const fetchTools = async () => {
    try {
      const response = await fetch('/api/tools')
      const data = await response.json()
      setTools(data.tools || [])
    } catch (error) {
      console.error('Failed to fetch tools:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleToggle = async (toolId: string, enabled: boolean) => {
    try {
      await fetch(`/api/tools/${toolId}/toggle`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      })
      setTools(tools.map(t => 
        t.id === toolId ? { ...t, enabled } : t
      ))
    } catch (error) {
      console.error('Failed to toggle tool:', error)
    }
  }

  const handleTest = async () => {
    if (!testingTool) return
    setTestLoading(true)
    setTestResult(null)

    try {
      const response = await fetch('/api/tools/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool_id: testingTool.id,
          arguments: testArgs,
        }),
      })
      const result = await response.json()
      setTestResult(result)
    } catch (error) {
      setTestResult({
        success: false,
        error: String(error),
        duration_ms: 0,
      })
    } finally {
      setTestLoading(false)
    }
  }

  const getToolIcon = (tool: Tool) => {
    const name = tool.name.toLowerCase()
    if (name.includes('calculator') || name.includes('math')) return '📐'
    if (name.includes('search') || name.includes('web')) return '🔍'
    if (name.includes('weather')) return '🌤️'
    if (name.includes('file')) return '📁'
    return '🔧'
  }

  const filteredTools = tools.filter(tool => {
    const matchesFilter = filter === 'all' || tool.source === filter
    const matchesSearch = tool.name.toLowerCase().includes(search.toLowerCase()) ||
                          tool.description.toLowerCase().includes(search.toLowerCase())
    return matchesFilter && matchesSearch
  })

  const builtinCount = tools.filter(t => t.source === 'builtin').length
  const mcpCount = tools.filter(t => t.source === 'mcp').length

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading tools...</div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <h1>🔧 Tools Management</h1>
          <div className={styles.stats}>
            <span className={styles.stat}>{tools.length} total</span>
            <span className={`${styles.stat} ${styles.builtin}`}>{builtinCount} builtin</span>
            <span className={`${styles.stat} ${styles.mcp}`}>{mcpCount} MCP</span>
          </div>
        </div>
      </div>

      <div className={styles.filters}>
        <input
          type="text"
          placeholder="Search tools..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className={styles.searchInput}
        />
        <div className={styles.filterButtons}>
          <button
            className={`${styles.filterBtn} ${filter === 'all' ? styles.active : ''}`}
            onClick={() => setFilter('all')}
          >
            All
          </button>
          <button
            className={`${styles.filterBtn} ${filter === 'builtin' ? styles.active : ''}`}
            onClick={() => setFilter('builtin')}
          >
            Builtin
          </button>
          <button
            className={`${styles.filterBtn} ${filter === 'mcp' ? styles.active : ''}`}
            onClick={() => setFilter('mcp')}
          >
            MCP
          </button>
        </div>
      </div>

      <div className={styles.toolsGrid}>
        {filteredTools.map(tool => (
          <div key={tool.id} className={`${styles.toolCard} ${!tool.enabled ? styles.disabled : ''}`}>
            <div className={styles.toolHeader}>
              <span className={styles.toolIcon}>{getToolIcon(tool)}</span>
              <div className={styles.toolTitle}>
                <h3>{tool.name}</h3>
                <span className={`${styles.sourceBadge} ${styles[tool.source]}`}>
                  {tool.source}
                </span>
              </div>
              <label className={styles.toggleSwitch}>
                <input
                  type="checkbox"
                  checked={tool.enabled}
                  onChange={(e) => handleToggle(tool.id, e.target.checked)}
                />
                <span className={styles.slider}></span>
              </label>
            </div>

            <p className={styles.toolDescription}>{tool.description}</p>

            <div className={styles.toolParameters}>
              <h4>Parameters:</h4>
              <ul>
                {tool.parameters.map(param => (
                  <li key={param.name}>
                    <code>{param.name}</code>
                    <span className={styles.paramType}>({param.type})</span>
                    {param.required && <span className={styles.required}>*</span>}
                  </li>
                ))}
              </ul>
            </div>

            {tool.mcp_server && (
              <div className={styles.toolSource}>
                Source: <code>{tool.mcp_server}</code>
              </div>
            )}

            <div className={styles.toolActions}>
              <button 
                className={styles.testButton}
                onClick={() => {
                  setTestingTool(tool)
                  setTestArgs({})
                  setTestResult(null)
                }}
              >
                Test
              </button>
            </div>
          </div>
        ))}
      </div>

      {filteredTools.length === 0 && (
        <div className={styles.emptyState}>
          <p>No tools found matching your criteria.</p>
        </div>
      )}

      {/* Test Modal */}
      {testingTool && (
        <div className={styles.modalOverlay} onClick={() => setTestingTool(null)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>🔧 Test Tool: {testingTool.name}</h2>
              <button className={styles.closeButton} onClick={() => setTestingTool(null)}>×</button>
            </div>

            <div className={styles.modalBody}>
              <div className={styles.toolInfo}>
                <p>{testingTool.description}</p>
              </div>

              <div className={styles.parametersForm}>
                <h3>Parameters</h3>
                {testingTool.parameters.map(param => (
                  <div key={param.name} className={styles.formField}>
                    <label>
                      {param.name}
                      {param.required && <span className={styles.required}>*</span>}
                      <span className={styles.typeHint}>({param.type})</span>
                    </label>
                    {param.enum ? (
                      <select
                        value={testArgs[param.name] ?? param.default ?? ''}
                        onChange={(e) => setTestArgs({ ...testArgs, [param.name]: e.target.value })}
                      >
                        <option value="">Select...</option>
                        {param.enum.map(opt => (
                          <option key={String(opt)} value={String(opt)}>{String(opt)}</option>
                        ))}
                      </select>
                    ) : param.type === 'boolean' ? (
                      <select
                        value={String(testArgs[param.name] ?? param.default ?? false)}
                        onChange={(e) => setTestArgs({ ...testArgs, [param.name]: e.target.value === 'true' })}
                      >
                        <option value="true">true</option>
                        <option value="false">false</option>
                      </select>
                    ) : (
                      <input
                        type={param.type === 'integer' || param.type === 'number' ? 'number' : 'text'}
                        value={testArgs[param.name] ?? param.default ?? ''}
                        onChange={(e) => {
                          let value: any = e.target.value
                          if (param.type === 'integer') value = parseInt(value) || 0
                          if (param.type === 'number') value = parseFloat(value) || 0
                          setTestArgs({ ...testArgs, [param.name]: value })
                        }}
                        placeholder={param.description}
                      />
                    )}
                    <small>{param.description}</small>
                  </div>
                ))}
              </div>

              <div className={styles.modalActions}>
                <button 
                  className={styles.clearButton} 
                  onClick={() => setTestArgs({})}
                >
                  Clear
                </button>
                <button 
                  className={styles.executeButton} 
                  onClick={handleTest}
                  disabled={testLoading}
                >
                  {testLoading ? 'Executing...' : 'Execute'}
                </button>
              </div>

              {testResult && (
                <div className={`${styles.resultPanel} ${testResult.success ? styles.success : styles.error}`}>
                  <div className={styles.resultHeader}>
                    <span>{testResult.success ? '✅ Success' : '❌ Error'}</span>
                    <span className={styles.duration}>{testResult.duration_ms}ms</span>
                  </div>
                  <pre className={styles.resultContent}>
                    {testResult.success 
                      ? JSON.stringify(testResult.result, null, 2)
                      : testResult.error
                    }
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ToolsPage
