/**
 * MCP Configuration Panel Component
 * MCP 服务器与工具配置管理面板
 */

import React, { useState, useCallback, useEffect } from 'react'
import { useMCPServers } from '../tools/mcp'
import type { MCPServerConfig, MCPServerState, MCPTransportType, MCPToolDefinition } from '../tools/mcp'
import styles from './MCPConfigPanel.module.css'

// 内置工具类型定义
interface BuiltInToolParameter {
  type: string
  description?: string
  enum?: string[]
  default?: unknown
}

interface BuiltInTool {
  tool: {
    type: string
    function: {
      name: string
      description: string
      parameters: {
        type: string
        properties: Record<string, BuiltInToolParameter>
        required?: string[]
      }
    }
  }
  description: string
  category: string
  tags: string[]
}

interface MCPConfigPanelProps {
  onClose?: () => void
}

export const MCPConfigPanel: React.FC<MCPConfigPanelProps> = ({ onClose }) => {
  const {
    servers,
    tools,
    loading,
    error,
    addServer,
    updateServer,
    deleteServer,
    connect,
    disconnect,
    testConnection,
    refreshServers,
  } = useMCPServers()

  const [showAddDialog, setShowAddDialog] = useState(false)
  const [editingServer, setEditingServer] = useState<MCPServerConfig | null>(null)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  // 折叠状态：默认已连接的展开，未连接的折叠
  const [expandedServers, setExpandedServers] = useState<Set<string>>(new Set())
  // 内置工具展开状态
  const [builtInExpanded, setBuiltInExpanded] = useState(false)
  // 内置工具数据
  const [builtInTools, setBuiltInTools] = useState<BuiltInTool[]>([])
  const [builtInLoading, setBuiltInLoading] = useState(false)

  // 初始化折叠状态
  useEffect(() => {
    const connectedIds = servers
      .filter(s => s.status === 'connected')
      .map(s => s.config.id)
    setExpandedServers(new Set(connectedIds))
  }, [servers.length]) // 只在服务器数量变化时更新

  // 获取内置工具
  useEffect(() => {
    const fetchBuiltInTools = async () => {
      setBuiltInLoading(true)
      try {
        const response = await fetch('/api/tools-db')
        if (response.ok) {
          const data = await response.json()
          setBuiltInTools(data || [])
        }
      } catch (err) {
        console.error('Failed to load built-in tools:', err)
      } finally {
        setBuiltInLoading(false)
      }
    }
    fetchBuiltInTools()
  }, [])

  // 切换服务器展开/折叠
  const toggleServerExpand = useCallback((serverId: string) => {
    setExpandedServers(prev => {
      const next = new Set(prev)
      if (next.has(serverId)) {
        next.delete(serverId)
      } else {
        next.add(serverId)
      }
      return next
    })
  }, [])

  // 处理连接/断开
  const handleToggleConnection = useCallback(async (server: MCPServerState) => {
    setActionLoading(server.config.id)
    try {
      if (server.status === 'connected') {
        await disconnect(server.config.id)
      } else {
        await connect(server.config.id)
        // 连接后自动展开
        setExpandedServers(prev => new Set(prev).add(server.config.id))
      }
    } catch (err) {
      console.error('Connection toggle failed:', err)
    } finally {
      setActionLoading(null)
    }
  }, [connect, disconnect])

  // 处理删除
  const handleDelete = useCallback(async (id: string) => {
    if (!window.confirm('Are you sure you want to delete this MCP server?')) {
      return
    }
    setActionLoading(id)
    try {
      await deleteServer(id)
    } catch (err) {
      console.error('Delete failed:', err)
    } finally {
      setActionLoading(null)
    }
  }, [deleteServer])

  // 获取状态图标
  const getStatusIcon = (status: MCPServerState['status']) => {
    switch (status) {
      case 'connected':
        return <span className={styles.statusDot} data-status="connected">●</span>
      case 'connecting':
        return <span className={styles.statusDot} data-status="connecting">◐</span>
      case 'error':
        return <span className={styles.statusDot} data-status="error">●</span>
      default:
        return <span className={styles.statusDot} data-status="disconnected">○</span>
    }
  }

  // 获取传输类型标签
  const getTransportLabel = (transport: MCPTransportType) => {
    switch (transport) {
      case 'stdio':
        return 'Stdio'
      case 'streamable-http':
        return 'HTTP'
      default:
        return transport
    }
  }

  // 解析工具参数
  const renderToolParameters = (tool: MCPToolDefinition) => {
    const schema = tool.inputSchema
    if (!schema || schema.type !== 'object') {
      return <span className={styles.noParams}>No parameters</span>
    }
    
    const properties = schema.properties || {}
    const required = schema.required || []
    const params = Object.entries(properties)
    
    if (params.length === 0) {
      return <span className={styles.noParams}>No parameters</span>
    }

    return (
      <div className={styles.paramsList}>
        {params.map(([name, prop]) => {
          const propData = prop as { type?: string; description?: string }
          return (
            <div key={name} className={styles.paramItem}>
              <span className={styles.paramName}>{name}</span>
              <span className={styles.paramType}>({propData.type || 'any'})</span>
              {required.includes(name) && <span className={styles.paramRequired}>*</span>}
              {propData.description && (
                <span className={styles.paramDesc}>{propData.description}</span>
              )}
            </div>
          )
        })}
      </div>
    )
  }

  // 渲染内置工具参数
  const renderBuiltInToolParameters = (tool: BuiltInTool) => {
    const params = tool.tool.function.parameters
    if (!params || !params.properties) {
      return <span className={styles.noParams}>No parameters</span>
    }
    
    const properties = params.properties
    const required = params.required || []
    const entries = Object.entries(properties)
    
    if (entries.length === 0) {
      return <span className={styles.noParams}>No parameters</span>
    }

    return (
      <div className={styles.paramsList}>
        {entries.map(([name, prop]) => (
          <div key={name} className={styles.paramItem}>
            <span className={styles.paramName}>{name}</span>
            <span className={styles.paramType}>({prop.type || 'any'})</span>
            {required.includes(name) && <span className={styles.paramRequired}>*</span>}
            {prop.description && (
              <span className={styles.paramDesc}>{prop.description}</span>
            )}
          </div>
        ))}
      </div>
    )
  }

  // 计算统计信息
  const connectedCount = servers.filter(s => s.status === 'connected').length
  const mcpToolsCount = tools.length
  const builtInCount = builtInTools.length
  const totalToolsCount = mcpToolsCount + builtInCount

  if (loading) {
    return (
      <div className={styles.panel}>
        <div className={styles.header}>
          <h2>🔌 MCP Servers & Tools</h2>
          {onClose && <button className={styles.closeBtn} onClick={onClose}>×</button>}
        </div>
        <div className={styles.loading}>Loading...</div>
      </div>
    )
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h2>🔌 MCP Servers & Tools</h2>
        <div className={styles.headerActions}>
          <button 
            className={styles.refreshBtn} 
            onClick={() => refreshServers()}
            title="Refresh"
          >
            ↻
          </button>
          {onClose && <button className={styles.closeBtn} onClick={onClose}>×</button>}
        </div>
      </div>

      {error && (
        <div className={styles.error}>
          {error}
        </div>
      )}

      <div className={styles.serverList}>
        {/* MCP Servers Section */}
        {servers.length === 0 && builtInTools.length === 0 ? (
          <div className={styles.empty}>
            No MCP servers configured and no built-in tools available.
            <br />
            Click "Add MCP Server" to get started.
          </div>
        ) : (
          <>
            {/* MCP Servers */}
            {servers.map((server) => {
              const isExpanded = expandedServers.has(server.config.id)
              const hasTools = server.status === 'connected' && server.tools && server.tools.length > 0

              return (
                <div key={server.config.id} className={styles.serverCard}>
                  {/* Server Header - Clickable to expand/collapse */}
                  <div 
                    className={styles.serverHeader}
                    onClick={() => hasTools && toggleServerExpand(server.config.id)}
                    style={{ cursor: hasTools ? 'pointer' : 'default' }}
                  >
                    <div className={styles.serverInfo}>
                      {hasTools && (
                        <span className={styles.expandIcon}>
                          {isExpanded ? '▼' : '▶'}
                        </span>
                      )}
                      {getStatusIcon(server.status)}
                      <span className={styles.serverName}>{server.config.name}</span>
                      <span className={styles.transportBadge}>
                        {getTransportLabel(server.config.transport)}
                      </span>
                      <span className={`${styles.statusBadge} ${styles[server.status]}`}>
                        {server.status}
                      </span>
                      {hasTools && (
                        <span className={styles.toolCount}>
                          {server.tools!.length} tools
                        </span>
                      )}
                    </div>
                    <div className={styles.serverActions} onClick={e => e.stopPropagation()}>
                      <button
                        className={styles.actionBtn}
                        onClick={() => handleToggleConnection(server)}
                        disabled={actionLoading === server.config.id}
                        title={server.status === 'connected' ? 'Disconnect' : 'Connect'}
                      >
                        {actionLoading === server.config.id ? '...' : 
                          server.status === 'connected' ? '⏹' : '▶'}
                      </button>
                      <button
                        className={styles.actionBtn}
                        onClick={() => setEditingServer(server.config)}
                        title="Edit"
                      >
                        ⚙
                      </button>
                      <button
                        className={styles.actionBtn}
                        onClick={() => handleDelete(server.config.id)}
                        disabled={actionLoading === server.config.id}
                        title="Delete"
                      >
                        🗑
                      </button>
                    </div>
                  </div>

                  {/* Server Description */}
                  {server.config.description && (
                    <div className={styles.serverDescription}>
                      {server.config.description}
                    </div>
                  )}

                  {/* Server Error */}
                  {server.error && (
                    <div className={styles.serverError}>
                      {server.error}
                    </div>
                  )}

                  {/* Disconnected hint */}
                  {server.status !== 'connected' && !server.error && (
                    <div className={styles.connectionHint}>
                      Click ▶ to connect and load tools
                    </div>
                  )}

                  {/* Tools List - Collapsible */}
                  {hasTools && isExpanded && (
                    <div className={styles.toolsContainer}>
                      {server.tools!.map((tool) => (
                        <div key={tool.name} className={styles.toolCard}>
                          <div className={styles.toolHeader}>
                            <span className={styles.toolIcon}>🔧</span>
                            <span className={styles.toolName}>{tool.name}</span>
                          </div>
                          {tool.description && (
                            <div className={styles.toolDescription}>
                              {tool.description}
                            </div>
                          )}
                          <div className={styles.toolParams}>
                            <span className={styles.paramsLabel}>Parameters:</span>
                            {renderToolParameters(tool)}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}

            {/* Built-in Tools Section */}
            {builtInTools.length > 0 && (
              <div className={`${styles.serverCard} ${styles.builtInSection}`}>
                <div 
                  className={styles.serverHeader}
                  onClick={() => setBuiltInExpanded(!builtInExpanded)}
                  style={{ cursor: 'pointer' }}
                >
                  <div className={styles.serverInfo}>
                    <span className={styles.expandIcon}>
                      {builtInExpanded ? '▼' : '▶'}
                    </span>
                    <span className={styles.statusDot} data-status="connected">●</span>
                    <span className={styles.serverName}>Built-in Tools</span>
                    <span className={styles.transportBadge}>Internal</span>
                    <span className={`${styles.statusBadge} ${styles.connected}`}>
                      active
                    </span>
                    <span className={styles.toolCount}>
                      {builtInTools.length} tools
                    </span>
                  </div>
                </div>
                <div className={styles.serverDescription}>
                  Pre-configured tools for common tasks (weather, search, calculation, etc.)
                </div>

                {builtInExpanded && (
                  <div className={styles.toolsContainer}>
                    {builtInLoading ? (
                      <div className={styles.toolsLoading}>Loading built-in tools...</div>
                    ) : (
                      builtInTools.map((tool) => (
                        <div key={tool.tool.function.name} className={styles.toolCard}>
                          <div className={styles.toolHeader}>
                            <span className={styles.toolIcon}>🔧</span>
                            <span className={styles.toolName}>{tool.tool.function.name}</span>
                            <span className={styles.categoryBadge}>{tool.category}</span>
                          </div>
                          <div className={styles.toolDescription}>
                            {tool.tool.function.description}
                          </div>
                          {tool.tags && tool.tags.length > 0 && (
                            <div className={styles.toolTags}>
                              {tool.tags.map(tag => (
                                <span key={tag} className={styles.tagBadge}>{tag}</span>
                              ))}
                            </div>
                          )}
                          <div className={styles.toolParams}>
                            <span className={styles.paramsLabel}>Parameters:</span>
                            {renderBuiltInToolParameters(tool)}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      <div className={styles.footer}>
        <button 
          className={styles.addBtn}
          onClick={() => setShowAddDialog(true)}
        >
          + Add MCP Server
        </button>
        <div className={styles.summary}>
          {totalToolsCount} tools ({mcpToolsCount} from MCP, {builtInCount} built-in) • {connectedCount} connected servers
        </div>
      </div>

      {/* Add/Edit Dialog */}
      {(showAddDialog || editingServer) && (
        <MCPServerDialog
          server={editingServer}
          onSave={async (config) => {
            if (editingServer) {
              await updateServer(editingServer.id, config)
            } else {
              await addServer(config)
            }
            setShowAddDialog(false)
            setEditingServer(null)
          }}
          onTest={testConnection}
          onClose={() => {
            setShowAddDialog(false)
            setEditingServer(null)
          }}
        />
      )}
    </div>
  )
}

// ========== Server Dialog Component ==========

interface MCPServerDialogProps {
  server: MCPServerConfig | null
  onSave: (config: Omit<MCPServerConfig, 'id'>) => Promise<void>
  onTest: (config: MCPServerConfig) => Promise<{ success: boolean; error?: string }>
  onClose: () => void
}

const MCPServerDialog: React.FC<MCPServerDialogProps> = ({
  server,
  onSave,
  onTest,
  onClose,
}) => {
  const [name, setName] = useState(server?.name || '')
  const [description, setDescription] = useState(server?.description || '')
  const [transport, setTransport] = useState<MCPTransportType>(server?.transport || 'stdio')
  const [enabled, setEnabled] = useState(server?.enabled ?? true)
  
  // Stdio config
  const [command, setCommand] = useState(server?.connection?.command || '')
  const [args, setArgs] = useState(server?.connection?.args?.join('\n') || '')
  
  // HTTP config
  const [url, setUrl] = useState(server?.connection?.url || '')
  
  // Options
  const [timeout, setTimeout] = useState(server?.options?.timeout?.toString() || '30000')
  const [autoReconnect, setAutoReconnect] = useState(server?.options?.autoReconnect ?? true)
  
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ success: boolean; error?: string } | null>(null)

  const handleSave = async () => {
    setSaving(true)
    try {
      const config: Omit<MCPServerConfig, 'id'> = {
        name,
        description: description || undefined,
        transport,
        enabled,
        connection: transport === 'stdio'
          ? {
              command,
              args: args.split('\n').filter(a => a.trim()),
            }
          : {
              url,
            },
        options: {
          timeout: parseInt(timeout) || 30000,
          autoReconnect,
        },
      }
      await onSave(config)
    } catch (err) {
      console.error('Save failed:', err)
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const config: MCPServerConfig = {
        id: server?.id || 'test',
        name,
        description,
        transport,
        enabled,
        connection: transport === 'stdio'
          ? { command, args: args.split('\n').filter(a => a.trim()) }
          : { url },
        options: { timeout: parseInt(timeout) || 30000 },
      }
      const result = await onTest(config)
      setTestResult(result)
    } catch (err) {
      setTestResult({ success: false, error: err instanceof Error ? err.message : 'Test failed' })
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className={styles.dialogOverlay} onClick={onClose}>
      <div className={styles.dialog} onClick={e => e.stopPropagation()}>
        <div className={styles.dialogHeader}>
          <h3>{server ? 'Edit MCP Server' : 'Add MCP Server'}</h3>
          <button className={styles.closeBtn} onClick={onClose}>×</button>
        </div>

        <div className={styles.dialogContent}>
          <div className={styles.formGroup}>
            <label>Name *</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="My MCP Server"
            />
          </div>

          <div className={styles.formGroup}>
            <label>Description</label>
            <input
              type="text"
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Optional description"
            />
          </div>

          <div className={styles.formGroup}>
            <label>Transport Protocol *</label>
            <div className={styles.radioGroup}>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  name="transport"
                  value="stdio"
                  checked={transport === 'stdio'}
                  onChange={() => setTransport('stdio')}
                />
                <span>Stdio</span>
                <small>Local command line (filesystem, git, etc.)</small>
              </label>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  name="transport"
                  value="streamable-http"
                  checked={transport === 'streamable-http'}
                  onChange={() => setTransport('streamable-http')}
                />
                <span>Streamable HTTP</span>
                <small>Remote service with streaming support</small>
              </label>
            </div>
          </div>

          {transport === 'stdio' ? (
            <>
              <div className={styles.formGroup}>
                <label>Command *</label>
                <input
                  type="text"
                  value={command}
                  onChange={e => setCommand(e.target.value)}
                  placeholder="npx"
                />
              </div>
              <div className={styles.formGroup}>
                <label>Arguments (one per line)</label>
                <textarea
                  value={args}
                  onChange={e => setArgs(e.target.value)}
                  placeholder={"-y\n@modelcontextprotocol/server-filesystem\n/Users/workspace"}
                  rows={4}
                />
              </div>
            </>
          ) : (
            <div className={styles.formGroup}>
              <label>URL *</label>
              <input
                type="text"
                value={url}
                onChange={e => setUrl(e.target.value)}
                placeholder="https://api.example.com/mcp"
              />
            </div>
          )}

          <div className={styles.formGroup}>
            <label>Timeout (ms)</label>
            <input
              type="number"
              value={timeout}
              onChange={e => setTimeout(e.target.value)}
              placeholder="30000"
            />
          </div>

          <div className={styles.formGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={autoReconnect}
                onChange={e => setAutoReconnect(e.target.checked)}
              />
              <span>Auto Reconnect</span>
            </label>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={enabled}
                onChange={e => setEnabled(e.target.checked)}
              />
              <span>Enabled</span>
            </label>
          </div>

          {testResult && (
            <div className={testResult.success ? styles.testSuccess : styles.testError}>
              {testResult.success ? '✓ Connection successful!' : `✗ ${testResult.error}`}
            </div>
          )}
        </div>

        <div className={styles.dialogFooter}>
          <button className={styles.cancelBtn} onClick={onClose}>
            Cancel
          </button>
          <button 
            className={styles.testBtn} 
            onClick={handleTest}
            disabled={testing || !name || (transport === 'stdio' ? !command : !url)}
          >
            {testing ? 'Testing...' : 'Test Connection'}
          </button>
          <button 
            className={styles.saveBtn} 
            onClick={handleSave}
            disabled={saving || !name || (transport === 'stdio' ? !command : !url)}
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default MCPConfigPanel
