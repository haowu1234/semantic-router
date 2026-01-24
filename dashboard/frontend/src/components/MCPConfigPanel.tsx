/**
 * MCP Configuration Panel Component
 * MCP 服务器配置管理面板
 */

import React, { useState, useCallback } from 'react'
import { useMCPServers } from '../tools/mcp'
import type { MCPServerConfig, MCPServerState, MCPTransportType } from '../tools/mcp'
import styles from './MCPConfigPanel.module.css'

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

  // 处理连接/断开
  const handleToggleConnection = useCallback(async (server: MCPServerState) => {
    setActionLoading(server.config.id)
    try {
      if (server.status === 'connected') {
        await disconnect(server.config.id)
      } else {
        await connect(server.config.id)
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

  if (loading) {
    return (
      <div className={styles.panel}>
        <div className={styles.header}>
          <h2>🔌 MCP Servers</h2>
          {onClose && <button className={styles.closeBtn} onClick={onClose}>×</button>}
        </div>
        <div className={styles.loading}>Loading...</div>
      </div>
    )
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h2>🔌 MCP Servers</h2>
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
        {servers.length === 0 ? (
          <div className={styles.empty}>
            No MCP servers configured. Click "Add Server" to get started.
          </div>
        ) : (
          servers.map((server) => (
            <div key={server.config.id} className={styles.serverCard}>
              <div className={styles.serverHeader}>
                <div className={styles.serverInfo}>
                  {getStatusIcon(server.status)}
                  <span className={styles.serverName}>{server.config.name}</span>
                  <span className={styles.transportBadge}>
                    {getTransportLabel(server.config.transport)}
                  </span>
                </div>
                <div className={styles.serverActions}>
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

              {server.config.description && (
                <div className={styles.serverDescription}>
                  {server.config.description}
                </div>
              )}

              {server.status === 'connected' && server.tools && server.tools.length > 0 && (
                <div className={styles.toolsList}>
                  <span className={styles.toolsLabel}>Tools:</span>
                  {server.tools.slice(0, 5).map((tool) => (
                    <span key={tool.name} className={styles.toolBadge}>
                      {tool.name}
                    </span>
                  ))}
                  {server.tools.length > 5 && (
                    <span className={styles.moreTools}>
                      +{server.tools.length - 5} more
                    </span>
                  )}
                </div>
              )}

              {server.error && (
                <div className={styles.serverError}>
                  {server.error}
                </div>
              )}
            </div>
          ))
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
          {tools.length} tools available from {servers.filter(s => s.status === 'connected').length} connected servers
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
