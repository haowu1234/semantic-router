import React, { useState, useEffect } from 'react'
import { MCPServer, MCPTransportType, MCPToolDefinition } from '../types/tools'
import styles from './MCPPage.module.css'

const MCPPage: React.FC = () => {
  const [servers, setServers] = useState<MCPServer[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddModal, setShowAddModal] = useState(false)
  const [viewingTools, setViewingTools] = useState<{ server: MCPServer; tools: MCPToolDefinition[] } | null>(null)
  
  // Form state
  const [formData, setFormData] = useState({
    name: '',
    transport_type: 'stdio' as MCPTransportType,
    command: '',
    args: '',
    url: '',
  })
  const [formError, setFormError] = useState('')
  const [formLoading, setFormLoading] = useState(false)

  useEffect(() => {
    fetchServers()
  }, [])

  const fetchServers = async () => {
    try {
      const response = await fetch('/api/mcp/servers')
      const data = await response.json()
      setServers(data.servers || [])
    } catch (error) {
      console.error('Failed to fetch MCP servers:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleAdd = async () => {
    setFormError('')
    setFormLoading(true)

    try {
      const payload: Partial<MCPServer> = {
        name: formData.name,
        transport_type: formData.transport_type,
      }

      if (formData.transport_type === 'stdio') {
        payload.command = formData.command
        if (formData.args.trim()) {
          payload.args = formData.args.split(' ').filter(a => a.trim())
        }
      } else {
        payload.url = formData.url
      }

      const response = await fetch('/api/mcp/servers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      const result = await response.json()

      if (!response.ok) {
        setFormError(result.error || 'Failed to add server')
        return
      }

      setServers([...servers, result])
      setShowAddModal(false)
      resetForm()
    } catch (error) {
      setFormError(String(error))
    } finally {
      setFormLoading(false)
    }
  }

  const handleDelete = async (serverId: string) => {
    if (!confirm('Are you sure you want to delete this MCP server?')) return

    try {
      await fetch(`/api/mcp/servers/${serverId}`, { method: 'DELETE' })
      setServers(servers.filter(s => s.id !== serverId))
    } catch (error) {
      console.error('Failed to delete server:', error)
    }
  }

  const handleReconnect = async (serverId: string) => {
    try {
      const response = await fetch(`/api/mcp/servers/${serverId}/reconnect`, {
        method: 'POST',
      })
      const result = await response.json()
      
      if (response.ok) {
        setServers(servers.map(s => s.id === serverId ? result : s))
      }
    } catch (error) {
      console.error('Failed to reconnect:', error)
    }
  }

  const handleViewTools = async (server: MCPServer) => {
    try {
      const response = await fetch(`/api/mcp/servers/${server.id}/tools`)
      const data = await response.json()
      setViewingTools({ server, tools: data.tools || [] })
    } catch (error) {
      console.error('Failed to fetch tools:', error)
    }
  }

  const handleRefresh = async (serverId: string) => {
    try {
      const response = await fetch(`/api/mcp/servers/${serverId}/refresh`, {
        method: 'POST',
      })
      const data = await response.json()
      
      setServers(servers.map(s => 
        s.id === serverId ? { ...s, tool_count: data.total } : s
      ))
    } catch (error) {
      console.error('Failed to refresh:', error)
    }
  }

  const resetForm = () => {
    setFormData({
      name: '',
      transport_type: 'stdio',
      command: '',
      args: '',
      url: '',
    })
    setFormError('')
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return '🟢'
      case 'connecting': return '🟡'
      case 'disconnected': return '⚪'
      case 'error': return '🔴'
      default: return '⚪'
    }
  }

  const connectedCount = servers.filter(s => s.status === 'connected').length
  const totalTools = servers.reduce((sum, s) => sum + s.tool_count, 0)

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading MCP servers...</div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <h1>🔌 MCP Servers</h1>
          <div className={styles.stats}>
            <span className={styles.stat}>{servers.length} servers</span>
            <span className={`${styles.stat} ${styles.connected}`}>{connectedCount} connected</span>
            <span className={`${styles.stat} ${styles.tools}`}>{totalTools} tools</span>
          </div>
        </div>
        <button className={styles.addButton} onClick={() => setShowAddModal(true)}>
          + Add Server
        </button>
      </div>

      {servers.length === 0 ? (
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>🔌</div>
          <h3>No MCP servers configured</h3>
          <p>Add an MCP server to discover and use external tools.</p>
          <button className={styles.addButton} onClick={() => setShowAddModal(true)}>
            Add your first MCP server
          </button>
        </div>
      ) : (
        <div className={styles.serversList}>
          {servers.map(server => (
            <div key={server.id} className={styles.serverCard}>
              <div className={styles.serverHeader}>
                <div className={styles.serverTitle}>
                  <span className={styles.statusIcon}>{getStatusIcon(server.status)}</span>
                  <h3>{server.name}</h3>
                </div>
                <span className={`${styles.transportBadge} ${styles[server.transport_type]}`}>
                  {server.transport_type}
                </span>
              </div>

              <div className={styles.serverDetails}>
                {server.transport_type === 'stdio' ? (
                  <div className={styles.detail}>
                    <span className={styles.detailLabel}>Command:</span>
                    <code>{server.command} {server.args?.join(' ')}</code>
                  </div>
                ) : (
                  <div className={styles.detail}>
                    <span className={styles.detailLabel}>URL:</span>
                    <code>{server.url}</code>
                  </div>
                )}

                <div className={styles.detail}>
                  <span className={styles.detailLabel}>Status:</span>
                  <span className={`${styles.status} ${styles[server.status]}`}>
                    {server.status}
                    {server.status === 'connected' && ` (${server.tool_count} tools)`}
                  </span>
                </div>

                {server.error && (
                  <div className={styles.errorDetail}>
                    <span className={styles.detailLabel}>Error:</span>
                    <span>{server.error}</span>
                  </div>
                )}
              </div>

              <div className={styles.serverActions}>
                {server.status === 'connected' && (
                  <>
                    <button 
                      className={styles.actionButton}
                      onClick={() => handleViewTools(server)}
                    >
                      View Tools
                    </button>
                    <button 
                      className={styles.actionButton}
                      onClick={() => handleRefresh(server.id)}
                    >
                      Refresh
                    </button>
                  </>
                )}
                {(server.status === 'disconnected' || server.status === 'error') && (
                  <button 
                    className={styles.actionButton}
                    onClick={() => handleReconnect(server.id)}
                  >
                    Reconnect
                  </button>
                )}
                <button 
                  className={`${styles.actionButton} ${styles.danger}`}
                  onClick={() => handleDelete(server.id)}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Add Server Modal */}
      {showAddModal && (
        <div className={styles.modalOverlay} onClick={() => { setShowAddModal(false); resetForm(); }}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>Add MCP Server</h2>
              <button className={styles.closeButton} onClick={() => { setShowAddModal(false); resetForm(); }}>×</button>
            </div>

            <div className={styles.modalBody}>
              {formError && (
                <div className={styles.formError}>{formError}</div>
              )}

              <div className={styles.formField}>
                <label>Server Name *</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="my-mcp-server"
                />
              </div>

              <div className={styles.formField}>
                <label>Transport Type *</label>
                <div className={styles.radioGroup}>
                  <label className={styles.radioLabel}>
                    <input
                      type="radio"
                      name="transport"
                      value="stdio"
                      checked={formData.transport_type === 'stdio'}
                      onChange={() => setFormData({ ...formData, transport_type: 'stdio' })}
                    />
                    stdio
                  </label>
                  <label className={styles.radioLabel}>
                    <input
                      type="radio"
                      name="transport"
                      value="sse"
                      checked={formData.transport_type === 'sse'}
                      onChange={() => setFormData({ ...formData, transport_type: 'sse' })}
                    />
                    SSE
                  </label>
                  <label className={styles.radioLabel}>
                    <input
                      type="radio"
                      name="transport"
                      value="http"
                      checked={formData.transport_type === 'http'}
                      onChange={() => setFormData({ ...formData, transport_type: 'http' })}
                    />
                    HTTP
                  </label>
                </div>
              </div>

              {formData.transport_type === 'stdio' ? (
                <>
                  <div className={styles.formField}>
                    <label>Command *</label>
                    <input
                      type="text"
                      value={formData.command}
                      onChange={(e) => setFormData({ ...formData, command: e.target.value })}
                      placeholder="npx"
                    />
                  </div>
                  <div className={styles.formField}>
                    <label>Arguments</label>
                    <input
                      type="text"
                      value={formData.args}
                      onChange={(e) => setFormData({ ...formData, args: e.target.value })}
                      placeholder="-y @anthropic/mcp-server-filesystem /tmp"
                    />
                    <small>Space-separated arguments</small>
                  </div>
                </>
              ) : (
                <div className={styles.formField}>
                  <label>URL *</label>
                  <input
                    type="text"
                    value={formData.url}
                    onChange={(e) => setFormData({ ...formData, url: e.target.value })}
                    placeholder="http://localhost:3001/sse"
                  />
                </div>
              )}

              <div className={styles.modalActions}>
                <button 
                  className={styles.cancelButton}
                  onClick={() => { setShowAddModal(false); resetForm(); }}
                >
                  Cancel
                </button>
                <button 
                  className={styles.saveButton}
                  onClick={handleAdd}
                  disabled={formLoading || !formData.name || (formData.transport_type === 'stdio' ? !formData.command : !formData.url)}
                >
                  {formLoading ? 'Adding...' : 'Add Server'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* View Tools Modal */}
      {viewingTools && (
        <div className={styles.modalOverlay} onClick={() => setViewingTools(null)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>Tools from {viewingTools.server.name}</h2>
              <button className={styles.closeButton} onClick={() => setViewingTools(null)}>×</button>
            </div>

            <div className={styles.modalBody}>
              {viewingTools.tools.length === 0 ? (
                <div className={styles.noTools}>No tools discovered from this server.</div>
              ) : (
                <div className={styles.toolsList}>
                  {viewingTools.tools.map(tool => (
                    <div key={tool.name} className={styles.toolItem}>
                      <h4>{tool.name}</h4>
                      <p>{tool.description}</p>
                      {tool.inputSchema?.properties && (
                        <div className={styles.toolParams}>
                          <span>Parameters: </span>
                          {Object.keys(tool.inputSchema.properties).join(', ')}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default MCPPage
