import React, { useState, useEffect } from 'react'
import { Tool, PlaygroundToolSettings } from '../types/tools'
import styles from './PlaygroundToolPanel.module.css'

interface PlaygroundToolPanelProps {
  settings: PlaygroundToolSettings
  onSettingsChange: (settings: PlaygroundToolSettings) => void
  collapsed?: boolean
  onToggleCollapse?: () => void
}

const PlaygroundToolPanel: React.FC<PlaygroundToolPanelProps> = ({
  settings,
  onSettingsChange,
  collapsed = false,
  onToggleCollapse,
}) => {
  const [tools, setTools] = useState<Tool[]>([])
  const [loading, setLoading] = useState(true)

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

  const enabledTools = tools.filter(t => t.enabled)

  const handleToolSelect = (toolId: string, selected: boolean) => {
    const newSelected = selected
      ? [...settings.selectedTools, toolId]
      : settings.selectedTools.filter(id => id !== toolId)
    onSettingsChange({ ...settings, selectedTools: newSelected })
  }

  const handleSelectAll = () => {
    onSettingsChange({
      ...settings,
      selectedTools: enabledTools.map(t => t.id),
    })
  }

  const handleSelectNone = () => {
    onSettingsChange({ ...settings, selectedTools: [] })
  }

  const getToolIcon = (tool: Tool) => {
    const name = tool.name.toLowerCase()
    if (name.includes('calculator') || name.includes('math')) return '📐'
    if (name.includes('search') || name.includes('web')) return '🔍'
    if (name.includes('weather')) return '🌤️'
    if (name.includes('file')) return '📁'
    return '🔧'
  }

  if (collapsed) {
    return (
      <div className={styles.collapsed} onClick={onToggleCollapse}>
        <span className={styles.collapsedIcon}>🔧</span>
        <span className={styles.collapsedLabel}>Tools</span>
        {settings.enabled && settings.selectedTools.length > 0 && (
          <span className={styles.collapsedBadge}>{settings.selectedTools.length}</span>
        )}
      </div>
    )
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h3>🔧 Tool Settings</h3>
        {onToggleCollapse && (
          <button className={styles.collapseBtn} onClick={onToggleCollapse}>
            ▼
          </button>
        )}
      </div>

      <div className={styles.content}>
        <label className={styles.enableToggle}>
          <input
            type="checkbox"
            checked={settings.enabled}
            onChange={(e) => onSettingsChange({ ...settings, enabled: e.target.checked })}
          />
          <span>Enable Tools</span>
        </label>

        {settings.enabled && (
          <>
            <div className={styles.modeSection}>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  name="selectionMode"
                  value="auto"
                  checked={settings.selectionMode === 'auto'}
                  onChange={() => onSettingsChange({ ...settings, selectionMode: 'auto' })}
                />
                <span>Auto (Semantic Match)</span>
              </label>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  name="selectionMode"
                  value="manual"
                  checked={settings.selectionMode === 'manual'}
                  onChange={() => onSettingsChange({ ...settings, selectionMode: 'manual' })}
                />
                <span>Manual Selection</span>
              </label>
            </div>

            {settings.selectionMode === 'auto' && (
              <div className={styles.semanticSettings}>
                <div className={styles.settingRow}>
                  <label>Top-K:</label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={settings.topK}
                    onChange={(e) => onSettingsChange({ ...settings, topK: Number(e.target.value) })}
                  />
                </div>
                <div className={styles.settingRow}>
                  <label>Threshold:</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    value={settings.similarityThreshold}
                    onChange={(e) => onSettingsChange({ ...settings, similarityThreshold: Number(e.target.value) })}
                  />
                </div>
              </div>
            )}

            {settings.selectionMode === 'manual' && (
              <div className={styles.toolSelection}>
                <div className={styles.selectionActions}>
                  <button onClick={handleSelectAll}>All</button>
                  <button onClick={handleSelectNone}>None</button>
                  <span className={styles.count}>
                    {settings.selectedTools.length}/{enabledTools.length}
                  </span>
                </div>

                {loading ? (
                  <div className={styles.loading}>Loading tools...</div>
                ) : (
                  <div className={styles.toolList}>
                    {enabledTools.map(tool => (
                      <label key={tool.id} className={styles.toolCheckbox}>
                        <input
                          type="checkbox"
                          checked={settings.selectedTools.includes(tool.id)}
                          onChange={(e) => handleToolSelect(tool.id, e.target.checked)}
                        />
                        <span className={styles.toolIcon}>{getToolIcon(tool)}</span>
                        <span className={styles.toolName}>{tool.name}</span>
                        <span className={`${styles.sourceBadge} ${styles[tool.source]}`}>
                          {tool.source}
                        </span>
                      </label>
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default PlaygroundToolPanel
