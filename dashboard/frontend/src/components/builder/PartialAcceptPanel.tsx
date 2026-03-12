/**
 * PartialAcceptPanel — UI for selecting which intents to accept.
 *
 * Allows users to:
 *   - Select/deselect individual intents
 *   - View grouped intents by category (signals, routes, plugins, etc.)
 *   - See dependency warnings when selection breaks references
 *   - Preview the DSL that will be generated
 *   - Apply only selected intents to the editor
 */

import React, { useCallback, useMemo, useState } from 'react'
import { useNLStore } from '@/stores/nlStore'
import type { Intent } from '@/types/intentIR'
import {
  getIntentDisplayName,
  groupIntentsByCategory,
  getSelectionStats,
} from '@/lib/nlPartialAccept'
import type { DependencyIssue, IntentSelectionState } from '@/lib/nlPartialAccept'
import IntentEditor from './IntentEditor'
import styles from './NLMode.module.css'

// ─────────────────────────────────────────────
// Main Panel Component
// ─────────────────────────────────────────────

interface PartialAcceptPanelProps {
  onApply: (result: { dsl: string }) => void
  onCancel: () => void
}

const PartialAcceptPanel: React.FC<PartialAcceptPanelProps> = ({
  onApply,
  onCancel,
}) => {
  const {
    partialAcceptState,
    toggleIntentSelection,
    selectAllIntents,
    autoSelectDependencies,
    selectCategory,
    invertSelections,
    applyPartialAccept,
    editIntent,
    revertIntentEdit,
  } = useNLStore()

  const [showPreview, setShowPreview] = useState(true)
  const [editingIndex, setEditingIndex] = useState<number | null>(null)

  // Early return if no state
  if (!partialAcceptState) {
    return null
  }

  const { originalIR, selections, dependencyIssues, previewDSL, previewValidation } = partialAcceptState
  const stats = getSelectionStats(selections)
  const groups = groupIntentsByCategory(originalIR, selections)

  // Handle apply
  const handleApply = useCallback(() => {
    const result = applyPartialAccept()
    if (result) {
      onApply(result)
    }
  }, [applyPartialAccept, onApply])

  // Handle auto-select for a dependency issue
  const handleAutoSelect = useCallback((issue: DependencyIssue) => {
    if (issue.missingIndex >= 0) {
      toggleIntentSelection(issue.missingIndex)
    }
  }, [toggleIntentSelection])

  // Handle edit intent
  const handleEditIntent = useCallback((index: number) => {
    setEditingIndex(index)
  }, [])

  // Handle save edited intent
  const handleSaveEdit = useCallback((editedIntent: Intent) => {
    if (editingIndex !== null) {
      editIntent(editingIndex, editedIntent)
      setEditingIndex(null)
    }
  }, [editingIndex, editIntent])

  // Handle cancel edit
  const handleCancelEdit = useCallback(() => {
    setEditingIndex(null)
  }, [])

  // Handle revert intent
  const handleRevertIntent = useCallback((index: number) => {
    revertIntentEdit(index)
  }, [revertIntentEdit])

  // Check if all are selected
  const allSelected = stats.selected === stats.total
  const noneSelected = stats.selected === 0
  const someSelected = !allSelected && !noneSelected

  return (
    <div className={styles.partialAcceptOverlay}>
      {/* Header */}
      <div className={styles.partialAcceptHeader}>
        <div className={styles.partialAcceptTitle}>
          <span>Select Items to Accept</span>
          <span className={styles.selectionCount}>
            ({stats.selected}/{stats.total} selected)
          </span>
        </div>
        <div className={styles.partialAcceptActions}>
          <button className={styles.batchBtn} onClick={invertSelections}>
            Invert
          </button>
          <button className={styles.batchBtn} onClick={autoSelectDependencies}>
            Auto-select Deps
          </button>
        </div>
      </div>

      {/* Content */}
      <div className={styles.partialAcceptContent}>
        {/* Select All Row */}
        <div className={styles.selectAllRow}>
          <label className={styles.selectAllCheckbox}>
            <input
              type="checkbox"
              checked={allSelected}
              ref={(el) => {
                if (el) el.indeterminate = someSelected
              }}
              onChange={(e) => selectAllIntents(e.target.checked)}
            />
            <span>Select All</span>
          </label>
          <div className={styles.batchActions}>
            {Object.keys(groups).map(category => (
              <button
                key={category}
                className={styles.batchBtn}
                onClick={() => {
                  const categoryItems = groups[category]
                  const allCategorySelected = categoryItems.every(item => item.selection.selected)
                  selectCategory(category, !allCategorySelected)
                }}
              >
                {category}
              </button>
            ))}
          </div>
        </div>

        {/* Intent Groups */}
        {Object.entries(groups).map(([category, items]) => (
          <IntentGroup
            key={category}
            category={category}
            items={items}
            dependencyIssues={dependencyIssues}
            onToggle={toggleIntentSelection}
            onSelectAll={(selected) => selectCategory(category, selected)}
            onEdit={handleEditIntent}
            onRevert={handleRevertIntent}
          />
        ))}

        {/* Dependency Issues */}
        {dependencyIssues.length > 0 && (
          <div className={styles.dependencyIssues}>
            <div className={styles.issuesHeader}>
              ⚠ Dependency Issues ({dependencyIssues.length})
            </div>
            {dependencyIssues.map((issue, i) => (
              <div key={i} className={styles.issueItem}>
                <span className={styles.issueText}>
                  {getIntentDisplayName(originalIR.intents[issue.sourceIndex])} requires{' '}
                  <strong>{issue.missingRef}</strong>
                </span>
                {issue.missingIndex >= 0 && (
                  <button
                    className={styles.issueAutoSelect}
                    onClick={() => handleAutoSelect(issue)}
                  >
                    Auto-select
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Preview Section */}
        <div className={styles.previewSection}>
          <div className={styles.previewHeader}>
            <span className={styles.previewTitle}>
              <button
                className={styles.toggleBtn}
                onClick={() => setShowPreview(!showPreview)}
              >
                {showPreview ? '▾' : '▸'} Preview DSL
              </button>
            </span>
            {previewValidation && (
              <span className={`${styles.previewValidation} ${previewValidation.isValid ? styles.previewValid : styles.previewInvalid}`}>
                {previewValidation.isValid ? '✓ Valid' : `⚠ ${previewValidation.diagnostics.length} issues`}
              </span>
            )}
          </div>
          {showPreview && previewDSL && (
            <pre className={styles.previewCode}>{previewDSL}</pre>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className={styles.partialAcceptFooter}>
        <div className={styles.footerInfo}>
          {stats.edited > 0 && <span>{stats.edited} item(s) edited</span>}
        </div>
        <div className={styles.footerActions}>
          <button className={styles.cancelBtn} onClick={onCancel}>
            Cancel
          </button>
          <button
            className={styles.applyPartialBtn}
            onClick={handleApply}
            disabled={noneSelected || dependencyIssues.length > 0}
          >
            Apply {stats.selected} Item{stats.selected !== 1 ? 's' : ''}
          </button>
        </div>
      </div>

      {/* Intent Editor Modal */}
      {editingIndex !== null && (
        <IntentEditor
          intent={
            partialAcceptState.selections[editingIndex].edited &&
            partialAcceptState.selections[editingIndex].editedIntent
              ? partialAcceptState.selections[editingIndex].editedIntent!
              : originalIR.intents[editingIndex]
          }
          onSave={handleSaveEdit}
          onCancel={handleCancelEdit}
        />
      )}
    </div>
  )
}

// ─────────────────────────────────────────────
// Intent Group Component
// ─────────────────────────────────────────────

interface IntentGroupProps {
  category: string
  items: Array<{ intent: Intent; selection: IntentSelectionState }>
  dependencyIssues: DependencyIssue[]
  onToggle: (index: number) => void
  onSelectAll: (selected: boolean) => void
  onEdit: (index: number) => void
  onRevert: (index: number) => void
}

const IntentGroup: React.FC<IntentGroupProps> = ({
  category,
  items,
  dependencyIssues,
  onToggle,
  onSelectAll,
  onEdit,
  onRevert,
}) => {
  const selectedCount = items.filter(item => item.selection.selected).length
  const allSelected = selectedCount === items.length

  // Category display names
  const categoryLabels: Record<string, string> = {
    signals: '🔔 Signals',
    routes: '🛤️ Routes',
    plugins: '🔌 Plugins',
    backends: '💾 Backends',
    global: '🌐 Global',
    modifications: '✏️ Modifications',
  }

  return (
    <div className={styles.intentGroup}>
      <div className={styles.intentGroupHeader}>
        <span>{categoryLabels[category] || category} ({selectedCount}/{items.length})</span>
        <button
          className={styles.groupSelectBtn}
          onClick={() => onSelectAll(!allSelected)}
        >
          {allSelected ? 'Deselect all' : 'Select all'}
        </button>
      </div>
      {items.map(({ intent, selection }) => {
        const hasIssue = dependencyIssues.some(issue => issue.sourceIndex === selection.index)
        return (
          <IntentItem
            key={selection.index}
            intent={intent}
            selection={selection}
            hasIssue={hasIssue}
            onToggle={() => onToggle(selection.index)}
            onEdit={() => onEdit(selection.index)}
            onRevert={() => onRevert(selection.index)}
          />
        )
      })}
    </div>
  )
}

// ─────────────────────────────────────────────
// Intent Item Component
// ─────────────────────────────────────────────

interface IntentItemProps {
  intent: Intent
  selection: IntentSelectionState
  hasIssue: boolean
  onToggle: () => void
  onEdit: () => void
  onRevert: () => void
}

const IntentItem: React.FC<IntentItemProps> = ({
  intent,
  selection,
  hasIssue,
  onToggle,
  onEdit,
  onRevert,
}) => {
  // Build class names
  const itemClasses = [
    styles.intentItem,
    selection.selected && styles.selected,
    selection.edited && styles.edited,
    hasIssue && styles.hasIssue,
  ].filter(Boolean).join(' ')

  // Get intent details for display
  const details = useMemo(() => getIntentDetails(intent), [intent])

  // Can edit? (not modify intents for now)
  const canEdit = intent.type !== 'modify'

  return (
    <div className={itemClasses}>
      <input
        type="checkbox"
        className={styles.intentCheckbox}
        checked={selection.selected}
        onChange={onToggle}
      />
      <div className={styles.intentInfo}>
        <div className={styles.intentName}>{details.name}</div>
        <div className={styles.intentMeta}>
          <span className={styles.intentTag}>{details.type}</span>
          {details.subType && (
            <span className={styles.intentTag}>{details.subType}</span>
          )}
          {selection.edited && (
            <span className={styles.intentEditedTag}>✏️ Edited</span>
          )}
          {hasIssue && (
            <span className={styles.intentEditedTag} style={{ background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444' }}>
              ⚠ Missing deps
            </span>
          )}
        </div>
        {details.description && (
          <div className={styles.intentMeta} style={{ marginTop: 4 }}>
            <span style={{ color: 'var(--color-text-muted)', fontSize: '0.65rem' }}>
              {details.description}
            </span>
          </div>
        )}
      </div>
      {/* Action buttons */}
      <div className={styles.intentActions}>
        {canEdit && (
          <button
            className={styles.intentActionBtn}
            onClick={(e) => {
              e.stopPropagation()
              onEdit()
            }}
            title="Edit this item"
          >
            ✏️ Edit
          </button>
        )}
        {selection.edited && (
          <button
            className={styles.intentActionBtn}
            onClick={(e) => {
              e.stopPropagation()
              onRevert()
            }}
            title="Revert to original"
          >
            ↩️ Revert
          </button>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────

interface IntentDetails {
  name: string
  type: string
  subType?: string
  description?: string
}

function getIntentDetails(intent: Intent): IntentDetails {
  switch (intent.type) {
    case 'signal':
      return {
        name: intent.name,
        type: 'signal',
        subType: intent.signal_type,
        description: intent.fields.description as string | undefined,
      }
    case 'route':
      return {
        name: intent.name,
        type: 'route',
        description: intent.description,
      }
    case 'plugin_template':
      return {
        name: intent.name,
        type: 'plugin',
        subType: intent.plugin_type,
      }
    case 'backend':
      return {
        name: intent.name,
        type: 'backend',
        subType: intent.backend_type,
      }
    case 'global':
      return {
        name: 'GLOBAL',
        type: 'global',
      }
    case 'modify':
      return {
        name: `${intent.action} ${intent.target_name}`,
        type: 'modify',
        subType: intent.target_construct,
      }
    default:
      return {
        name: 'unknown',
        type: 'unknown',
      }
  }
}

export default PartialAcceptPanel
