/**
 * IntentEditor — Modal dialog for editing a single Intent.
 *
 * Uses the existing FieldSchema system from dslMutations.ts to render
 * appropriate form fields for each intent type (signal, route, plugin, etc.).
 *
 * Features:
 *   - Dynamic form generation based on intent type
 *   - Real-time field validation
 *   - Support for all field types: string, number, boolean, select, string[], json
 */

import React, { useCallback, useMemo, useState } from 'react'
import type { Intent, SignalIntent, PluginTemplateIntent, BackendIntent, RouteIntent } from '@/types/intentIR'
import {
  getSignalFieldSchema,
  getPluginFieldSchema,
  getAlgorithmFieldSchema,
  type FieldSchema,
} from '@/lib/dslMutations'
import styles from './IntentEditor.module.css'

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface IntentEditorProps {
  intent: Intent
  onSave: (editedIntent: Intent) => void
  onCancel: () => void
}

// ─────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────

const IntentEditor: React.FC<IntentEditorProps> = ({
  intent,
  onSave,
  onCancel,
}) => {
  // Deep clone the intent for editing
  const [editedIntent, setEditedIntent] = useState<Intent>(() => JSON.parse(JSON.stringify(intent)))
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Get the appropriate schema for this intent type
  const { schema, fields, title } = useMemo(() => getIntentSchemaAndFields(editedIntent), [editedIntent])

  // Update a field value
  const handleFieldChange = useCallback((key: string, value: unknown) => {
    setEditedIntent(prev => {
      const updated = JSON.parse(JSON.stringify(prev))
      
      // Handle nested fields based on intent type
      if (updated.type === 'signal' || updated.type === 'plugin_template' || updated.type === 'backend') {
        updated.fields = updated.fields || {}
        updated.fields[key] = value
      } else if (updated.type === 'route') {
        // Route has some top-level fields
        if (key === 'name' || key === 'description' || key === 'priority') {
          updated[key] = value
        } else if (key.startsWith('algorithm.')) {
          const algoKey = key.replace('algorithm.', '')
          if (!updated.algorithm) {
            updated.algorithm = { algo_type: 'confidence', params: {} }
          }
          updated.algorithm.params[algoKey] = value
        }
      } else if (updated.type === 'global') {
        updated.fields = updated.fields || {}
        updated.fields[key] = value
      }
      
      return updated
    })
    
    // Clear error for this field
    setErrors(prev => {
      const updated = { ...prev }
      delete updated[key]
      return updated
    })
  }, [])

  // Update top-level field (name, etc.)
  const handleNameChange = useCallback((value: string) => {
    setEditedIntent(prev => {
      const updated = JSON.parse(JSON.stringify(prev))
      if ('name' in updated) {
        updated.name = value
      }
      return updated
    })
  }, [])

  // Validate and save
  const handleSave = useCallback(() => {
    // Basic validation
    const newErrors: Record<string, string> = {}
    
    for (const field of schema) {
      if (field.required) {
        const value = fields[field.key]
        if (value === undefined || value === null || value === '') {
          newErrors[field.key] = `${field.label} is required`
        }
      }
    }
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }
    
    onSave(editedIntent)
  }, [schema, fields, editedIntent, onSave])

  return (
    <div className={styles.overlay}>
      <div className={styles.modal}>
        <div className={styles.header}>
          <h3 className={styles.title}>{title}</h3>
          <button className={styles.closeBtn} onClick={onCancel}>
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <div className={styles.content}>
          {/* Name field (for most intent types) */}
          {'name' in editedIntent && editedIntent.type !== 'global' && (
            <div className={styles.field}>
              <label className={styles.label}>
                Name <span className={styles.required}>*</span>
              </label>
              <input
                type="text"
                className={styles.input}
                value={(editedIntent as { name: string }).name}
                onChange={(e) => handleNameChange(e.target.value)}
                placeholder="Entity name"
              />
            </div>
          )}

          {/* Dynamic fields from schema */}
          {schema.map(field => (
            <DynamicField
              key={field.key}
              schema={field}
              value={fields[field.key]}
              error={errors[field.key]}
              onChange={(value) => handleFieldChange(field.key, value)}
            />
          ))}
        </div>

        <div className={styles.footer}>
          <button className={styles.cancelBtn} onClick={onCancel}>
            Cancel
          </button>
          <button className={styles.saveBtn} onClick={handleSave}>
            Save Changes
          </button>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// Dynamic Field Component
// ─────────────────────────────────────────────

interface DynamicFieldProps {
  schema: FieldSchema
  value: unknown
  error?: string
  onChange: (value: unknown) => void
}

const DynamicField: React.FC<DynamicFieldProps> = ({
  schema,
  value,
  error,
  onChange,
}) => {
  const { key, label, type, options, required, placeholder, description } = schema

  const renderInput = () => {
    switch (type) {
      case 'string':
        return (
          <input
            type="text"
            className={`${styles.input} ${error ? styles.inputError : ''}`}
            value={(value as string) ?? ''}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder}
          />
        )

      case 'number':
        return (
          <input
            type="number"
            className={`${styles.input} ${error ? styles.inputError : ''}`}
            value={value !== undefined && value !== null ? String(value) : ''}
            onChange={(e) => {
              const v = e.target.value
              onChange(v === '' ? undefined : parseFloat(v))
            }}
            placeholder={placeholder}
            step="any"
          />
        )

      case 'boolean':
        return (
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={Boolean(value)}
              onChange={(e) => onChange(e.target.checked)}
            />
            <span>{label}</span>
          </label>
        )

      case 'select':
        return (
          <select
            className={`${styles.select} ${error ? styles.inputError : ''}`}
            value={(value as string) ?? ''}
            onChange={(e) => onChange(e.target.value || undefined)}
          >
            <option value="">-- Select --</option>
            {options?.map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        )

      case 'string[]':
        return (
          <StringArrayField
            value={(value as string[]) ?? []}
            onChange={onChange}
            placeholder={placeholder}
            error={!!error}
          />
        )

      case 'json':
        return (
          <JsonField
            value={value}
            onChange={onChange}
            placeholder={placeholder}
            error={!!error}
          />
        )

      default:
        return (
          <input
            type="text"
            className={styles.input}
            value={String(value ?? '')}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder}
          />
        )
    }
  }

  // Boolean fields render differently
  if (type === 'boolean') {
    return (
      <div className={styles.field}>
        {renderInput()}
        {description && <div className={styles.description}>{description}</div>}
      </div>
    )
  }

  return (
    <div className={styles.field}>
      <label className={styles.label}>
        {label}
        {required && <span className={styles.required}>*</span>}
      </label>
      {renderInput()}
      {description && <div className={styles.description}>{description}</div>}
      {error && <div className={styles.error}>{error}</div>}
    </div>
  )
}

// ─────────────────────────────────────────────
// String Array Field
// ─────────────────────────────────────────────

interface StringArrayFieldProps {
  value: string[]
  onChange: (value: string[]) => void
  placeholder?: string
  error?: boolean
}

const StringArrayField: React.FC<StringArrayFieldProps> = ({
  value,
  onChange,
  placeholder,
  error,
}) => {
  const [inputValue, setInputValue] = useState('')

  const handleAdd = () => {
    const trimmed = inputValue.trim()
    if (trimmed && !value.includes(trimmed)) {
      onChange([...value, trimmed])
      setInputValue('')
    }
  }

  const handleRemove = (index: number) => {
    onChange(value.filter((_, i) => i !== index))
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAdd()
    }
  }

  return (
    <div className={styles.arrayField}>
      <div className={styles.arrayInput}>
        <input
          type="text"
          className={`${styles.input} ${error ? styles.inputError : ''}`}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || 'Add item...'}
        />
        <button type="button" className={styles.addBtn} onClick={handleAdd}>
          +
        </button>
      </div>
      {value.length > 0 && (
        <div className={styles.arrayItems}>
          {value.map((item, index) => (
            <span key={index} className={styles.arrayItem}>
              {item}
              <button
                type="button"
                className={styles.removeBtn}
                onClick={() => handleRemove(index)}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────
// JSON Field
// ─────────────────────────────────────────────

interface JsonFieldProps {
  value: unknown
  onChange: (value: unknown) => void
  placeholder?: string
  error?: boolean
}

const JsonField: React.FC<JsonFieldProps> = ({
  value,
  onChange,
  placeholder,
  error,
}) => {
  const [text, setText] = useState(() => {
    if (value === undefined || value === null) return ''
    return JSON.stringify(value, null, 2)
  })
  const [jsonError, setJsonError] = useState<string | null>(null)

  const handleChange = (newText: string) => {
    setText(newText)
    if (!newText.trim()) {
      setJsonError(null)
      onChange(undefined)
      return
    }
    try {
      const parsed = JSON.parse(newText)
      setJsonError(null)
      onChange(parsed)
    } catch (e) {
      setJsonError('Invalid JSON')
    }
  }

  return (
    <div className={styles.jsonField}>
      <textarea
        className={`${styles.textarea} ${error || jsonError ? styles.inputError : ''}`}
        value={text}
        onChange={(e) => handleChange(e.target.value)}
        placeholder={placeholder || '{ "key": "value" }'}
        rows={4}
      />
      {jsonError && <div className={styles.error}>{jsonError}</div>}
    </div>
  )
}

// ─────────────────────────────────────────────
// Helper: Get Schema and Fields
// ─────────────────────────────────────────────

function getIntentSchemaAndFields(intent: Intent): {
  schema: FieldSchema[]
  fields: Record<string, unknown>
  title: string
} {
  switch (intent.type) {
    case 'signal': {
      const signalIntent = intent as SignalIntent
      return {
        schema: getSignalFieldSchema(signalIntent.signal_type),
        fields: signalIntent.fields || {},
        title: `Edit Signal: ${signalIntent.signal_type}("${signalIntent.name}")`,
      }
    }

    case 'plugin_template': {
      const pluginIntent = intent as PluginTemplateIntent
      return {
        schema: getPluginFieldSchema(pluginIntent.plugin_type),
        fields: pluginIntent.fields || {},
        title: `Edit Plugin: ${pluginIntent.name} (${pluginIntent.plugin_type})`,
      }
    }

    case 'backend': {
      const backendIntent = intent as BackendIntent
      // Backend doesn't have a schema helper, use generic fields
      return {
        schema: [
          { key: 'endpoint', label: 'Endpoint', type: 'string', placeholder: 'http://...' },
          { key: 'api_key', label: 'API Key', type: 'string', placeholder: 'Optional API key' },
        ],
        fields: backendIntent.fields || {},
        title: `Edit Backend: ${backendIntent.name} (${backendIntent.backend_type})`,
      }
    }

    case 'route': {
      const routeIntent = intent as RouteIntent
      const baseSchema: FieldSchema[] = [
        { key: 'description', label: 'Description', type: 'string', placeholder: 'Route description' },
        { key: 'priority', label: 'Priority', type: 'number', placeholder: '10' },
      ]
      
      // Add algorithm fields if present
      let algoSchema: FieldSchema[] = []
      if (routeIntent.algorithm) {
        algoSchema = getAlgorithmFieldSchema(routeIntent.algorithm.algo_type).map(f => ({
          ...f,
          key: `algorithm.${f.key}`,
          label: `[Algo] ${f.label}`,
        }))
      }

      return {
        schema: [...baseSchema, ...algoSchema],
        fields: {
          description: routeIntent.description,
          priority: routeIntent.priority,
          ...(routeIntent.algorithm?.params
            ? Object.fromEntries(
                Object.entries(routeIntent.algorithm.params).map(([k, v]) => [`algorithm.${k}`, v])
              )
            : {}),
        },
        title: `Edit Route: ${routeIntent.name}`,
      }
    }

    case 'global':
      return {
        schema: [
          { key: 'default_timeout', label: 'Default Timeout', type: 'string', placeholder: '30s' },
          { key: 'max_retries', label: 'Max Retries', type: 'number', placeholder: '3' },
        ],
        fields: intent.fields || {},
        title: 'Edit Global Settings',
      }

    default:
      return {
        schema: [],
        fields: {},
        title: 'Edit Intent',
      }
  }
}

export default IntentEditor
