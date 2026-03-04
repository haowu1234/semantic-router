import React, { useState, useEffect, useRef } from 'react'
import styles from './InviteCodeDialog.module.css'

interface InviteCodeDialogProps {
  isOpen: boolean
  onClose: () => void
  onSubmit: (code: string) => Promise<{ ok: boolean; error?: string }>
}

const InviteCodeDialog: React.FC<InviteCodeDialogProps> = ({
  isOpen,
  onClose,
  onSubmit,
}) => {
  const [code, setCode] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (isOpen) {
      setCode('')
      setError(null)
      // Focus input after dialog opens
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isOpen])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    const trimmedCode = code.trim()
    if (!trimmedCode) {
      setError('Please enter an invite code')
      return
    }

    setSubmitting(true)
    setError(null)

    try {
      const result = await onSubmit(trimmedCode)
      if (result.ok) {
        onClose()
      } else {
        setError(result.error || 'Invalid invite code')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to verify invite code')
    } finally {
      setSubmitting(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose()
    }
  }

  if (!isOpen) return null

  return (
    <div className={styles.overlay} onClick={onClose} onKeyDown={handleKeyDown}>
      <div className={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h2 className={styles.title}>🔑 Enter Invite Code</h2>
          <button className={styles.closeButton} onClick={onClose} type="button">
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.content}>
            <p className={styles.description}>
              Enter your beta invite code to unlock write permissions and access all dashboard features.
            </p>

            <div className={styles.field}>
              <label className={styles.label} htmlFor="invite-code">
                Invite Code
              </label>
              <input
                ref={inputRef}
                id="invite-code"
                type="text"
                className={styles.input}
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="invite-xxxxxx.xxxxxx"
                disabled={submitting}
                autoComplete="off"
                spellCheck={false}
              />
            </div>

            {error && (
              <div className={styles.error}>
                <span className={styles.errorIcon}>⚠️</span>
                {error}
              </div>
            )}
          </div>

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.cancelButton}
              onClick={onClose}
              disabled={submitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className={styles.submitButton}
              disabled={submitting || !code.trim()}
            >
              {submitting ? 'Verifying...' : 'Unlock Access'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default InviteCodeDialog
