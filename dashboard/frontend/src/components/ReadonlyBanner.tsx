import React, { useState } from 'react'
import { useReadonly } from '../contexts/ReadonlyContext'
import InviteCodeDialog from './InviteCodeDialog'
import styles from './ReadonlyBanner.module.css'

const ReadonlyBanner: React.FC = () => {
  const { isReadonly, hasInvite, inviteEnabled, submitInvite, logout } = useReadonly()
  const [dialogOpen, setDialogOpen] = useState(false)
  const [loggingOut, setLoggingOut] = useState(false)

  // If readonly is disabled, don't show banner
  if (!isReadonly) {
    return null
  }

  // If user has valid invite, show unlocked state
  if (hasInvite) {
    return (
      <div className={`${styles.banner} ${styles.unlocked}`}>
        <span className={styles.icon}>🔓</span>
        <span className={styles.text}>
          Beta access unlocked. You have write permissions.
        </span>
        <button
          className={styles.logoutButton}
          onClick={async () => {
            setLoggingOut(true)
            await logout()
            setLoggingOut(false)
          }}
          disabled={loggingOut}
        >
          {loggingOut ? 'Logging out...' : 'Logout'}
        </button>
      </div>
    )
  }

  return (
    <>
      <div className={styles.banner}>
        <span className={styles.icon}>🔒</span>
        <span className={styles.text}>
          Dashboard is in read-only mode. Configuration editing is disabled.
        </span>
        {inviteEnabled && (
          <button
            className={styles.inviteButton}
            onClick={() => setDialogOpen(true)}
          >
            🔑 Enter Invite Code
          </button>
        )}
      </div>

      <InviteCodeDialog
        isOpen={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onSubmit={submitInvite}
      />
    </>
  )
}

export default ReadonlyBanner
