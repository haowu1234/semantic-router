import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'
import { preloadPlatformAssets } from '../utils/platformAssets'

interface ReadonlyContextType {
  isReadonly: boolean
  isLoading: boolean
  platform: string
  envoyUrl: string
  hasInvite: boolean
  inviteEnabled: boolean
  submitInvite: (code: string) => Promise<{ ok: boolean; error?: string }>
  logout: () => Promise<void>
  refresh: () => Promise<void>
}

const ReadonlyContext = createContext<ReadonlyContextType>({
  isReadonly: false,
  isLoading: true,
  platform: '',
  envoyUrl: '',
  hasInvite: false,
  inviteEnabled: false,
  submitInvite: async () => ({ ok: false, error: 'Not initialized' }),
  logout: async () => {},
  refresh: async () => {},
})

// eslint-disable-next-line react-refresh/only-export-components
export const useReadonly = (): ReadonlyContextType => useContext(ReadonlyContext)

interface ReadonlyProviderProps {
  children: ReactNode
}

export const ReadonlyProvider: React.FC<ReadonlyProviderProps> = ({ children }) => {
  const [isReadonly, setIsReadonly] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [platform, setPlatform] = useState('')
  const [envoyUrl, setEnvoyUrl] = useState('')
  const [hasInvite, setHasInvite] = useState(false)
  const [inviteEnabled, setInviteEnabled] = useState(false)

  const fetchSettings = useCallback(async () => {
    try {
      const response = await fetch('/api/settings')
      if (response.ok) {
        const data = await response.json()
        setIsReadonly(data.readonlyMode || false)
        setHasInvite(data.hasInvite || false)
        setInviteEnabled(data.inviteEnabled || false)
        const platformValue = data.platform || ''
        setPlatform(platformValue)
        setEnvoyUrl(data.envoyUrl || '')
        // Preload platform-specific assets immediately
        preloadPlatformAssets(platformValue)
      }
    } catch (error) {
      console.warn('Failed to fetch dashboard settings:', error)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchSettings()
  }, [fetchSettings])

  const submitInvite = useCallback(async (code: string): Promise<{ ok: boolean; error?: string }> => {
    try {
      const response = await fetch('/api/invite/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
        credentials: 'include', // Important: include cookies
      })

      if (response.ok) {
        // Refresh settings to update hasInvite state
        await fetchSettings()
        return { ok: true }
      } else {
        const data = await response.json().catch(() => ({}))
        return { ok: false, error: data.error || 'Invalid invite code' }
      }
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : 'Network error' }
    }
  }, [fetchSettings])

  const logout = useCallback(async () => {
    try {
      await fetch('/api/invite/logout', {
        method: 'POST',
        credentials: 'include',
      })
      // Refresh settings to update hasInvite state
      await fetchSettings()
    } catch (error) {
      console.warn('Failed to logout:', error)
    }
  }, [fetchSettings])

  const refresh = useCallback(async () => {
    await fetchSettings()
  }, [fetchSettings])

  return (
    <ReadonlyContext.Provider value={{
      isReadonly,
      isLoading,
      platform,
      envoyUrl,
      hasInvite,
      inviteEnabled,
      submitInvite,
      logout,
      refresh,
    }}>
      {children}
    </ReadonlyContext.Provider>
  )
}
