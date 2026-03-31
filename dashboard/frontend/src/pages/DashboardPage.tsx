import React, {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { useNavigate } from 'react-router-dom'
import RouterModelInventory from '../components/RouterModelInventory'
import { useAuth } from '../contexts/AuthContext'
import { canAccessMLSetup } from '../utils/accessControl'
import {
  getLoadedModelCount,
  getModelStatusSummary,
  getRouterModelAnchor,
  getTotalKnownModelCount,
  type SystemStatus,
} from '../utils/routerRuntime'
import styles from './DashboardPage.module.css'
import DashboardPageFlow from './dashboardPageFlow'
import {
  DashboardTelemetryStrip,
  DashboardDecisionsCard,
  DashboardHealthCard,
  DashboardOverviewHero,
  DashboardSignalBreakdownCard,
  DashboardStatsGrid,
} from './dashboardPageSections'
import {
  buildDashboardActionLinks,
  buildDashboardHeroMeta,
  buildDashboardStatCards,
  buildDashboardStatusPills,
  buildDashboardTelemetryCards,
  categorizeDecisions,
  countDecisions,
  countModels,
  countPlugins,
  countSignals,
  formatSurfaceTimestamp,
  type RouterConfig,
} from './dashboardPageSupport'

type DashboardFetchScope = 'all' | 'status'

const DashboardPage: React.FC = () => {
  const navigate = useNavigate()
  const { user } = useAuth()

  const [config, setConfig] = useState<RouterConfig | null>(null)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const hasDataRef = useRef(false)
  const pollTickRef = useRef(0)
  const requestIdRef = useRef(0)

  const fetchDashboard = useCallback(async (scope: DashboardFetchScope, manual = false) => {
    if (manual) {
      setRefreshing(true)
    }

    const requestId = ++requestIdRef.current

    try {
      let nextConfig: RouterConfig | undefined
      let nextStatus: SystemStatus | undefined

      if (scope === 'all') {
        const [configResponse, statusResponse] = await Promise.all([
          fetch('/api/router/config/all'),
          fetch('/api/status'),
        ])

        if (configResponse.ok) {
          nextConfig = await configResponse.json() as RouterConfig
        }

        if (statusResponse.ok) {
          nextStatus = await statusResponse.json() as SystemStatus
        }
      } else {
        const statusResponse = await fetch('/api/status')
        if (statusResponse.ok) {
          nextStatus = await statusResponse.json() as SystemStatus
        }
      }

      if (requestId !== requestIdRef.current) {
        return
      }

      if (nextConfig === undefined && nextStatus === undefined) {
        if (!hasDataRef.current || manual) {
          throw new Error('Dashboard endpoints did not return data.')
        }
        return
      }

      hasDataRef.current = true
      const updatedAt = new Date()

      startTransition(() => {
        if (nextConfig !== undefined) {
          setConfig(nextConfig)
        }
        if (nextStatus !== undefined) {
          setStatus(nextStatus)
        }
        setLastUpdated(updatedAt)
        setError(null)
      })
    } catch (err) {
      if (requestId !== requestIdRef.current) {
        return
      }

      if (manual || !hasDataRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to load dashboard data')
      }
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false)
        if (manual) {
          setRefreshing(false)
        }
      }
    }
  }, [])

  useEffect(() => {
    void fetchDashboard('all')

    const intervalId = window.setInterval(() => {
      if (document.visibilityState !== 'visible') {
        return
      }

      pollTickRef.current += 1
      void fetchDashboard(pollTickRef.current % 3 === 0 ? 'all' : 'status')
    }, 10000)

    const onConfigDeployed = () => {
      pollTickRef.current = 0
      void fetchDashboard('all')
    }

    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        pollTickRef.current = 0
        void fetchDashboard('all')
      }
    }

    window.addEventListener('config-deployed', onConfigDeployed)
    document.addEventListener('visibilitychange', onVisibilityChange)

    return () => {
      clearInterval(intervalId)
      window.removeEventListener('config-deployed', onConfigDeployed)
      document.removeEventListener('visibilitychange', onVisibilityChange)
    }
  }, [fetchDashboard])

  const signalStats = useMemo(
    () => (config ? countSignals(config) : { total: 0, byType: {} }),
    [config],
  )
  const decisionCount = useMemo(() => (config ? countDecisions(config) : 0), [config])
  const modelCount = useMemo(() => (config ? countModels(config) : 0), [config])
  const pluginCount = useMemo(() => (config ? countPlugins(config) : 0), [config])
  const currentDecisions = useMemo(
    () => config?.routing?.decisions ?? config?.decisions ?? [],
    [config],
  )
  const categorizedDecisions = useMemo(
    () => categorizeDecisions(currentDecisions),
    [currentDecisions],
  )

  const healthyServices = useMemo(
    () => status?.services.filter((service) => service.healthy).length ?? 0,
    [status],
  )
  const totalServices = useMemo(() => status?.services.length ?? 0, [status])
  const modelStatus = useMemo(() => getModelStatusSummary(status), [status])
  const loadedModels = useMemo(() => getLoadedModelCount(status?.models), [status])
  const knownModels = useMemo(() => getTotalKnownModelCount(status?.models), [status])
  const showMLSetupQuickLink = canAccessMLSetup(user)

  const heroMeta = useMemo(
    () => buildDashboardHeroMeta({
      modelCount,
      decisionCount,
      signalCount: signalStats.total,
      loadedModels,
      knownModels,
    }),
    [decisionCount, knownModels, loadedModels, modelCount, signalStats.total],
  )

  const heroStatuses = useMemo(
    () => buildDashboardStatusPills({
      overall: status?.overall,
      deploymentType: status?.deployment_type,
      loadedModels,
      knownModels,
      pluginCount,
    }),
    [knownModels, loadedModels, pluginCount, status?.deployment_type, status?.overall],
  )

  const heroActions = useMemo(
    () => buildDashboardActionLinks(showMLSetupQuickLink),
    [showMLSetupQuickLink],
  )

  const statCards = useMemo(
    () => buildDashboardStatCards({
      modelCount,
      decisionCount,
      signalCount: signalStats.total,
      healthyServices,
      totalServices,
      modelStatus,
      overall: status?.overall,
      loadedModels,
      knownModels,
    }),
    [
      decisionCount,
      healthyServices,
      knownModels,
      loadedModels,
      modelCount,
      modelStatus,
      signalStats.total,
      status?.overall,
      totalServices,
    ],
  )
  const telemetryCards = useMemo(
    () => buildDashboardTelemetryCards({
      loadedModels,
      knownModels,
      healthyServices,
      totalServices,
      decisionCount,
      signalStats,
      pluginCount,
      overall: status?.overall,
    }),
    [
      decisionCount,
      healthyServices,
      knownModels,
      loadedModels,
      pluginCount,
      signalStats,
      status?.overall,
      totalServices,
    ],
  )

  const previewModelLimit = 6
  const lastUpdatedLabel = useMemo(
    () => formatSurfaceTimestamp(lastUpdated),
    [lastUpdated],
  )
  const bottomSectionCount = Number(signalStats.total > 0) + Number(currentDecisions.length > 0)

  if (loading && !config && !status) {
    return (
      <div className={styles.page}>
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <p>Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.page} data-testid="dashboard-page">
      <div className={`${styles.surfaceReveal} ${styles.revealHero}`}>
        <DashboardOverviewHero
          meta={heroMeta}
          statusPills={heroStatuses}
          actions={heroActions}
          lastUpdatedLabel={lastUpdatedLabel}
          refreshing={refreshing}
          onRefresh={() => {
            pollTickRef.current = 0
            void fetchDashboard('all', true)
          }}
          onNavigate={(to) => navigate(to)}
        />
      </div>

      {error ? (
        <div className={styles.errorBanner}>
          <span>Failed to load data: {error}</span>
          <button
            onClick={() => {
              pollTickRef.current = 0
              void fetchDashboard('all', true)
            }}
          >
            Retry
          </button>
        </div>
      ) : null}

      <div className={`${styles.surfaceReveal} ${styles.revealTelemetry}`}>
        <DashboardTelemetryStrip items={telemetryCards} onNavigate={(to) => navigate(to)} />
      </div>

      <div className={`${styles.surfaceReveal} ${styles.revealStats}`}>
        <DashboardStatsGrid stats={statCards} onNavigate={(to) => navigate(to)} />
      </div>

      <div className={`${styles.mainGrid} ${styles.surfaceReveal} ${styles.revealMain}`}>
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <div className={styles.cardHeaderInfo}>
              <h2 className={styles.cardTitle}>Intelligence Layers</h2>
              <span className={styles.cardSubtitle}>Routing preview before the full topology view</span>
            </div>
            <button className={styles.cardAction} onClick={() => navigate('/topology')}>
              Open Topology &rsaquo;
            </button>
          </div>
          <div className={styles.flowContainer}>
            {config ? (
              <DashboardPageFlow
                signals={signalStats}
                decisions={decisionCount}
                models={modelCount}
                plugins={pluginCount}
              />
            ) : (
              <div className={styles.emptyState}>No configuration loaded</div>
            )}
          </div>
        </div>

        <DashboardHealthCard
          status={status}
          onOpenStatus={() => navigate('/status')}
        />
      </div>

      <section className={`${styles.deferredSection} ${styles.surfaceReveal} ${styles.revealInventory}`}>
        <div className={styles.card}>
          <div className={styles.cardHeader}>
            <div className={styles.cardHeaderInfo}>
              <h2 className={styles.cardTitle}>Loaded Models</h2>
              <span className={styles.cardSubtitle}>
                {knownModels > 0 ? `${loadedModels}/${knownModels} ready` : 'Runtime inventory'}
              </span>
            </div>
            <button className={styles.cardAction} onClick={() => navigate('/status')}>
              Status &rsaquo;
            </button>
          </div>
          <RouterModelInventory
            mode="preview"
            previewLimit={previewModelLimit > 0 ? previewModelLimit : undefined}
            modelsInfo={status?.models}
            emptyMessage="Router model inventory will appear here after the router reports its active models."
            onSelectModel={(model) => navigate(`/status#${encodeURIComponent(getRouterModelAnchor(model))}`)}
          />
        </div>
      </section>

      {bottomSectionCount > 0 ? (
        <section
          className={`${styles.bottomGrid} ${bottomSectionCount === 1 ? styles.bottomGridSingle : ''} ${styles.deferredSection} ${styles.surfaceReveal} ${styles.revealBottom}`}
        >
          <DashboardSignalBreakdownCard
            signalStats={signalStats}
            onManageSignals={() => navigate('/config/signals')}
          />
          <DashboardDecisionsCard
            currentDecisions={currentDecisions}
            categorizedDecisions={categorizedDecisions}
            onManageDecisions={() => navigate('/config/decisions')}
          />
        </section>
      ) : null}
    </div>
  )
}

export default DashboardPage
