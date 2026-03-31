import DashboardSurfaceHero from '../components/DashboardSurfaceHero'
import { type SystemStatus } from '../utils/routerRuntime'
import styles from './DashboardPage.module.css'
import {
  formatOverallLabel,
  getDecisionCategory,
  SIGNAL_COLORS,
  type DashboardTelemetryCard,
  type CategorizedDecisions,
  type DashboardActionLink,
  type DashboardHeroMeta,
  type DashboardStatCard,
  type DashboardSurfaceStatus,
  type DecisionRule,
  type SignalStats,
} from './dashboardPageSupport'

interface DashboardOverviewHeroProps {
  meta: DashboardHeroMeta[]
  statusPills: DashboardSurfaceStatus[]
  actions: DashboardActionLink[]
  lastUpdatedLabel: string | null
  refreshing: boolean
  onRefresh: () => void
  onNavigate: (to: string) => void
}

interface DashboardStatsGridProps {
  stats: DashboardStatCard[]
  onNavigate: (to: string) => void
}

interface DashboardHealthCardProps {
  status: SystemStatus | null
  onOpenStatus: () => void
}

interface DashboardTelemetryStripProps {
  items: DashboardTelemetryCard[]
  onNavigate: (to: string) => void
}

interface DashboardSignalBreakdownCardProps {
  signalStats: SignalStats
  onManageSignals: () => void
}

interface DashboardDecisionsCardProps {
  currentDecisions: DecisionRule[]
  categorizedDecisions: CategorizedDecisions
  onManageDecisions: () => void
}

function renderStatIcon(key: DashboardStatCard['key']): JSX.Element {
  switch (key) {
    case 'models':
      return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <rect x="2" y="3" width="20" height="18" rx="3" />
          <path d="M8 7v10M12 7v10M16 7v10" />
        </svg>
      )
    case 'decisions':
      return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <path d="M4 6h16M4 12h8M4 18h12" />
        </svg>
      )
    case 'signals':
      return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <circle cx="12" cy="12" r="3" />
          <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
        </svg>
      )
    case 'services':
      return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
        </svg>
      )
    case 'model-status':
      return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3v10" />
          <path d="M8.5 9.5 12 13l3.5-3.5" />
          <path d="M4 19h16" />
        </svg>
      )
  }
}

function getStatToneClass(tone: DashboardStatCard['tone']): string {
  switch (tone) {
    case 'purple':
      return styles.statCardPurple
    case 'cyan':
      return styles.statCardCyan
    case 'lime':
      return styles.statCardLime
    case 'success':
      return styles.statCardSuccess
    case 'warning':
      return styles.statCardWarning
    case 'danger':
      return styles.statCardDanger
  }
}

function getStatusToneClass(tone: DashboardSurfaceStatus['tone']): string {
  switch (tone) {
    case 'ok':
      return styles.heroStatusOk
    case 'warn':
      return styles.heroStatusWarn
    case 'danger':
      return styles.heroStatusDanger
    case 'neutral':
      return styles.heroStatusNeutral
  }
}

function getTelemetryToneClass(tone: DashboardTelemetryCard['tone']): string {
  switch (tone) {
    case 'lime':
      return styles.telemetryCardLime
    case 'cyan':
      return styles.telemetryCardCyan
    case 'purple':
      return styles.telemetryCardPurple
    case 'amber':
      return styles.telemetryCardAmber
  }
}

function getTelemetryStatusClass(status: DashboardTelemetryCard['status']): string {
  switch (status) {
    case 'stable':
      return styles.telemetryStatusStable
    case 'warming':
      return styles.telemetryStatusWarming
    case 'attention':
      return styles.telemetryStatusAttention
  }
}

function getTelemetryStatusLabel(status: DashboardTelemetryCard['status']): string {
  switch (status) {
    case 'stable':
      return 'Stable'
    case 'warming':
      return 'Warming'
    case 'attention':
      return 'Attention'
  }
}

export function DashboardOverviewHero({
  meta,
  statusPills,
  actions,
  lastUpdatedLabel,
  refreshing,
  onRefresh,
  onNavigate,
}: DashboardOverviewHeroProps) {
  return (
    <DashboardSurfaceHero
      eyebrow="Operator Surface"
      title="Dashboard"
      description="Monitor runtime readiness, decision coverage, and active routing surfaces from one operator console."
      meta={meta}
      panelEyebrow="Live Workspace"
      panelTitle="Router Pulse"
      panelDescription="Track the control plane, confirm model readiness, and jump straight into the next operational surface."
      panelFooter={(
        <div className={styles.heroPanelFooter}>
          <div className={styles.heroStatusRail}>
            {statusPills.map((pill) => (
              <span
                key={pill.key}
                className={`${styles.heroStatusPill} ${getStatusToneClass(pill.tone)}`}
              >
                {pill.label}
              </span>
            ))}
            {lastUpdatedLabel ? (
              <span className={styles.heroTimestamp}>{lastUpdatedLabel}</span>
            ) : null}
          </div>
          <div className={styles.heroActionsRow}>
            <button
              type="button"
              className={styles.heroPrimaryAction}
              onClick={onRefresh}
              disabled={refreshing}
            >
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M14 8A6 6 0 1 1 8 2" strokeLinecap="round" />
                <path d="M14 2v6h-6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              {refreshing ? 'Refreshing...' : 'Refresh'}
            </button>
            {actions.map((action) => (
              <button
                key={action.key}
                type="button"
                className={styles.heroSecondaryAction}
                onClick={() => onNavigate(action.to)}
              >
                {action.label}
              </button>
            ))}
          </div>
        </div>
      )}
    />
  )
}

export function DashboardStatsGrid({ stats, onNavigate }: DashboardStatsGridProps) {
  return (
    <div className={styles.statsGrid}>
      {stats.map((stat) => (
        <button
          key={stat.key}
          type="button"
          className={`${styles.statCard} ${getStatToneClass(stat.tone)} ${stat.emphasis === 'wide' ? styles.statCardWide : ''}`}
          onClick={() => onNavigate(stat.to)}
        >
          <div className={styles.statLead}>
            <div className={styles.statIcon}>
              {renderStatIcon(stat.key)}
            </div>
            <span className={styles.statArrow}>&rsaquo;</span>
          </div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>{stat.label}</span>
            <span className={styles.statValue}>{stat.value}</span>
            {stat.detail ? <span className={styles.statDetail}>{stat.detail}</span> : null}
          </div>
        </button>
      ))}
    </div>
  )
}

export function DashboardTelemetryStrip({ items, onNavigate }: DashboardTelemetryStripProps) {
  return (
    <section className={styles.telemetryStrip} aria-label="Live telemetry">
      {items.map((item) => (
        <button
          key={item.key}
          type="button"
          className={`${styles.telemetryCard} ${getTelemetryToneClass(item.tone)}`}
          onClick={() => onNavigate(item.to)}
        >
          <div className={styles.telemetryTopline}>
            <span className={styles.telemetryLabel}>{item.label}</span>
            <span className={`${styles.telemetryStatus} ${getTelemetryStatusClass(item.status)}`}>
              {getTelemetryStatusLabel(item.status)}
            </span>
          </div>
          <div className={styles.telemetryValueRow}>
            <strong className={styles.telemetryValue}>{item.value}</strong>
            <span className={styles.telemetryPercent}>{item.progress}%</span>
          </div>
          <p className={styles.telemetryDetail}>{item.detail}</p>
          <div className={styles.telemetryProgressTrack}>
            <span className={styles.telemetryProgressFill} style={{ width: `${item.progress}%` }} />
          </div>
        </button>
      ))}
    </section>
  )
}

export function DashboardHealthCard({ status, onOpenStatus }: DashboardHealthCardProps) {
  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div className={styles.cardHeaderInfo}>
          <h2 className={styles.cardTitle}>System Health</h2>
          <span className={styles.cardSubtitle}>Control plane and runtime services</span>
        </div>
        <button className={styles.cardAction} onClick={onOpenStatus}>
          Details &rsaquo;
        </button>
      </div>
      <div className={styles.healthContent}>
        {status ? (
          <>
            <div className={styles.healthOverall}>
              <span
                className={`${styles.healthDot} ${
                  status.overall === 'healthy'
                    ? styles.healthDotGreen
                    : status.overall === 'degraded'
                      ? styles.healthDotYellow
                      : styles.healthDotRed
                }`}
              />
              <span className={styles.healthLabel}>{formatOverallLabel(status.overall)}</span>
              {status.version ? <span className={styles.versionBadge}>v{status.version}</span> : null}
              {status.deployment_type && status.deployment_type !== 'none' ? (
                <span className={styles.deployBadge}>{status.deployment_type}</span>
              ) : null}
            </div>
            <div className={styles.servicesList}>
              {status.services.slice(0, 6).map((service) => (
                <div key={service.name} className={styles.serviceRow}>
                  <span className={`${styles.svcDot} ${service.healthy ? styles.svcDotOk : styles.svcDotFail}`} />
                  <span className={styles.svcName}>{service.name}</span>
                  <span className={`${styles.svcStatus} ${service.healthy ? styles.svcStatusOk : styles.svcStatusFail}`}>
                    {service.status}
                  </span>
                </div>
              ))}
              {status.services.length > 6 ? (
                <div className={styles.moreServices}>+{status.services.length - 6} more</div>
              ) : null}
            </div>
          </>
        ) : (
          <div className={styles.emptyState}>Unable to fetch status</div>
        )}
      </div>
    </div>
  )
}

export function DashboardSignalBreakdownCard({
  signalStats,
  onManageSignals,
}: DashboardSignalBreakdownCardProps) {
  if (signalStats.total === 0) {
    return null
  }

  const maxCount = Math.max(...Object.values(signalStats.byType))

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div className={styles.cardHeaderInfo}>
          <h2 className={styles.cardTitle}>Signal Breakdown</h2>
          <span className={styles.cardSubtitle}>{signalStats.total} total</span>
        </div>
        <button className={styles.cardAction} onClick={onManageSignals}>
          Manage &rsaquo;
        </button>
      </div>
      <div className={styles.signalBreakdown}>
        {Object.entries(signalStats.byType)
          .sort((left, right) => right[1] - left[1])
          .map(([type, count]) => {
            const percentage = maxCount > 0 ? Math.round((count / maxCount) * 100) : 0
            const color = SIGNAL_COLORS[type] || '#999'

            return (
              <div key={type} className={styles.breakdownRow} title={`${type}: ${count} signal(s)`}>
                <span className={styles.breakdownLabel}>
                  <span className={styles.breakdownDot} style={{ background: color }} />
                  {type}
                </span>
                <div className={styles.breakdownBar}>
                  <div className={styles.breakdownFill} style={{ width: `${percentage}%`, background: color }} />
                </div>
                <span className={styles.breakdownCount}>{count}</span>
              </div>
            )
          })}
      </div>
    </div>
  )
}

export function DashboardDecisionsCard({
  currentDecisions,
  categorizedDecisions,
  onManageDecisions,
}: DashboardDecisionsCardProps) {
  if (currentDecisions.length === 0) {
    return null
  }

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div className={styles.cardHeaderInfo}>
          <h2 className={styles.cardTitle}>Decisions Overview</h2>
          <span className={styles.cardSubtitle}>Ordered guardrails, routes, and fallbacks</span>
        </div>
        <button className={styles.cardAction} onClick={onManageDecisions}>
          Manage &rsaquo;
        </button>
      </div>
      <div className={styles.decisionTable}>
        <div className={styles.decisionTableHead}>
          <span>Name</span>
          <span>Priority</span>
          <span>Type</span>
          <span>Models</span>
        </div>
        {[...categorizedDecisions.guardrails, ...categorizedDecisions.routing, ...categorizedDecisions.fallbacks]
          .slice(0, 10)
          .map((decision, index) => {
            const modelNames = Array.isArray(decision.modelRefs)
              ? decision.modelRefs
                .map((modelRef) => {
                  if (modelRef && typeof modelRef === 'object' && 'model' in modelRef) {
                    const name = (modelRef as { model?: string }).model
                    return name ?? ''
                  }
                  return ''
                })
                .filter(Boolean)
                .join(', ')
              : '—'
            const category = getDecisionCategory(decision.priority)

            return (
              <div key={`${decision.name ?? 'decision'}-${index}`} className={styles.decisionTableRow}>
                <span className={styles.decisionName} title={decision.description || decision.name || ''}>
                  {decision.name || `Decision ${index + 1}`}
                </span>
                <span className={styles.decisionPriority}>{decision.priority ?? '—'}</span>
                <span
                  className={`${styles.decisionBadge} ${
                    category === 'guardrail'
                      ? styles.badgeGuardrail
                      : category === 'fallback'
                        ? styles.badgeFallback
                        : styles.badgeRouting
                  }`}
                >
                  {category === 'guardrail' ? 'Guard' : category === 'fallback' ? 'Default' : 'Route'}
                </span>
                <span className={styles.decisionModels} title={modelNames}>{modelNames}</span>
              </div>
            )
          })}
        {currentDecisions.length > 10 ? (
          <button className={styles.decisionTableMore} onClick={onManageDecisions}>
            +{currentDecisions.length - 10} more decisions &rsaquo;
          </button>
        ) : null}
      </div>
    </div>
  )
}
