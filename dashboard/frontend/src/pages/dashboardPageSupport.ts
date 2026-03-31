export interface SignalConfig {
  name?: string
  type?: string
  [key: string]: unknown
}

export interface DecisionRule {
  name?: string
  description?: string
  priority?: number
  rules?: unknown[]
  modelRefs?: unknown[]
  plugins?: unknown[]
  [key: string]: unknown
}

export interface RouterConfig {
  signals?: Record<string, SignalConfig[]>
  decisions?: DecisionRule[]
  providers?: {
    defaults?: {
      default_model?: string
    }
    models?: Array<{
      name?: string
      backend_refs?: Array<{ name?: string }>
      endpoints?: Array<{ name?: string }>
      preferred_endpoints?: string[]
      [key: string]: unknown
    }>
    vllm_endpoints?: unknown[]
    [key: string]: unknown
  }
  routing?: {
    signals?: Record<string, SignalConfig[]>
    decisions?: DecisionRule[]
  }
  vllm_endpoints?: Array<{ name?: string }>
  plugins?: Record<string, unknown>
  global?: Record<string, unknown>
  [key: string]: unknown
}

export interface SignalStats {
  total: number
  byType: Record<string, number>
}

export interface CategorizedDecisions {
  guardrails: DecisionRule[]
  routing: DecisionRule[]
  fallbacks: DecisionRule[]
}

export interface DashboardHeroMeta {
  label: string
  value: string
}

export interface DashboardSurfaceStatus {
  key: string
  label: string
  tone: 'ok' | 'warn' | 'danger' | 'neutral'
}

export interface DashboardActionLink {
  key: string
  label: string
  to: string
}

export interface DashboardStatCard {
  key: 'models' | 'decisions' | 'signals' | 'services' | 'model-status'
  value: string
  label: string
  detail?: string
  to: string
  tone: 'purple' | 'cyan' | 'lime' | 'success' | 'warning' | 'danger'
  emphasis?: 'default' | 'wide'
}

export interface ModelStatusSnapshot {
  value: string
  detail: string
  tone: 'ok' | 'warn' | 'error' | 'down'
}

export type DecisionCategory = 'guardrail' | 'routing' | 'fallback'

export const SIGNAL_COLORS: Record<string, string> = {
  keywords: '#4EC9B0',
  embeddings: '#9CDCFE',
  domains: '#DCDCAA',
  fact_check: '#CE9178',
  user_feedbacks: '#C586C0',
  preferences: '#4FC1FF',
  language: '#B5CEA8',
  context: '#D7BA7D',
  complexity: '#569CD6',
  modality: '#D4D4D4',
  authz: '#F48771',
  jailbreak: '#F48771',
  pii: '#FF6B6B',
}

export function countSignals(cfg: RouterConfig): SignalStats {
  const byType: Record<string, number> = {}
  let total = 0
  const signals = cfg.routing?.signals ?? cfg.signals
  if (signals) {
    for (const [type, items] of Object.entries(signals)) {
      if (Array.isArray(items)) {
        byType[type] = items.length
        total += items.length
      }
    }
  }
  return { total, byType }
}

export function countDecisions(cfg: RouterConfig): number {
  const decisions = cfg.routing?.decisions ?? cfg.decisions
  return Array.isArray(decisions) ? decisions.length : 0
}

export function countModels(cfg: RouterConfig): number {
  const models = cfg.providers?.models
  if (Array.isArray(models)) {
    return models.length
  }

  const legacyRootEndpoints = cfg.vllm_endpoints
  if (Array.isArray(legacyRootEndpoints)) return legacyRootEndpoints.length

  const legacyProviderEndpoints = cfg.providers?.vllm_endpoints
  return Array.isArray(legacyProviderEndpoints) ? legacyProviderEndpoints.length : 0
}

export function countPlugins(cfg: RouterConfig): number {
  const decisions = cfg.routing?.decisions ?? cfg.decisions
  if (Array.isArray(decisions)) {
    return decisions.reduce(
      (count, decision) => count + (Array.isArray(decision.plugins) ? decision.plugins.length : 0),
      0,
    )
  }

  if (!cfg.plugins || typeof cfg.plugins !== 'object') return 0
  return Object.keys(cfg.plugins).length
}

export function getDecisionCategory(priority?: number): DecisionCategory {
  if (priority == null) return 'routing'
  if (priority >= 999) return 'guardrail'
  if (priority <= 100) return 'fallback'
  return 'routing'
}

export function categorizeDecisions(decisions?: DecisionRule[]): CategorizedDecisions {
  if (!decisions?.length) {
    return { guardrails: [], routing: [], fallbacks: [] }
  }

  const guardrails: DecisionRule[] = []
  const routing: DecisionRule[] = []
  const fallbacks: DecisionRule[] = []

  for (const decision of decisions) {
    const category = getDecisionCategory(decision.priority)
    if (category === 'guardrail') {
      guardrails.push(decision)
    } else if (category === 'fallback') {
      fallbacks.push(decision)
    } else {
      routing.push(decision)
    }
  }

  return { guardrails, routing, fallbacks }
}

export function formatOverallLabel(overall?: string): string {
  if (!overall) return 'Unavailable'
  if (overall === 'not_running') return 'Not Running'
  return `${overall.charAt(0).toUpperCase()}${overall.slice(1)}`
}

export function formatSurfaceTimestamp(value: Date | null): string | null {
  if (!value) return null
  return `Updated ${value.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
}

export function buildDashboardHeroMeta(args: {
  modelCount: number
  decisionCount: number
  signalCount: number
  loadedModels: number
  knownModels: number
}): DashboardHeroMeta[] {
  const runtimeCoverage = args.knownModels > 0
    ? `${args.loadedModels}/${args.knownModels} models ready`
    : `${args.modelCount} models tracked`

  return [
    { label: 'Runtime coverage', value: runtimeCoverage },
    { label: 'Decision stack', value: `${args.decisionCount} live layers` },
    { label: 'Signal mesh', value: `${args.signalCount} active signals` },
  ]
}

export function buildDashboardStatusPills(args: {
  overall?: string
  deploymentType?: string
  loadedModels: number
  knownModels: number
  pluginCount: number
}): DashboardSurfaceStatus[] {
  const overallTone: DashboardSurfaceStatus['tone'] = args.overall === 'healthy'
    ? 'ok'
    : args.overall === 'degraded'
      ? 'warn'
      : args.overall
        ? 'danger'
        : 'neutral'

  const readinessTone: DashboardSurfaceStatus['tone'] = args.knownModels > 0 && args.loadedModels >= args.knownModels
    ? 'ok'
    : args.loadedModels > 0
      ? 'warn'
      : 'neutral'

  const pills: DashboardSurfaceStatus[] = [
    {
      key: 'overall',
      label: formatOverallLabel(args.overall),
      tone: overallTone,
    },
  ]

  if (args.deploymentType && args.deploymentType !== 'none') {
    pills.push({
      key: 'deployment',
      label: args.deploymentType,
      tone: 'neutral',
    })
  }

  if (args.knownModels > 0) {
    pills.push({
      key: 'readiness',
      label: `${args.loadedModels}/${args.knownModels} ready`,
      tone: readinessTone,
    })
  }

  if (args.pluginCount > 0) {
    pills.push({
      key: 'plugins',
      label: `${args.pluginCount} plugins`,
      tone: 'neutral',
    })
  }

  return pills
}

export function buildDashboardActionLinks(showMLSetupQuickLink: boolean): DashboardActionLink[] {
  const actions: DashboardActionLink[] = [
    { key: 'status', label: 'Status', to: '/status' },
    { key: 'playground', label: 'Playground', to: '/playground' },
    { key: 'topology', label: 'Topology', to: '/topology' },
    { key: 'builder', label: 'Builder', to: '/builder' },
  ]

  if (showMLSetupQuickLink) {
    actions.push({ key: 'ml-setup', label: 'ML Setup', to: '/ml-setup' })
  }

  return actions
}

export function buildDashboardStatCards(args: {
  modelCount: number
  decisionCount: number
  signalCount: number
  healthyServices: number
  totalServices: number
  modelStatus: ModelStatusSnapshot
  overall?: string
  loadedModels: number
  knownModels: number
}): DashboardStatCard[] {
  const servicesTone: DashboardStatCard['tone'] = args.overall === 'healthy'
    ? 'success'
    : args.overall === 'degraded'
      ? 'warning'
      : 'danger'

  const modelTone: DashboardStatCard['tone'] = args.modelStatus.tone === 'ok'
    ? 'success'
    : args.modelStatus.tone === 'warn'
      ? 'warning'
      : 'danger'

  const modelDetail = args.knownModels > 0
    ? `${args.loadedModels}/${args.knownModels} ready`
    : 'Open runtime inventory'

  return [
    {
      key: 'models',
      value: `${args.modelCount}`,
      label: 'Models',
      detail: modelDetail,
      to: '/config/models',
      tone: 'purple',
    },
    {
      key: 'decisions',
      value: `${args.decisionCount}`,
      label: 'Decisions',
      detail: 'Inspect live routing layers',
      to: '/config/decisions',
      tone: 'cyan',
    },
    {
      key: 'signals',
      value: `${args.signalCount}`,
      label: 'Signals',
      detail: 'Review active detectors and heuristics',
      to: '/config/signals',
      tone: 'lime',
    },
    {
      key: 'services',
      value: `${args.healthyServices}/${args.totalServices}`,
      label: 'Services Healthy',
      detail: `${formatOverallLabel(args.overall)} runtime`,
      to: '/status',
      tone: servicesTone,
    },
    {
      key: 'model-status',
      value: args.modelStatus.value,
      label: 'Model Status',
      detail: args.modelStatus.detail,
      to: '/status',
      tone: modelTone,
      emphasis: 'wide',
    },
  ]
}
