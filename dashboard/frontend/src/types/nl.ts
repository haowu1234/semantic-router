export type NLFieldType =
  | 'string'
  | 'number'
  | 'boolean'
  | 'string[]'
  | 'number[]'
  | 'select'
  | 'json'

export type NLConstructKind = 'route' | 'signal' | 'plugin' | 'algorithm' | 'backend'
export type NLOperationMode = 'generate' | 'modify' | 'fix'
export type NLPlannerResultStatus =
  | 'ready'
  | 'needs_clarification'
  | 'unsupported'
  | 'error'

export interface NLFieldSchema {
  key: string
  label: string
  type: NLFieldType
  options?: string[]
  required?: boolean
  placeholder?: string
  description?: string
}

export interface NLTypeSchemaEntry {
  typeName: string
  description: string
  fields?: NLFieldSchema[]
}

export interface NLSchemaManifest {
  version: string
  signals: NLTypeSchemaEntry[]
  plugins: NLTypeSchemaEntry[]
  algorithms: NLTypeSchemaEntry[]
  backends: NLTypeSchemaEntry[]
}

export interface NLAuthoringCapabilities {
  enabled: boolean
  preview: boolean
  plannerAvailable: boolean
  plannerBackend: string
  schemaVersion: string
  supportedOperations: NLOperationMode[]
  supportedConstructs: NLConstructKind[]
  supportedSignalTypes: string[]
  supportedPluginTypes: string[]
  supportedBackendTypes: string[]
  supportedAlgorithmTypes: string[]
  supportsClarification: boolean
  supportsSessionApi: boolean
  supportsStreaming: boolean
  supportsApply: boolean
  readonlyMode: boolean
}

export interface NLSymbolInfo {
  name: string
  type: string
}

export interface NLSymbolSnapshot {
  signals?: NLSymbolInfo[]
  models?: string[]
  plugins?: string[]
  backends?: NLSymbolInfo[]
  routes?: string[]
}

export interface NLDiagnosticSnapshot {
  level: string
  message: string
  line?: number
  column?: number
}

export interface NLSessionContext {
  baseDsl?: string
  symbols?: NLSymbolSnapshot
  diagnostics?: NLDiagnosticSnapshot[]
}

export interface IntentIR {
  version: string
  operation: NLOperationMode
  intents: Intent[]
}

export type Intent =
  | SignalIntent
  | RouteIntent
  | PluginTemplateIntent
  | BackendIntent
  | GlobalIntent
  | ModifyIntent

export interface SignalIntent {
  type: 'signal'
  signal_type: string
  name: string
  fields: Record<string, unknown>
}

export interface RouteIntent {
  type: 'route'
  name: string
  description?: string
  priority?: number
  condition: ConditionNode
  models: ModelIntent[]
  algorithm?: AlgorithmIntent
  plugins?: PluginRefIntent[]
}

export type ConditionNode =
  | { op: 'AND'; operands: ConditionNode[] }
  | { op: 'OR'; operands: ConditionNode[] }
  | { op: 'NOT'; operand: ConditionNode }
  | { op: 'SIGNAL_REF'; signal_type: string; signal_name: string }

export interface ModelIntent {
  model: string
  reasoning?: boolean
  effort?: 'low' | 'medium' | 'high'
  lora?: string
  param_size?: string
  weight?: number
  reasoning_family?: string
}

export interface AlgorithmIntent {
  algo_type: string
  params: Record<string, unknown>
}

export interface PluginRefIntent {
  name: string
  overrides?: Record<string, unknown>
}

export interface PluginTemplateIntent {
  type: 'plugin_template'
  name: string
  plugin_type: string
  fields: Record<string, unknown>
}

export interface BackendIntent {
  type: 'backend'
  backend_type: string
  name: string
  fields: Record<string, unknown>
}

export interface GlobalIntent {
  type: 'global'
  fields: Record<string, unknown>
}

export interface ModifyIntent {
  type: 'modify'
  action: 'add' | 'update' | 'delete'
  target_construct: 'signal' | 'route' | 'plugin' | 'backend' | 'global'
  target_name: string
  target_signal_type?: string
  target_plugin_type?: string
  target_backend_type?: string
  changes?: Record<string, unknown>
}

export interface NLClarificationOption {
  id: string
  label: string
  description?: string
}

export interface NLClarification {
  question: string
  options: NLClarificationOption[]
}

export interface NLPlannerWarning {
  code: string
  message: string
}

export interface NLPlannerResult {
  status: NLPlannerResultStatus
  intentIr?: IntentIR
  explanation?: string
  clarification?: NLClarification
  warnings?: NLPlannerWarning[]
  error?: string
}

export interface NLSessionCreateRequest {
  schemaVersion?: string
  context?: NLSessionContext
}

export interface NLSessionCreateResponse {
  sessionId: string
  schemaVersion: string
  expiresAt?: string
  capabilities: NLAuthoringCapabilities
}

export interface NLTurnRequest {
  prompt: string
  modeHint?: NLOperationMode
  schemaVersion?: string
  context?: NLSessionContext
}

export interface NLTurnResponse {
  sessionId: string
  turnId: string
  schemaVersion: string
  expiresAt?: string
  result: NLPlannerResult
}
