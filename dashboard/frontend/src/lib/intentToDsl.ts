/**
 * Intent IR → DSL Code Generator
 *
 * Pure deterministic functions that convert an IntentIR JSON structure into
 * syntactically valid DSL text. No LLM calls — all output is guaranteed to
 * conform to the parser grammar defined in ast.go / parser.go.
 *
 * Key DSL syntax rules (from parser.go):
 *   SIGNAL <type> <name> { key: value }
 *   PLUGIN <name> <type> { key: value }      ← name before type!
 *   BACKEND <type> <name> { key: value }     ← type before name!
 *   ROUTE <name> (description = "...") { PRIORITY n  WHEN expr  MODEL "m" (attrs)  ALGORITHM t {} PLUGIN ref }
 *   GLOBAL { key: value }
 *
 * Boolean expression precedence: NOT > AND > OR
 * Values: "string" | 123 | 4.5 | true | false | [a, b] | { k: v }
 */

import type {
  IntentIR,
  Intent,
  SignalIntent,
  RouteIntent,
  PluginTemplateIntent,
  BackendIntent,
  GlobalIntent,
  ModifyIntent,
  ConditionNode,
  ModelIntent,
} from '../types/intentIR'

import {
  addSignal,
  addPlugin,
  addBackend,
  addRoute,
  updateSignal,
  updatePlugin,
  updateBackend,
  updateGlobal,
  deleteSignal,
  deletePlugin,
  deleteBackend,
  deleteRoute,
  type RouteInput,
} from './dslMutations'

// ─────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────

/**
 * Convert an IntentIR to DSL text.
 * For "modify" operations with an existing DSL, applies incremental edits.
 * For "generate"/"fix" operations, produces a full DSL document.
 */
export function intentIRToDSL(ir: IntentIR, existingDSL?: string): string {
  if (ir.operation === 'modify' && existingDSL) {
    return applyModifications(ir, existingDSL)
  }
  return generateFullDSL(ir)
}

// ─────────────────────────────────────────────
// Full DSL Generation
// ─────────────────────────────────────────────

function generateFullDSL(ir: IntentIR): string {
  const sections: string[] = []

  const signals = ir.intents.filter((i): i is SignalIntent => i.type === 'signal')
  const templates = ir.intents.filter((i): i is PluginTemplateIntent => i.type === 'plugin_template')
  const routes = ir.intents.filter((i): i is RouteIntent => i.type === 'route')
  const backends = ir.intents.filter((i): i is BackendIntent => i.type === 'backend')
  const globals = ir.intents.filter((i): i is GlobalIntent => i.type === 'global')

  if (signals.length > 0) {
    sections.push('# SIGNALS\n')
    for (const sig of signals) {
      sections.push(emitSignal(sig))
    }
  }

  if (templates.length > 0) {
    sections.push('# PLUGINS\n')
    for (const tmpl of templates) {
      sections.push(emitPluginTemplate(tmpl))
    }
  }

  if (routes.length > 0) {
    sections.push('# ROUTES\n')
    for (const route of routes) {
      sections.push(emitRoute(route))
    }
  }

  if (backends.length > 0) {
    sections.push('# BACKENDS\n')
    for (const be of backends) {
      sections.push(emitBackend(be))
    }
  }

  if (globals.length > 0) {
    sections.push('# GLOBAL\n')
    for (const g of globals) {
      sections.push(emitGlobal(g))
    }
  }

  return sections.join('\n')
}

// ─────────────────────────────────────────────
// Emitters — produce individual DSL blocks
// ─────────────────────────────────────────────

function emitSignal(sig: SignalIntent): string {
  const name = sanitizeName(sig.name)
  const lines = [`SIGNAL ${sig.signal_type} ${name} {`]
  for (const [key, value] of Object.entries(sig.fields)) {
    if (value === undefined || value === null) continue
    lines.push(`  ${key}: ${serializeValue(value, '  ')}`)
  }
  lines.push('}\n')
  return lines.join('\n')
}

function emitPluginTemplate(tmpl: PluginTemplateIntent): string {
  // PLUGIN <name> <type> { ... }  ← name before type
  const name = sanitizeName(tmpl.name)
  const lines = [`PLUGIN ${name} ${tmpl.plugin_type} {`]
  for (const [key, value] of Object.entries(tmpl.fields)) {
    if (value === undefined || value === null) continue
    lines.push(`  ${key}: ${serializeValue(value, '  ')}`)
  }
  lines.push('}\n')
  return lines.join('\n')
}

function emitRoute(route: RouteIntent): string {
  const name = sanitizeName(route.name)
  const descPart = route.description
    ? ` (description = "${escapeString(route.description)}")`
    : ''
  const lines = [`ROUTE ${name}${descPart} {`]

  // PRIORITY
  if (route.priority !== undefined) {
    lines.push(`  PRIORITY ${route.priority}`)
    lines.push('')
  }

  // WHEN
  if (route.condition) {
    const condStr = emitCondition(route.condition)
    if (condStr) {
      lines.push(`  WHEN ${condStr}`)
      lines.push('')
    }
  }

  // MODEL(s)
  if (route.models && route.models.length > 0) {
    const modelParts = route.models.map(emitModelRef)
    if (modelParts.length === 1) {
      lines.push(`  MODEL ${modelParts[0]}`)
    } else {
      lines.push(`  MODEL ${modelParts.join(',\n        ')}`)
    }
    lines.push('')
  }

  // ALGORITHM
  if (route.algorithm) {
    const params = Object.entries(route.algorithm.params).filter(
      ([, v]) => v !== undefined && v !== null
    )
    if (params.length > 0) {
      lines.push(`  ALGORITHM ${route.algorithm.algo_type} {`)
      for (const [k, v] of params) {
        lines.push(`    ${k}: ${serializeValue(v, '    ')}`)
      }
      lines.push('  }')
    } else {
      lines.push(`  ALGORITHM ${route.algorithm.algo_type} {}`)
    }
    lines.push('')
  }

  // PLUGIN references
  if (route.plugins && route.plugins.length > 0) {
    for (const plugin of route.plugins) {
      const pName = sanitizeName(plugin.name)
      if (plugin.overrides && Object.keys(plugin.overrides).length > 0) {
        lines.push(`  PLUGIN ${pName} {`)
        for (const [k, v] of Object.entries(plugin.overrides)) {
          if (v === undefined || v === null) continue
          lines.push(`    ${k}: ${serializeValue(v, '    ')}`)
        }
        lines.push('  }')
      } else {
        lines.push(`  PLUGIN ${pName}`)
      }
    }
  }

  lines.push('}\n')
  return lines.join('\n')
}

function emitBackend(be: BackendIntent): string {
  // BACKEND <type> <name> { ... }  ← type before name
  const name = sanitizeName(be.name)
  const lines = [`BACKEND ${be.backend_type} ${name} {`]
  for (const [key, value] of Object.entries(be.fields)) {
    if (value === undefined || value === null) continue
    lines.push(`  ${key}: ${serializeValue(value, '  ')}`)
  }
  lines.push('}\n')
  return lines.join('\n')
}

function emitGlobal(g: GlobalIntent): string {
  const lines = ['GLOBAL {']
  for (const [key, value] of Object.entries(g.fields)) {
    if (value === undefined || value === null) continue
    lines.push(`  ${key}: ${serializeValue(value, '  ')}`)
  }
  lines.push('}\n')
  return lines.join('\n')
}

// ─────────────────────────────────────────────
// Boolean Expression Emitter
// ─────────────────────────────────────────────

/**
 * Emit a ConditionNode tree to DSL WHEN clause text.
 *
 * Parenthesization rules (matching parser.go precedence):
 *   - NOT binds tightest (prefix)
 *   - AND binds medium
 *   - OR binds loosest
 * Therefore: OR inside AND needs parens, AND inside OR does not.
 */
export function emitCondition(node: ConditionNode): string {
  switch (node.op) {
    case 'SIGNAL_REF':
      return `${node.signal_type}("${node.signal_name}")`

    case 'NOT':
      return `NOT ${emitConditionWrapped(node.operand, 'NOT')}`

    case 'AND':
      return node.operands
        .map(o => emitConditionWrapped(o, 'AND'))
        .join(' AND ')

    case 'OR':
      return node.operands
        .map(o => emitConditionWrapped(o, 'OR'))
        .join(' OR ')
  }
}

/**
 * Wrap a child expression in parens if needed based on parent operator.
 */
function emitConditionWrapped(child: ConditionNode, parentOp: string): string {
  const inner = emitCondition(child)

  // OR inside AND → needs parens
  if (parentOp === 'AND' && child.op === 'OR') {
    return `(${inner})`
  }
  // Any compound expression after NOT → needs parens
  if (parentOp === 'NOT' && (child.op === 'AND' || child.op === 'OR')) {
    return `(${inner})`
  }

  return inner
}

// ─────────────────────────────────────────────
// Model Reference Emitter
// ─────────────────────────────────────────────

function emitModelRef(model: ModelIntent): string {
  const attrs: string[] = []
  if (model.reasoning !== undefined) attrs.push(`reasoning = ${model.reasoning}`)
  if (model.effort) attrs.push(`effort = "${model.effort}"`)
  if (model.lora) attrs.push(`lora = "${model.lora}"`)
  if (model.param_size) attrs.push(`param_size = "${model.param_size}"`)
  if (model.weight !== undefined) attrs.push(`weight = ${model.weight}`)
  if (model.reasoning_family) attrs.push(`reasoning_family = "${model.reasoning_family}"`)

  const attrStr = attrs.length > 0 ? ` (${attrs.join(', ')})` : ''
  return `"${escapeString(model.model)}"${attrStr}`
}

// ─────────────────────────────────────────────
// Value Serialization (mirrors dslMutations.ts)
// ─────────────────────────────────────────────

function serializeValue(v: unknown, currentIndent = '  '): string {
  if (typeof v === 'string') return `"${escapeString(v)}"`
  if (typeof v === 'number') return String(v)
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (Array.isArray(v)) {
    if (v.length === 0) return '[]'
    const simple = v.every(
      item => typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean'
    )
    if (simple) {
      return `[${v.map(item => serializeValue(item, currentIndent)).join(', ')}]`
    }
    const childIndent = currentIndent + '  '
    const items = v.map(item => `${childIndent}${serializeValue(item, childIndent)}`).join(',\n')
    return `[\n${items}\n${currentIndent}]`
  }
  if (typeof v === 'object' && v !== null) {
    const obj = v as Record<string, unknown>
    const entries = Object.entries(obj).filter(([, val]) => val !== undefined && val !== null)
    if (entries.length === 0) return '{}'
    // Inline small flat objects (≤3 entries, all primitive)
    const allPrimitive = entries.every(([, val]) => typeof val !== 'object' || val === null)
    if (allPrimitive && entries.length <= 3) {
      const parts = entries.map(([k, val]) => `${k}: ${serializeValue(val, currentIndent)}`)
      return `{ ${parts.join(', ')} }`
    }
    const childIndent = currentIndent + '  '
    const inner = entries
      .map(([k, val]) => `${childIndent}${k}: ${serializeValue(val, childIndent)}`)
      .join('\n')
    return `{\n${inner}\n${currentIndent}}`
  }
  return String(v)
}

// ─────────────────────────────────────────────
// Name Sanitization
// ─────────────────────────────────────────────

/**
 * Sanitize a name to be a valid DSL identifier.
 * Allows [a-zA-Z0-9_\-\.\/] per parser.go Ident rule.
 * Non-matching chars are replaced with '_'.
 */
export function sanitizeName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_\-]/g, '_')
}

function escapeString(s: string): string {
  return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"')
}

// ─────────────────────────────────────────────
// Modification Mode — Incremental Edits
// ─────────────────────────────────────────────

function applyModifications(ir: IntentIR, existingDSL: string): string {
  let dsl = existingDSL

  // 1. Apply ModifyIntent entries first
  const modifyIntents = ir.intents.filter((i): i is ModifyIntent => i.type === 'modify')
  for (const mod of modifyIntents) {
    switch (mod.action) {
      case 'add':
        dsl = applyAdd(dsl, mod)
        break
      case 'update':
        dsl = applyUpdate(dsl, mod)
        break
      case 'delete':
        dsl = applyDelete(dsl, mod)
        break
    }
  }

  // 2. Handle new signal/plugin/backend/route/global intents alongside modifications
  const newSignals = ir.intents.filter((i): i is SignalIntent => i.type === 'signal')
  for (const sig of newSignals) {
    dsl = addSignal(dsl, sig.signal_type, sanitizeName(sig.name), sig.fields)
  }

  const newTemplates = ir.intents.filter((i): i is PluginTemplateIntent => i.type === 'plugin_template')
  for (const tmpl of newTemplates) {
    dsl = addPlugin(dsl, sanitizeName(tmpl.name), tmpl.plugin_type, tmpl.fields)
  }

  const newRoutes = ir.intents.filter((i): i is RouteIntent => i.type === 'route')
  for (const route of newRoutes) {
    const routeInput: RouteInput = {
      description: route.description,
      priority: route.priority ?? 10,
      when: route.condition ? emitCondition(route.condition) : undefined,
      models: route.models.map(m => ({
        model: m.model,
        reasoning: m.reasoning,
        effort: m.effort,
        lora: m.lora,
        paramSize: m.param_size,
        weight: m.weight,
        reasoningFamily: m.reasoning_family,
      })),
      algorithm: route.algorithm
        ? { algoType: route.algorithm.algo_type, fields: route.algorithm.params }
        : undefined,
      plugins: (route.plugins ?? []).map(p => ({
        name: sanitizeName(p.name),
        fields: p.overrides,
      })),
    }
    dsl = addRoute(dsl, sanitizeName(route.name), routeInput)
  }

  const newBackends = ir.intents.filter((i): i is BackendIntent => i.type === 'backend')
  for (const be of newBackends) {
    dsl = addBackend(dsl, be.backend_type, sanitizeName(be.name), be.fields)
  }

  const newGlobals = ir.intents.filter((i): i is GlobalIntent => i.type === 'global')
  for (const g of newGlobals) {
    dsl = updateGlobal(dsl, g.fields)
  }

  return dsl
}

function applyAdd(dsl: string, mod: ModifyIntent): string {
  const fields = (mod.changes as Record<string, unknown>) ?? {}
  switch (mod.target_construct) {
    case 'signal':
      return addSignal(
        dsl,
        mod.target_signal_type ?? 'keyword',
        sanitizeName(mod.target_name),
        fields,
      )
    case 'plugin':
      return addPlugin(
        dsl,
        sanitizeName(mod.target_name),
        mod.target_plugin_type ?? 'system_prompt',
        fields,
      )
    case 'backend':
      return addBackend(
        dsl,
        mod.target_backend_type ?? 'vllm_endpoint',
        sanitizeName(mod.target_name),
        fields,
      )
    case 'route': {
      const routeInput: RouteInput = {
        priority: (fields.priority as number) ?? 10,
        models: (fields.models as RouteInput['models']) ?? [],
        plugins: (fields.plugins as RouteInput['plugins']) ?? [],
        description: fields.description as string | undefined,
        when: fields.when as string | undefined,
        algorithm: fields.algorithm as RouteInput['algorithm'],
      }
      return addRoute(dsl, sanitizeName(mod.target_name), routeInput)
    }
    default:
      return dsl
  }
}

function applyUpdate(dsl: string, mod: ModifyIntent): string {
  const fields = (mod.changes as Record<string, unknown>) ?? {}
  switch (mod.target_construct) {
    case 'signal':
      return updateSignal(
        dsl,
        mod.target_signal_type ?? 'keyword',
        sanitizeName(mod.target_name),
        fields,
      )
    case 'plugin':
      return updatePlugin(
        dsl,
        sanitizeName(mod.target_name),
        mod.target_plugin_type ?? 'system_prompt',
        fields,
      )
    case 'backend':
      return updateBackend(
        dsl,
        mod.target_backend_type ?? 'vllm_endpoint',
        sanitizeName(mod.target_name),
        fields,
      )
    case 'global':
      return updateGlobal(dsl, fields)
    default:
      return dsl
  }
}

function applyDelete(dsl: string, mod: ModifyIntent): string {
  switch (mod.target_construct) {
    case 'signal':
      return deleteSignal(dsl, mod.target_signal_type ?? '', mod.target_name)
    case 'route':
      return deleteRoute(dsl, mod.target_name)
    case 'plugin':
      return deletePlugin(dsl, mod.target_name, mod.target_plugin_type ?? '')
    case 'backend':
      return deleteBackend(dsl, mod.target_backend_type ?? '', mod.target_name)
    default:
      return dsl
  }
}

// ─────────────────────────────────────────────
// Dependency Resolution
// ─────────────────────────────────────────────

/** Algorithm → required signal types */
const ALGO_SIGNAL_DEPS: Record<string, Array<{ type: string; name: string }>> = {
  rl_driven: [{ type: 'user_feedback', name: 'feedback' }],
  ratings: [{ type: 'user_feedback', name: 'feedback' }],
  elo: [{ type: 'user_feedback', name: 'feedback' }],
  gmtrouter: [{ type: 'user_feedback', name: 'feedback' }],
  hybrid: [{ type: 'user_feedback', name: 'feedback' }],
}

/**
 * Collect all SIGNAL_REF nodes from a condition tree.
 */
export function collectSignalRefs(node: ConditionNode): Array<{ signal_type: string; signal_name: string }> {
  const refs: Array<{ signal_type: string; signal_name: string }> = []

  function walk(n: ConditionNode) {
    switch (n.op) {
      case 'SIGNAL_REF':
        refs.push({ signal_type: n.signal_type, signal_name: n.signal_name })
        break
      case 'AND':
      case 'OR':
        for (const child of n.operands) walk(child)
        break
      case 'NOT':
        walk(n.operand)
        break
    }
  }

  walk(node)
  return refs
}

/**
 * Auto-create missing signal declarations referenced in route conditions
 * and algorithm dependencies.
 */
export function resolveImplicitDependencies(ir: IntentIR): IntentIR {
  const definedSignals = new Set<string>()

  for (const intent of ir.intents) {
    if (intent.type === 'signal') {
      definedSignals.add(`${intent.signal_type}:${intent.name}`)
    }
  }

  const missingSignals: SignalIntent[] = []

  // Check route conditions for signal references
  for (const intent of ir.intents) {
    if (intent.type === 'route') {
      const refs = collectSignalRefs(intent.condition)
      for (const ref of refs) {
        const key = `${ref.signal_type}:${ref.signal_name}`
        if (!definedSignals.has(key)) {
          missingSignals.push(createDefaultSignal(ref.signal_type, ref.signal_name))
          definedSignals.add(key)
        }
      }
    }
  }

  // Check algorithm dependencies
  for (const intent of ir.intents) {
    if (intent.type === 'route' && intent.algorithm) {
      const deps = ALGO_SIGNAL_DEPS[intent.algorithm.algo_type] ?? []
      for (const dep of deps) {
        const key = `${dep.type}:${dep.name}`
        if (!definedSignals.has(key)) {
          missingSignals.push(createDefaultSignal(dep.type, dep.name))
          definedSignals.add(key)
        }
      }
    }
  }

  if (missingSignals.length === 0) return ir

  return {
    ...ir,
    intents: [...missingSignals, ...ir.intents],
  }
}

/**
 * Create a default signal with minimal required fields.
 */
function createDefaultSignal(signalType: string, name: string): SignalIntent {
  const defaults: Record<string, Record<string, unknown>> = {
    keyword: { operator: 'any', keywords: [name], method: 'regex', case_sensitive: false },
    embedding: { threshold: 0.75, candidates: [name], aggregation_method: 'max' },
    domain: { description: name },
    fact_check: { description: name },
    user_feedback: { description: name },
    preference: { description: name },
    language: { description: name },
    context: { min_tokens: '1K', max_tokens: '128K' },
    complexity: { threshold: 0.5 },
    modality: { description: name },
    authz: { role: name, subjects: [{ kind: 'Group', name }] },
    jailbreak: { method: 'classifier', threshold: 0.9 },
    pii: { threshold: 0.8, pii_types_allowed: [] },
  }

  return {
    type: 'signal',
    signal_type: signalType as SignalIntent['signal_type'],
    name,
    fields: defaults[signalType] ?? { description: name },
  }
}
