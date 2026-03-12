/**
 * NL Partial Accept Module
 *
 * Provides functionality for users to selectively accept parts of a generated
 * IntentIR, instead of all-or-nothing acceptance.
 *
 * Key capabilities:
 *   1. Dependency analysis - understand which intents depend on others
 *   2. Selection management - track which intents user wants to accept
 *   3. Partial IR generation - build a new IR with only selected intents
 *   4. Dependency issue detection - warn when selection breaks dependencies
 *   5. Intent editing - allow inline modifications before acceptance
 */

import type {
  IntentIR,
  Intent,
} from '@/types/intentIR'
import type { SymbolTable, WasmBridge } from '@/types/dsl'
import { collectSignalRefs, resolveImplicitDependencies, intentIRToDSL } from './intentToDsl'
import { validateGeneratedDSL } from './nlValidation'
import type { ValidationResult } from './nlValidation'

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/** Selection state for a single intent */
export interface IntentSelectionState {
  /** Intent index in the original IntentIR.intents array */
  index: number
  /** Whether this intent is selected for acceptance */
  selected: boolean
  /** Whether the user has edited this intent */
  edited: boolean
  /** Edited intent data (if modified by user) */
  editedIntent?: Intent
  /** Per-intent validation errors */
  validationErrors?: string[]
}

/** Dependency relationship types */
export type DependencyType =
  | 'signal_ref'      // Route WHEN clause references a signal
  | 'plugin_ref'      // Route uses a plugin template
  | 'backend_ref'     // Plugin references a backend
  | 'algorithm_dep'   // Algorithm requires a signal type

/** A single dependency edge in the graph */
export interface DependencyEdge {
  /** Source intent index (the one that has the dependency) */
  from: number
  /** Target intent index (the one being depended upon) */
  to: number
  /** Type of dependency */
  type: DependencyType
  /** Name of the reference (for display) */
  refName: string
}

/** Node in the dependency graph */
export interface DependencyNode {
  index: number
  intent: Intent
  /** Indices of intents this one depends on */
  dependsOn: number[]
  /** Indices of intents that depend on this one */
  dependedBy: number[]
}

/** Complete dependency graph for an IntentIR */
export interface IntentDependencyGraph {
  nodes: DependencyNode[]
  edges: DependencyEdge[]
}

/** A detected dependency issue when selection is incomplete */
export interface DependencyIssue {
  /** Type of missing reference */
  type: 'missing_signal' | 'missing_plugin' | 'missing_backend'
  /** Index of the intent that has the broken dependency */
  sourceIndex: number
  /** Index of the missing dependency (in the IR) or -1 if external */
  missingIndex: number
  /** Name of the missing reference */
  missingRef: string
  /** Human-readable suggestion */
  suggestion: string
}

/** Complete state for partial accept UI */
export interface PartialAcceptState {
  /** The original IntentIR from LLM */
  originalIR: IntentIR
  /** Selection state for each intent */
  selections: IntentSelectionState[]
  /** Analyzed dependency graph */
  dependencyGraph: IntentDependencyGraph
  /** Preview DSL based on current selection */
  previewDSL: string
  /** Validation result for the preview DSL */
  previewValidation: ValidationResult | null
  /** Detected dependency issues */
  dependencyIssues: DependencyIssue[]
  /** Whether there are unresolved dependency issues */
  hasDependencyIssues: boolean
}

// ─────────────────────────────────────────────
// Dependency Analysis
// ─────────────────────────────────────────────

/**
 * Analyze the dependency relationships within an IntentIR.
 * Builds a directed graph where edges point from dependent → dependency.
 */
export function analyzeIntentDependencies(ir: IntentIR): IntentDependencyGraph {
  const nodes: DependencyNode[] = ir.intents.map((intent, index) => ({
    index,
    intent,
    dependsOn: [],
    dependedBy: [],
  }))
  const edges: DependencyEdge[] = []

  // Build name → index maps for each construct type
  const signalMap = new Map<string, number>() // "signal_type:name" → index
  const pluginMap = new Map<string, number>() // "name" → index
  const backendMap = new Map<string, number>() // "name" → index

  ir.intents.forEach((intent, index) => {
    if (intent.type === 'signal') {
      signalMap.set(`${intent.signal_type}:${intent.name}`, index)
    } else if (intent.type === 'plugin_template') {
      pluginMap.set(intent.name, index)
    } else if (intent.type === 'backend') {
      backendMap.set(intent.name, index)
    }
  })

  // Analyze each intent's dependencies
  ir.intents.forEach((intent, index) => {
    // Routes depend on signals (via WHEN condition) and plugins
    if (intent.type === 'route') {
      // Signal dependencies from WHEN clause
      if (intent.condition) {
        const signalRefs = collectSignalRefs(intent.condition)
        for (const ref of signalRefs) {
          const depKey = `${ref.signal_type}:${ref.signal_name}`
          const depIndex = signalMap.get(depKey)
          if (depIndex !== undefined) {
            nodes[index].dependsOn.push(depIndex)
            nodes[depIndex].dependedBy.push(index)
            edges.push({
              from: index,
              to: depIndex,
              type: 'signal_ref',
              refName: `${ref.signal_type}("${ref.signal_name}")`,
            })
          }
        }
      }

      // Plugin dependencies
      if (intent.plugins) {
        for (const pluginRef of intent.plugins) {
          const depIndex = pluginMap.get(pluginRef.name)
          if (depIndex !== undefined) {
            nodes[index].dependsOn.push(depIndex)
            nodes[depIndex].dependedBy.push(index)
            edges.push({
              from: index,
              to: depIndex,
              type: 'plugin_ref',
              refName: pluginRef.name,
            })
          }
        }
      }

      // Algorithm signal dependencies (e.g., rl_driven needs user_feedback)
      if (intent.algorithm) {
        const algoDeps = getAlgorithmSignalDeps(intent.algorithm.algo_type)
        for (const dep of algoDeps) {
          const depKey = `${dep.type}:${dep.name}`
          const depIndex = signalMap.get(depKey)
          if (depIndex !== undefined) {
            nodes[index].dependsOn.push(depIndex)
            nodes[depIndex].dependedBy.push(index)
            edges.push({
              from: index,
              to: depIndex,
              type: 'algorithm_dep',
              refName: `${dep.type}("${dep.name}")`,
            })
          }
        }
      }
    }

    // Plugins may depend on backends (e.g., RAG plugin)
    if (intent.type === 'plugin_template') {
      const backendRef = intent.fields.backend_ref as string | undefined
      if (backendRef) {
        const depIndex = backendMap.get(backendRef)
        if (depIndex !== undefined) {
          nodes[index].dependsOn.push(depIndex)
          nodes[depIndex].dependedBy.push(index)
          edges.push({
            from: index,
            to: depIndex,
            type: 'backend_ref',
            refName: backendRef,
          })
        }
      }
    }
  })

  return { nodes, edges }
}

/** Algorithm → required signal types mapping */
const ALGO_SIGNAL_DEPS: Record<string, Array<{ type: string; name: string }>> = {
  rl_driven: [{ type: 'user_feedback', name: 'feedback' }],
  ratings: [{ type: 'user_feedback', name: 'feedback' }],
  elo: [{ type: 'user_feedback', name: 'feedback' }],
  gmtrouter: [{ type: 'user_feedback', name: 'feedback' }],
  hybrid: [{ type: 'user_feedback', name: 'feedback' }],
}

function getAlgorithmSignalDeps(algoType: string): Array<{ type: string; name: string }> {
  return ALGO_SIGNAL_DEPS[algoType] ?? []
}

// ─────────────────────────────────────────────
// Dependency Issue Detection
// ─────────────────────────────────────────────

/**
 * Detect dependency issues when certain intents are not selected.
 *
 * @param ir - Original IntentIR
 * @param selections - Current selection states
 * @param graph - Pre-computed dependency graph
 * @param existingSymbols - Symbols already in the DSL (can satisfy dependencies)
 */
export function detectDependencyIssues(
  ir: IntentIR,
  selections: IntentSelectionState[],
  graph: IntentDependencyGraph,
  existingSymbols?: SymbolTable,
): DependencyIssue[] {
  const issues: DependencyIssue[] = []
  const selectedIndices = new Set(
    selections.filter(s => s.selected).map(s => s.index)
  )

  // Check each selected intent's dependencies
  for (const selection of selections) {
    if (!selection.selected) continue

    const node = graph.nodes[selection.index]

    for (const depIndex of node.dependsOn) {
      if (selectedIndices.has(depIndex)) continue // Dependency is selected, OK

      // Check if dependency exists in current DSL
      const depIntent = ir.intents[depIndex]
      if (existsInSymbols(depIntent, existingSymbols)) continue // Already in DSL, OK

      // Found a missing dependency
      const edge = graph.edges.find(e => e.from === selection.index && e.to === depIndex)
      issues.push({
        type: getMissingType(depIntent),
        sourceIndex: selection.index,
        missingIndex: depIndex,
        missingRef: edge?.refName ?? getIntentDisplayName(depIntent),
        suggestion: `Select "${getIntentDisplayName(depIntent)}" or ensure it exists in current DSL`,
      })
    }
  }

  return issues
}

/** Check if an intent's entity already exists in the symbol table */
function existsInSymbols(intent: Intent, symbols?: SymbolTable): boolean {
  if (!symbols) return false

  switch (intent.type) {
    case 'signal':
      return (symbols.signals ?? []).some(
        s => s.type === intent.signal_type && s.name === intent.name
      )
    case 'plugin_template':
      return (symbols.plugins ?? []).includes(intent.name)
    case 'backend':
      return (symbols.backends ?? []).some(
        b => b.type === intent.backend_type && b.name === intent.name
      )
    case 'route':
      return (symbols.routes ?? []).includes(intent.name)
    default:
      return false
  }
}

function getMissingType(intent: Intent): 'missing_signal' | 'missing_plugin' | 'missing_backend' {
  switch (intent.type) {
    case 'signal': return 'missing_signal'
    case 'plugin_template': return 'missing_plugin'
    case 'backend': return 'missing_backend'
    default: return 'missing_signal'
  }
}

// ─────────────────────────────────────────────
// Selection State Management
// ─────────────────────────────────────────────

/**
 * Create initial selection state for an IntentIR.
 * By default, all intents are selected.
 */
export function createInitialSelections(ir: IntentIR): IntentSelectionState[] {
  return ir.intents.map((_, index) => ({
    index,
    selected: true,
    edited: false,
  }))
}

/**
 * Toggle selection for a single intent.
 */
export function toggleSelection(
  selections: IntentSelectionState[],
  index: number,
): IntentSelectionState[] {
  return selections.map(s =>
    s.index === index ? { ...s, selected: !s.selected } : s
  )
}

/**
 * Set all selections to a specific value.
 */
export function setAllSelections(
  selections: IntentSelectionState[],
  selected: boolean,
): IntentSelectionState[] {
  return selections.map(s => ({ ...s, selected }))
}

/**
 * Auto-select dependencies for all currently selected intents.
 * Returns updated selections with required dependencies also selected.
 */
export function autoSelectDependencies(
  selections: IntentSelectionState[],
  graph: IntentDependencyGraph,
): IntentSelectionState[] {
  const newSelections = [...selections]
  let changed = true

  // Iterate until no more changes (transitive closure)
  while (changed) {
    changed = false
    for (const selection of newSelections) {
      if (!selection.selected) continue

      const node = graph.nodes[selection.index]
      for (const depIndex of node.dependsOn) {
        if (!newSelections[depIndex].selected) {
          newSelections[depIndex] = { ...newSelections[depIndex], selected: true }
          changed = true
        }
      }
    }
  }

  return newSelections
}

/**
 * Update an intent's fields (for inline editing).
 */
export function updateIntentEdit(
  selections: IntentSelectionState[],
  index: number,
  editedIntent: Intent,
): IntentSelectionState[] {
  return selections.map(s =>
    s.index === index
      ? { ...s, edited: true, editedIntent }
      : s
  )
}

/**
 * Revert an intent to its original state (discard edits).
 */
export function revertIntentEdit(
  selections: IntentSelectionState[],
  index: number,
): IntentSelectionState[] {
  return selections.map(s =>
    s.index === index
      ? { ...s, edited: false, editedIntent: undefined }
      : s
  )
}

// ─────────────────────────────────────────────
// Partial IR Generation
// ─────────────────────────────────────────────

/**
 * Build a new IntentIR containing only selected intents.
 * Uses edited versions where available.
 */
export function buildPartialIntentIR(
  originalIR: IntentIR,
  selections: IntentSelectionState[],
): IntentIR {
  const selectedIntents: Intent[] = []

  for (const selection of selections) {
    if (selection.selected) {
      const intent = selection.edited && selection.editedIntent
        ? selection.editedIntent
        : originalIR.intents[selection.index]
      selectedIntents.push(intent)
    }
  }

  return {
    version: '1.0',
    operation: originalIR.operation,
    intents: selectedIntents,
  }
}

/**
 * Generate preview DSL and validation for current selection.
 */
export async function generatePartialPreview(
  originalIR: IntentIR,
  selections: IntentSelectionState[],
  existingDSL: string | undefined,
  wasm: WasmBridge,
): Promise<{ dsl: string; validation: ValidationResult }> {
  // Build partial IR
  const partialIR = buildPartialIntentIR(originalIR, selections)

  // Resolve any implicit dependencies that are still needed
  const resolvedIR = resolveImplicitDependencies(partialIR)

  // Convert to DSL
  const dsl = intentIRToDSL(resolvedIR, existingDSL)

  // Validate
  const validation = validateGeneratedDSL(dsl, wasm)

  return { dsl, validation }
}

// ─────────────────────────────────────────────
// State Factory
// ─────────────────────────────────────────────

/**
 * Create the complete partial accept state from an IntentIR.
 */
export function createPartialAcceptState(
  ir: IntentIR,
  _existingDSL?: string,
  existingSymbols?: SymbolTable,
): PartialAcceptState {
  const selections = createInitialSelections(ir)
  const dependencyGraph = analyzeIntentDependencies(ir)
  const dependencyIssues = detectDependencyIssues(ir, selections, dependencyGraph, existingSymbols)

  return {
    originalIR: ir,
    selections,
    dependencyGraph,
    previewDSL: '',
    previewValidation: null,
    dependencyIssues,
    hasDependencyIssues: dependencyIssues.length > 0,
  }
}

/**
 * Update partial accept state after selection changes.
 */
export function updatePartialAcceptState(
  state: PartialAcceptState,
  newSelections: IntentSelectionState[],
  existingSymbols?: SymbolTable,
): PartialAcceptState {
  const dependencyIssues = detectDependencyIssues(
    state.originalIR,
    newSelections,
    state.dependencyGraph,
    existingSymbols,
  )

  return {
    ...state,
    selections: newSelections,
    dependencyIssues,
    hasDependencyIssues: dependencyIssues.length > 0,
  }
}

// ─────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────

/**
 * Get a display-friendly name for an intent.
 */
export function getIntentDisplayName(intent: Intent): string {
  switch (intent.type) {
    case 'signal':
      return `${intent.signal_type}("${intent.name}")`
    case 'route':
      return `route "${intent.name}"`
    case 'plugin_template':
      return `plugin "${intent.name}" (${intent.plugin_type})`
    case 'backend':
      return `backend "${intent.name}" (${intent.backend_type})`
    case 'global':
      return 'GLOBAL settings'
    case 'modify':
      return `${intent.action} ${intent.target_construct} "${intent.target_name}"`
    default:
      return 'unknown intent'
  }
}

/**
 * Get the construct type of an intent for grouping.
 */
export function getIntentCategory(intent: Intent): string {
  switch (intent.type) {
    case 'signal': return 'signals'
    case 'route': return 'routes'
    case 'plugin_template': return 'plugins'
    case 'backend': return 'backends'
    case 'global': return 'global'
    case 'modify': return 'modifications'
    default: return 'other'
  }
}

/**
 * Group intents by category for UI display.
 */
export function groupIntentsByCategory(
  ir: IntentIR,
  selections: IntentSelectionState[],
): Record<string, Array<{ intent: Intent; selection: IntentSelectionState }>> {
  const groups: Record<string, Array<{ intent: Intent; selection: IntentSelectionState }>> = {}

  for (let i = 0; i < ir.intents.length; i++) {
    const intent = ir.intents[i]
    const selection = selections[i]
    const category = getIntentCategory(intent)

    if (!groups[category]) {
      groups[category] = []
    }
    groups[category].push({ intent, selection })
  }

  return groups
}

/**
 * Get selection statistics.
 */
export function getSelectionStats(selections: IntentSelectionState[]): {
  total: number
  selected: number
  edited: number
} {
  return {
    total: selections.length,
    selected: selections.filter(s => s.selected).length,
    edited: selections.filter(s => s.edited).length,
  }
}

/**
 * Check if a specific intent type is selected.
 */
export function hasSelectedIntentOfType(
  ir: IntentIR,
  selections: IntentSelectionState[],
  type: Intent['type'],
): boolean {
  return selections.some(
    s => s.selected && ir.intents[s.index].type === type
  )
}

// ─────────────────────────────────────────────
// Batch Operations
// ─────────────────────────────────────────────

/**
 * Select all intents of a specific category.
 */
export function selectByCategory(
  ir: IntentIR,
  selections: IntentSelectionState[],
  category: string,
  selected: boolean,
): IntentSelectionState[] {
  return selections.map((s, i) => {
    const intent = ir.intents[i]
    if (getIntentCategory(intent) === category) {
      return { ...s, selected }
    }
    return s
  })
}

/**
 * Select only intents that pass a filter predicate.
 */
export function selectByFilter(
  ir: IntentIR,
  selections: IntentSelectionState[],
  filter: (intent: Intent) => boolean,
): IntentSelectionState[] {
  return selections.map((s, i) => ({
    ...s,
    selected: filter(ir.intents[i]),
  }))
}

/**
 * Invert all selections.
 */
export function invertSelections(
  selections: IntentSelectionState[],
): IntentSelectionState[] {
  return selections.map(s => ({ ...s, selected: !s.selected }))
}
