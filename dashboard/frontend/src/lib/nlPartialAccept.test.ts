/**
 * Tests for nlPartialAccept module.
 */

import { describe, it, expect } from 'vitest'
import type { IntentIR, SignalIntent, RouteIntent, PluginTemplateIntent } from '@/types/intentIR'
import {
  analyzeIntentDependencies,
  detectDependencyIssues,
  createInitialSelections,
  toggleSelection,
  setAllSelections,
  autoSelectDependencies,
  buildPartialIntentIR,
  getIntentDisplayName,
  getIntentCategory,
  groupIntentsByCategory,
  getSelectionStats,
  selectByCategory,
  invertSelections,
} from './nlPartialAccept'

// ─────────────────────────────────────────────
// Test Fixtures
// ─────────────────────────────────────────────

const mathSignal: SignalIntent = {
  type: 'signal',
  signal_type: 'keyword',
  name: 'math',
  fields: { keywords: ['math', 'calculate'], operator: 'any' },
}

const codingSignal: SignalIntent = {
  type: 'signal',
  signal_type: 'keyword',
  name: 'coding',
  fields: { keywords: ['code', 'program'], operator: 'any' },
}

const mathRoute: RouteIntent = {
  type: 'route',
  name: 'math_route',
  description: 'Route for math questions',
  priority: 10,
  condition: {
    op: 'SIGNAL_REF',
    signal_type: 'keyword',
    signal_name: 'math',
  },
  models: [{ model: 'gpt-4o' }],
}

const codingRoute: RouteIntent = {
  type: 'route',
  name: 'coding_route',
  description: 'Route for coding questions',
  priority: 10,
  condition: {
    op: 'SIGNAL_REF',
    signal_type: 'keyword',
    signal_name: 'coding',
  },
  models: [{ model: 'deepseek-v3' }],
  plugins: [{ name: 'cache_plugin' }],
}

const cachePlugin: PluginTemplateIntent = {
  type: 'plugin_template',
  name: 'cache_plugin',
  plugin_type: 'semantic_cache',
  fields: { enabled: true, similarity_threshold: 0.95 },
}

const sampleIR: IntentIR = {
  version: '1.0',
  operation: 'generate',
  intents: [mathSignal, codingSignal, cachePlugin, mathRoute, codingRoute],
}

// ─────────────────────────────────────────────
// Dependency Analysis Tests
// ─────────────────────────────────────────────

describe('analyzeIntentDependencies', () => {
  it('should build correct dependency graph', () => {
    const graph = analyzeIntentDependencies(sampleIR)

    expect(graph.nodes).toHaveLength(5)
    expect(graph.edges.length).toBeGreaterThan(0)

    // mathRoute (index 3) depends on mathSignal (index 0)
    expect(graph.nodes[3].dependsOn).toContain(0)
    expect(graph.nodes[0].dependedBy).toContain(3)

    // codingRoute (index 4) depends on codingSignal (index 1) and cachePlugin (index 2)
    expect(graph.nodes[4].dependsOn).toContain(1)
    expect(graph.nodes[4].dependsOn).toContain(2)
  })

  it('should detect signal_ref edges', () => {
    const graph = analyzeIntentDependencies(sampleIR)
    const signalRefEdges = graph.edges.filter(e => e.type === 'signal_ref')

    expect(signalRefEdges.length).toBe(2) // math_route -> math, coding_route -> coding
  })

  it('should detect plugin_ref edges', () => {
    const graph = analyzeIntentDependencies(sampleIR)
    const pluginRefEdges = graph.edges.filter(e => e.type === 'plugin_ref')

    expect(pluginRefEdges.length).toBe(1) // coding_route -> cache_plugin
  })
})

// ─────────────────────────────────────────────
// Selection Management Tests
// ─────────────────────────────────────────────

describe('createInitialSelections', () => {
  it('should create all selected by default', () => {
    const selections = createInitialSelections(sampleIR)

    expect(selections).toHaveLength(5)
    expect(selections.every(s => s.selected)).toBe(true)
    expect(selections.every(s => !s.edited)).toBe(true)
  })
})

describe('toggleSelection', () => {
  it('should toggle single selection', () => {
    const selections = createInitialSelections(sampleIR)
    const updated = toggleSelection(selections, 0)

    expect(updated[0].selected).toBe(false)
    expect(updated[1].selected).toBe(true)
  })
})

describe('setAllSelections', () => {
  it('should set all to selected', () => {
    let selections = createInitialSelections(sampleIR)
    selections = setAllSelections(selections, false)
    selections = setAllSelections(selections, true)

    expect(selections.every(s => s.selected)).toBe(true)
  })

  it('should set all to deselected', () => {
    const selections = createInitialSelections(sampleIR)
    const updated = setAllSelections(selections, false)

    expect(updated.every(s => !s.selected)).toBe(true)
  })
})

describe('autoSelectDependencies', () => {
  it('should auto-select dependencies for selected intents', () => {
    const graph = analyzeIntentDependencies(sampleIR)
    let selections = createInitialSelections(sampleIR)

    // Deselect all first
    selections = setAllSelections(selections, false)
    // Select only coding_route (index 4)
    selections = toggleSelection(selections, 4)

    // Auto-select dependencies
    const updated = autoSelectDependencies(selections, graph)

    // Should now have coding_route (4), codingSignal (1), and cachePlugin (2) selected
    expect(updated[4].selected).toBe(true) // coding_route
    expect(updated[1].selected).toBe(true) // codingSignal (dependency)
    expect(updated[2].selected).toBe(true) // cachePlugin (dependency)
    expect(updated[0].selected).toBe(false) // mathSignal (not needed)
    expect(updated[3].selected).toBe(false) // mathRoute (not selected)
  })
})

// ─────────────────────────────────────────────
// Dependency Issue Detection Tests
// ─────────────────────────────────────────────

describe('detectDependencyIssues', () => {
  it('should detect missing signal dependency', () => {
    const graph = analyzeIntentDependencies(sampleIR)
    let selections = createInitialSelections(sampleIR)

    // Deselect mathSignal (index 0) but keep mathRoute (index 3)
    selections = toggleSelection(selections, 0)

    const issues = detectDependencyIssues(sampleIR, selections, graph)

    expect(issues.length).toBe(1)
    expect(issues[0].type).toBe('missing_signal')
    expect(issues[0].sourceIndex).toBe(3) // mathRoute has the issue
    expect(issues[0].missingIndex).toBe(0) // mathSignal is missing
  })

  it('should detect missing plugin dependency', () => {
    const graph = analyzeIntentDependencies(sampleIR)
    let selections = createInitialSelections(sampleIR)

    // Deselect cachePlugin (index 2) but keep codingRoute (index 4)
    selections = toggleSelection(selections, 2)

    const issues = detectDependencyIssues(sampleIR, selections, graph)

    expect(issues.some(i => i.type === 'missing_plugin')).toBe(true)
  })

  it('should not report issues when dependencies are satisfied', () => {
    const graph = analyzeIntentDependencies(sampleIR)
    const selections = createInitialSelections(sampleIR) // All selected

    const issues = detectDependencyIssues(sampleIR, selections, graph)

    expect(issues.length).toBe(0)
  })
})

// ─────────────────────────────────────────────
// Partial IR Generation Tests
// ─────────────────────────────────────────────

describe('buildPartialIntentIR', () => {
  it('should build IR with only selected intents', () => {
    let selections = createInitialSelections(sampleIR)
    // Deselect coding-related (indices 1, 2, 4)
    selections = toggleSelection(selections, 1)
    selections = toggleSelection(selections, 2)
    selections = toggleSelection(selections, 4)

    const partialIR = buildPartialIntentIR(sampleIR, selections)

    expect(partialIR.intents).toHaveLength(2) // mathSignal and mathRoute
    expect(partialIR.intents[0]).toBe(mathSignal)
    expect(partialIR.intents[1]).toBe(mathRoute)
  })

  it('should use edited intent when available', () => {
    let selections = createInitialSelections(sampleIR)

    // Mark first intent as edited
    const editedSignal = { ...mathSignal, name: 'edited_math' }
    selections[0] = {
      ...selections[0],
      edited: true,
      editedIntent: editedSignal,
    }

    const partialIR = buildPartialIntentIR(sampleIR, selections)

    expect((partialIR.intents[0] as SignalIntent).name).toBe('edited_math')
  })
})

// ─────────────────────────────────────────────
// Helper Function Tests
// ─────────────────────────────────────────────

describe('getIntentDisplayName', () => {
  it('should format signal names', () => {
    expect(getIntentDisplayName(mathSignal)).toBe('keyword("math")')
  })

  it('should format route names', () => {
    expect(getIntentDisplayName(mathRoute)).toBe('route "math_route"')
  })

  it('should format plugin names', () => {
    expect(getIntentDisplayName(cachePlugin)).toBe('plugin "cache_plugin" (semantic_cache)')
  })
})

describe('getIntentCategory', () => {
  it('should return correct categories', () => {
    expect(getIntentCategory(mathSignal)).toBe('signals')
    expect(getIntentCategory(mathRoute)).toBe('routes')
    expect(getIntentCategory(cachePlugin)).toBe('plugins')
  })
})

describe('groupIntentsByCategory', () => {
  it('should group intents correctly', () => {
    const selections = createInitialSelections(sampleIR)
    const groups = groupIntentsByCategory(sampleIR, selections)

    expect(groups['signals']).toHaveLength(2)
    expect(groups['routes']).toHaveLength(2)
    expect(groups['plugins']).toHaveLength(1)
  })
})

describe('getSelectionStats', () => {
  it('should compute correct stats', () => {
    let selections = createInitialSelections(sampleIR)
    selections = toggleSelection(selections, 0)
    selections = toggleSelection(selections, 1)

    const stats = getSelectionStats(selections)

    expect(stats.total).toBe(5)
    expect(stats.selected).toBe(3)
    expect(stats.edited).toBe(0)
  })
})

// ─────────────────────────────────────────────
// Batch Operation Tests
// ─────────────────────────────────────────────

describe('selectByCategory', () => {
  it('should select all of a category', () => {
    let selections = createInitialSelections(sampleIR)
    selections = setAllSelections(selections, false)
    selections = selectByCategory(sampleIR, selections, 'signals', true)

    // Only signals should be selected
    expect(selections[0].selected).toBe(true) // mathSignal
    expect(selections[1].selected).toBe(true) // codingSignal
    expect(selections[2].selected).toBe(false) // cachePlugin
    expect(selections[3].selected).toBe(false) // mathRoute
    expect(selections[4].selected).toBe(false) // codingRoute
  })
})

describe('invertSelections', () => {
  it('should invert all selections', () => {
    let selections = createInitialSelections(sampleIR)
    selections = toggleSelection(selections, 0)
    selections = toggleSelection(selections, 1)

    // Before: [false, false, true, true, true]
    const inverted = invertSelections(selections)

    // After: [true, true, false, false, false]
    expect(inverted[0].selected).toBe(true)
    expect(inverted[1].selected).toBe(true)
    expect(inverted[2].selected).toBe(false)
    expect(inverted[3].selected).toBe(false)
    expect(inverted[4].selected).toBe(false)
  })
})
