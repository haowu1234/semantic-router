/**
 * Intent IR Validator using Zod
 *
 * Provides runtime validation for LLM-generated Intent IR JSON.
 * Ensures the output conforms to the expected schema and catches
 * common LLM mistakes before they propagate to DSL generation.
 */

import { z } from 'zod'
import type { IntentIR } from '@/types/intentIR'

// ─────────────────────────────────────────────
// Condition Node Schema
// ─────────────────────────────────────────────

const ConditionNodeSchema: z.ZodType<unknown> = z.lazy(() =>
  z.discriminatedUnion('op', [
    z.object({
      op: z.literal('AND'),
      operands: z.array(ConditionNodeSchema),
    }),
    z.object({
      op: z.literal('OR'),
      operands: z.array(ConditionNodeSchema),
    }),
    z.object({
      op: z.literal('NOT'),
      operand: ConditionNodeSchema,
    }),
    z.object({
      op: z.literal('SIGNAL_REF'),
      signal_type: z.string(),
      signal_name: z.string(),
    }),
  ])
)

// ─────────────────────────────────────────────
// Model Intent Schema
// ─────────────────────────────────────────────

const ModelIntentSchema = z.object({
  model: z.string().min(1, 'Model name is required'),
  reasoning: z.boolean().optional(),
  effort: z.enum(['low', 'medium', 'high']).optional(),
  lora: z.string().optional(),
  param_size: z.string().optional(),
  weight: z.number().optional(),
  reasoning_family: z.string().optional(),
})

// ─────────────────────────────────────────────
// Algorithm Intent Schema
// ─────────────────────────────────────────────

const AlgorithmIntentSchema = z.object({
  algo_type: z.string(),
  params: z.record(z.unknown()).default({}),
})

// ─────────────────────────────────────────────
// Plugin Reference Schema
// ─────────────────────────────────────────────

const PluginRefIntentSchema = z.object({
  name: z.string().min(1, 'Plugin name is required'),
  overrides: z.record(z.unknown()).optional(),
})

// ─────────────────────────────────────────────
// Individual Intent Schemas
// ─────────────────────────────────────────────

const SignalIntentSchema = z.object({
  type: z.literal('signal'),
  signal_type: z.string().min(1, 'Signal type is required'),
  name: z.string().min(1, 'Signal name is required'),
  fields: z.record(z.unknown()).default({}),
})

const RouteIntentSchema = z.object({
  type: z.literal('route'),
  name: z.string().min(1, 'Route name is required'),
  description: z.string().optional(),
  priority: z.number().int().optional(),
  condition: ConditionNodeSchema,
  models: z.array(ModelIntentSchema).min(1, 'At least one model is required'),
  algorithm: AlgorithmIntentSchema.optional(),
  plugins: z.array(PluginRefIntentSchema).optional(),
})

const PluginTemplateIntentSchema = z.object({
  type: z.literal('plugin_template'),
  name: z.string().min(1, 'Plugin name is required'),
  plugin_type: z.string().min(1, 'Plugin type is required'),
  fields: z.record(z.unknown()).default({}),
})

const BackendIntentSchema = z.object({
  type: z.literal('backend'),
  backend_type: z.string().min(1, 'Backend type is required'),
  name: z.string().min(1, 'Backend name is required'),
  fields: z.record(z.unknown()).default({}),
})

const GlobalIntentSchema = z.object({
  type: z.literal('global'),
  fields: z.record(z.unknown()).default({}),
})

const ModifyIntentSchema = z.object({
  type: z.literal('modify'),
  action: z.enum(['add', 'update', 'delete']),
  target_construct: z.enum(['signal', 'route', 'plugin', 'backend', 'global']),
  target_name: z.string().min(1, 'Target name is required'),
  target_signal_type: z.string().optional(),
  target_plugin_type: z.string().optional(),
  target_backend_type: z.string().optional(),
  changes: z.record(z.unknown()).optional(),
})

// ─────────────────────────────────────────────
// Combined Intent Schema
// ─────────────────────────────────────────────

const IntentSchema = z.discriminatedUnion('type', [
  SignalIntentSchema,
  RouteIntentSchema,
  PluginTemplateIntentSchema,
  BackendIntentSchema,
  GlobalIntentSchema,
  ModifyIntentSchema,
])

// ─────────────────────────────────────────────
// Full Intent IR Schema
// ─────────────────────────────────────────────

const IntentIRSchema = z.object({
  version: z.literal('1.0'),
  operation: z.enum(['generate', 'modify', 'fix']),
  intents: z.array(IntentSchema),
})

// ─────────────────────────────────────────────
// Validation Result Types
// ─────────────────────────────────────────────

export interface IRValidationResult {
  success: boolean
  data?: IntentIR
  errors: IRValidationError[]
}

export interface IRValidationError {
  path: string
  message: string
  received?: unknown
}

// ─────────────────────────────────────────────
// Validation Functions
// ─────────────────────────────────────────────

/**
 * Validate an Intent IR object against the schema.
 * Returns a result object with detailed error information.
 */
export function validateIntentIR(input: unknown): IRValidationResult {
  const result = IntentIRSchema.safeParse(input)

  if (result.success) {
    return {
      success: true,
      data: result.data as IntentIR,
      errors: [],
    }
  }

  const errors: IRValidationError[] = result.error.issues.map(issue => ({
    path: issue.path.join('.'),
    message: issue.message,
    received: 'received' in issue ? issue.received : undefined,
  }))

  return {
    success: false,
    errors,
  }
}

/**
 * Validate and parse an Intent IR, throwing if invalid.
 */
export function parseIntentIR(input: unknown): IntentIR {
  return IntentIRSchema.parse(input) as IntentIR
}

/**
 * Try to repair common LLM mistakes in the Intent IR.
 * Returns the repaired object or null if not repairable.
 */
export function repairIntentIR(input: unknown): IntentIR | null {
  if (typeof input !== 'object' || input === null) {
    return null
  }

  const obj = input as Record<string, unknown>

  // Fix 1: Missing version
  if (!obj.version) {
    obj.version = '1.0'
  }

  // Fix 2: Wrong version format
  if (obj.version !== '1.0') {
    obj.version = '1.0'
  }

  // Fix 3: Missing operation
  if (!obj.operation) {
    obj.operation = 'generate'
  }

  // Fix 4: Wrong operation value
  if (!['generate', 'modify', 'fix'].includes(obj.operation as string)) {
    obj.operation = 'generate'
  }

  // Fix 5: Missing intents array
  if (!Array.isArray(obj.intents)) {
    obj.intents = []
  }

  // Fix 6: Repair individual intents
  const intents = obj.intents as Array<Record<string, unknown>>
  for (let i = 0; i < intents.length; i++) {
    const intent = intents[i]
    if (typeof intent !== 'object' || intent === null) continue

    // Fix route with empty models array
    if (intent.type === 'route') {
      if (!Array.isArray(intent.models) || (intent.models as unknown[]).length === 0) {
        // Add a placeholder model
        intent.models = [{ model: 'gpt-4o-mini' }]
      }
    }

    // Fix signal with missing fields
    if (intent.type === 'signal' && !intent.fields) {
      intent.fields = {}
    }

    // Fix plugin_template with missing fields
    if (intent.type === 'plugin_template' && !intent.fields) {
      intent.fields = {}
    }

    // Fix backend with missing fields
    if (intent.type === 'backend' && !intent.fields) {
      intent.fields = {}
    }

    // Fix global with missing fields
    if (intent.type === 'global' && !intent.fields) {
      intent.fields = {}
    }
  }

  // Try to parse again after repairs
  const result = IntentIRSchema.safeParse(obj)
  if (result.success) {
    return result.data as IntentIR
  }

  return null
}

/**
 * Validate a single intent object.
 */
export function validateIntent(input: unknown): { success: boolean; errors: IRValidationError[] } {
  const result = IntentSchema.safeParse(input)

  if (result.success) {
    return { success: true, errors: [] }
  }

  const errors: IRValidationError[] = result.error.issues.map(issue => ({
    path: issue.path.join('.'),
    message: issue.message,
    received: 'received' in issue ? issue.received : undefined,
  }))

  return { success: false, errors }
}

// ─────────────────────────────────────────────
// Semantic Validation (beyond schema)
// ─────────────────────────────────────────────

/**
 * Perform semantic validation on a validated Intent IR.
 * Checks for issues like undefined signal references, duplicate names, etc.
 */
export function semanticValidation(ir: IntentIR): IRValidationError[] {
  const errors: IRValidationError[] = []

  // Collect defined signal names
  const definedSignals = new Map<string, string>() // key: "type:name", value: "signal"
  const definedPlugins = new Set<string>()
  const definedBackends = new Set<string>()

  for (const intent of ir.intents) {
    if (intent.type === 'signal') {
      const key = `${intent.signal_type}:${intent.name}`
      if (definedSignals.has(key)) {
        errors.push({
          path: `intents.signal.${intent.name}`,
          message: `Duplicate signal definition: ${intent.signal_type}("${intent.name}")`,
        })
      }
      definedSignals.set(key, intent.name)
    }

    if (intent.type === 'plugin_template') {
      if (definedPlugins.has(intent.name)) {
        errors.push({
          path: `intents.plugin.${intent.name}`,
          message: `Duplicate plugin definition: "${intent.name}"`,
        })
      }
      definedPlugins.add(intent.name)
    }

    if (intent.type === 'backend') {
      if (definedBackends.has(intent.name)) {
        errors.push({
          path: `intents.backend.${intent.name}`,
          message: `Duplicate backend definition: "${intent.name}"`,
        })
      }
      definedBackends.add(intent.name)
    }
  }

  // Check route conditions for undefined signal references
  for (const intent of ir.intents) {
    if (intent.type === 'route') {
      const undefinedRefs = findUndefinedSignalRefs(intent.condition, definedSignals)
      for (const ref of undefinedRefs) {
        errors.push({
          path: `intents.route.${intent.name}.condition`,
          message: `Undefined signal reference: ${ref.type}("${ref.name}")`,
        })
      }

      // Check plugin references
      if (intent.plugins) {
        for (const pluginRef of intent.plugins) {
          if (!definedPlugins.has(pluginRef.name)) {
            errors.push({
              path: `intents.route.${intent.name}.plugins`,
              message: `Undefined plugin reference: "${pluginRef.name}"`,
            })
          }
        }
      }
    }
  }

  return errors
}

function findUndefinedSignalRefs(
  condition: unknown,
  definedSignals: Map<string, string>,
): Array<{ type: string; name: string }> {
  const undefinedRefs: Array<{ type: string; name: string }> = []

  function traverse(node: unknown): void {
    if (typeof node !== 'object' || node === null) return

    const n = node as Record<string, unknown>

    if (n.op === 'SIGNAL_REF') {
      const key = `${n.signal_type}:${n.signal_name}`
      if (!definedSignals.has(key)) {
        undefinedRefs.push({
          type: n.signal_type as string,
          name: n.signal_name as string,
        })
      }
    }

    if (n.op === 'AND' || n.op === 'OR') {
      const operands = n.operands as unknown[]
      for (const operand of operands) {
        traverse(operand)
      }
    }

    if (n.op === 'NOT') {
      traverse(n.operand)
    }
  }

  traverse(condition)
  return undefinedRefs
}

// ─────────────────────────────────────────────
// Exports
// ─────────────────────────────────────────────

export { IntentIRSchema, IntentSchema }
