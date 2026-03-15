import {
  addBackend,
  addPlugin,
  addRoute,
  addSignal,
  deleteBackend,
  deletePlugin,
  deleteRoute,
  deleteSignal,
  findBlock,
  serializeFields,
  updateBackend,
  updateGlobal,
  updatePlugin,
  updateRoute,
  updateSignal,
} from "@/lib/dslMutations";
import type { RouteInput } from "@/lib/dslMutations";
import type { ASTProgram } from "@/types/dsl";
import type {
  AlgorithmIntent,
  BackendIntent,
  ConditionNode,
  GlobalIntent,
  NLFieldType,
  NLPlannerResult,
  NLSchemaManifest,
  NLTypeSchemaEntry,
  IntentIR,
  ModelIntent,
  ModifyIntent,
  PluginRefIntent,
  PluginTemplateIntent,
  RouteIntent,
  SignalIntent,
} from "@/types/nl";
import { astRouteToInput } from "./builderPageRouteInputSupport";

export interface NLDraftResult {
  dsl: string;
  summary: string[];
}

export interface PreparedNLDraft {
  draft: NLDraftResult | null;
  error: string | null;
}

export function preparePlannerDraft(
  baseDsl: string,
  result: NLPlannerResult,
  schema: NLSchemaManifest | null,
  baseAst: ASTProgram | null = null,
): PreparedNLDraft {
  const schemaError = validatePlannerResultAgainstSchema(result, schema);
  if (schemaError) {
    return {
      draft: null,
      error: `Planner returned an invalid structured draft: ${schemaError}`,
    };
  }

  if (result.status !== "ready" || !result.intentIr) {
    return { draft: null, error: null };
  }

  try {
    return {
      draft: applyIntentIRToDSL(baseDsl, result.intentIr, baseAst),
      error: null,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return {
      draft: null,
      error: `Planner returned an invalid structured draft: ${message}`,
    };
  }
}

export function applyIntentIRToDSL(
  baseDsl: string,
  intentIR: IntentIR,
  baseAst: ASTProgram | null = null,
): NLDraftResult {
  let nextDsl = baseDsl;
  const summary: string[] = [];

  for (const intent of intentIR.intents) {
    switch (intent.type) {
      case "signal":
        nextDsl = applySignalIntent(nextDsl, intent);
        summary.push(`Signal ${intent.signal_type} ${intent.name}`);
        break;
      case "plugin_template":
        nextDsl = applyPluginIntent(nextDsl, intent);
        summary.push(`Plugin ${intent.plugin_type} ${intent.name}`);
        break;
      case "backend":
        nextDsl = applyBackendIntent(nextDsl, intent);
        summary.push(`Backend ${intent.backend_type} ${intent.name}`);
        break;
      case "route":
        nextDsl = applyRouteIntent(nextDsl, intent);
        summary.push(`Route ${intent.name}`);
        break;
      case "global":
        nextDsl = applyGlobalIntent(nextDsl, intent);
        summary.push("Global settings");
        break;
      case "modify":
        nextDsl = applyModifyIntent(nextDsl, intent, baseAst);
        summary.push(describeModifyIntent(intent));
        break;
      default:
        throw new Error(`Unsupported intent type: ${(intent as { type?: string }).type ?? "unknown"}`);
    }
  }

  return {
    dsl: normalizeDsl(nextDsl),
    summary,
  };
}

function applySignalIntent(src: string, intent: SignalIntent): string {
  const exists = !!findBlock(src, "SIGNAL", intent.signal_type, intent.name);
  return exists
    ? updateSignal(src, intent.signal_type, intent.name, intent.fields)
    : addSignal(src, intent.signal_type, intent.name, intent.fields);
}

function applyPluginIntent(src: string, intent: PluginTemplateIntent): string {
  const exists = !!findBlock(src, "PLUGIN", intent.plugin_type, intent.name);
  return exists
    ? updatePlugin(src, intent.name, intent.plugin_type, intent.fields)
    : addPlugin(src, intent.name, intent.plugin_type, intent.fields);
}

function applyBackendIntent(src: string, intent: BackendIntent): string {
  const exists = !!findBlock(src, "BACKEND", intent.backend_type, intent.name);
  return exists
    ? updateBackend(src, intent.backend_type, intent.name, intent.fields)
    : addBackend(src, intent.backend_type, intent.name, intent.fields);
}

function applyRouteIntent(src: string, intent: RouteIntent): string {
  const routeInput = toRouteInput(intent);
  const exists = !!findBlock(src, "ROUTE", null, intent.name);
  return exists
    ? updateRoute(src, intent.name, routeInput)
    : addRoute(src, intent.name, routeInput);
}

function applyGlobalIntent(src: string, intent: GlobalIntent): string {
  const globalBlock = findBlock(src, "GLOBAL", null, null);
  if (globalBlock) {
    return updateGlobal(src, intent.fields);
  }

  const body = serializeFields(intent.fields, "  ", { blankLineBefore: true });
  const block = `GLOBAL {\n${body}\n}\n`;
  return src.trim()
    ? `${src.trimEnd()}\n\n${block}`
    : block;
}

function applyModifyIntent(
  src: string,
  intent: ModifyIntent,
  baseAst: ASTProgram | null,
): string {
  switch (intent.action) {
    case "delete":
      return applyDeleteIntent(src, intent);
    case "add":
    case "update":
      return applyUpsertIntent(src, intent, baseAst);
    default:
      throw new Error(`Unsupported modify action: ${intent.action}`);
  }
}

function applyDeleteIntent(src: string, intent: ModifyIntent): string {
  switch (intent.target_construct) {
    case "signal": {
      const signalType = intent.target_signal_type ?? lookupSignalType(src, intent.target_name);
      if (!signalType) {
        throw new Error(`Unable to resolve signal type for ${intent.target_name}`);
      }
      return deleteSignal(src, signalType, intent.target_name);
    }
    case "plugin": {
      const pluginType = intent.target_plugin_type ?? lookupPluginType(src, intent.target_name);
      if (!pluginType) {
        throw new Error(`Unable to resolve plugin type for ${intent.target_name}`);
      }
      return deletePlugin(src, intent.target_name, pluginType);
    }
    case "backend": {
      const backendType = intent.target_backend_type ?? lookupBackendType(src, intent.target_name);
      if (!backendType) {
        throw new Error(`Unable to resolve backend type for ${intent.target_name}`);
      }
      return deleteBackend(src, backendType, intent.target_name);
    }
    case "route":
      return deleteRoute(src, intent.target_name);
    case "global":
      return src;
    default:
      throw new Error(`Unsupported delete target: ${intent.target_construct}`);
  }
}

function applyUpsertIntent(
  src: string,
  intent: ModifyIntent,
  baseAst: ASTProgram | null,
): string {
  const changes = intent.changes ?? {};

  switch (intent.target_construct) {
    case "signal": {
      const signalType = intent.target_signal_type;
      if (!signalType) {
        throw new Error(`Signal updates require target_signal_type for ${intent.target_name}`);
      }
      const exists = !!findBlock(src, "SIGNAL", signalType, intent.target_name);
      return exists
        ? updateSignal(src, signalType, intent.target_name, changes)
        : addSignal(src, signalType, intent.target_name, changes);
    }
    case "plugin": {
      const pluginType =
        intent.target_plugin_type ?? lookupPluginType(src, intent.target_name);
      if (!pluginType) {
        throw new Error(`Plugin updates require a plugin type for ${intent.target_name}`);
      }
      const exists = !!findBlock(src, "PLUGIN", pluginType, intent.target_name);
      return exists
        ? updatePlugin(src, intent.target_name, pluginType, changes)
        : addPlugin(src, intent.target_name, pluginType, changes);
    }
    case "backend": {
      const backendType =
        intent.target_backend_type ?? lookupBackendType(src, intent.target_name);
      if (!backendType) {
        throw new Error(`Backend updates require a backend type for ${intent.target_name}`);
      }
      const exists = !!findBlock(src, "BACKEND", backendType, intent.target_name);
      return exists
        ? updateBackend(src, backendType, intent.target_name, changes)
        : addBackend(src, backendType, intent.target_name, changes);
    }
    case "route": {
      const routeInput = mergeRouteChanges(
        intent.target_name,
        changes,
        resolveExistingRouteInput(intent.target_name, baseAst),
      );
      const exists = !!findBlock(src, "ROUTE", null, intent.target_name);
      return exists
        ? updateRoute(src, intent.target_name, routeInput)
        : addRoute(src, intent.target_name, routeInput);
    }
    case "global":
      return applyGlobalIntent(src, { type: "global", fields: changes });
    default:
      throw new Error(`Unsupported modify target: ${intent.target_construct}`);
  }
}

function toRouteInput(intent: RouteIntent): RouteInput {
  const models = intent.models.map(toRouteModel);
  if (models.length === 0) {
    throw new Error(`Route ${intent.name} requires at least one model`);
  }

  return {
    description: intent.description,
    priority: intent.priority ?? 100,
    when: intent.condition ? emitCondition(intent.condition) : undefined,
    models,
    algorithm: intent.algorithm
      ? {
          algoType: intent.algorithm.algo_type,
          fields: intent.algorithm.params ?? {},
        }
      : undefined,
    plugins: (intent.plugins ?? []).map(toRoutePlugin),
  };
}

function toRouteModel(intent: ModelIntent) {
  return {
    model: intent.model,
    reasoning: intent.reasoning,
    effort: intent.effort,
    lora: intent.lora,
    paramSize: intent.param_size,
    weight: intent.weight,
    reasoningFamily: intent.reasoning_family,
  };
}

function toRoutePlugin(intent: PluginRefIntent) {
  return {
    name: intent.name,
    fields: intent.overrides,
  };
}

function resolveExistingRouteInput(
  routeName: string,
  baseAst: ASTProgram | null,
): RouteInput | null {
  const route = baseAst?.routes?.find((candidate) => candidate.name === routeName);
  return route ? astRouteToInput(route) : null;
}

function mergeRouteChanges(
  routeName: string,
  changes: Record<string, unknown>,
  baseRoute: RouteInput | null,
): RouteInput {
  const hasDescription = hasOwn(changes, "description");
  const hasPriority = hasOwn(changes, "priority");
  const hasCondition = hasOwn(changes, "condition");
  const hasModels = hasOwn(changes, "models");
  const hasAlgorithm = hasOwn(changes, "algorithm");
  const hasPlugins = hasOwn(changes, "plugins");

  const models = hasModels
    ? asModelIntents(changes.models).map(toRouteModel)
    : (baseRoute?.models ?? []);
  if (models.length === 0) {
    throw new Error(`Route ${routeName} requires at least one model`);
  }

  const plugins = hasPlugins
    ? mergeRoutePlugins(
        baseRoute?.plugins ?? [],
        asPluginRefIntents(changes.plugins).map(toRoutePlugin),
      )
    : (baseRoute?.plugins ?? []);

  return {
    description: hasDescription
      ? asOptionalString(changes.description)
      : baseRoute?.description,
    priority: hasPriority
      ? asNumber(changes.priority, baseRoute?.priority ?? 100)
      : (baseRoute?.priority ?? 100),
    when: hasCondition
      ? renderChangedCondition(changes.condition)
      : baseRoute?.when,
    models,
    algorithm: hasAlgorithm
      ? renderChangedAlgorithm(changes.algorithm)
      : baseRoute?.algorithm,
    plugins,
  };
}

function mergeRoutePlugins(
  basePlugins: RouteInput["plugins"],
  incomingPlugins: RouteInput["plugins"],
): RouteInput["plugins"] {
  if (incomingPlugins.length === 0) {
    return [];
  }

  const merged = [...basePlugins];
  for (const plugin of incomingPlugins) {
    const existingIndex = merged.findIndex((candidate) => candidate.name === plugin.name);
    if (existingIndex >= 0) {
      merged[existingIndex] = {
        name: plugin.name,
        fields: plugin.fields ?? merged[existingIndex].fields,
      };
      continue;
    }
    merged.push(plugin);
  }
  return merged;
}

function renderChangedCondition(value: unknown): string | undefined {
  if (value === null) {
    return undefined;
  }
  const condition = asConditionNode(value);
  return condition ? emitCondition(condition) : undefined;
}

function renderChangedAlgorithm(value: unknown): RouteInput["algorithm"] {
  if (value === null) {
    return undefined;
  }
  const algorithm = asAlgorithmIntent(value);
  if (!algorithm) {
    return undefined;
  }
  return {
    algoType: algorithm.algo_type,
    fields: algorithm.params ?? {},
  };
}

function emitCondition(node: ConditionNode, parentOp?: ConditionNode["op"]): string {
  switch (node.op) {
    case "SIGNAL_REF":
      return `${node.signal_type}("${node.signal_name}")`;
    case "NOT": {
      const operand = emitCondition(node.operand, "NOT");
      const needsParens =
        node.operand.op === "AND" || node.operand.op === "OR";
      return needsParens ? `NOT (${operand})` : `NOT ${operand}`;
    }
    case "AND":
    case "OR": {
      const joiner = ` ${node.op} `;
      const rendered = node.operands
        .map((operand) => emitCondition(operand, node.op))
        .join(joiner);
      const needsParens = parentOp === "AND" && node.op === "OR";
      return needsParens ? `(${rendered})` : rendered;
    }
    default:
      throw new Error(`Unsupported condition operator: ${(node as { op?: string }).op ?? "unknown"}`);
  }
}

function describeModifyIntent(intent: ModifyIntent): string {
  switch (intent.action) {
    case "delete":
      return `Delete ${intent.target_construct} ${intent.target_name}`;
    case "add":
      return `Add ${intent.target_construct} ${intent.target_name}`;
    case "update":
      return `Update ${intent.target_construct} ${intent.target_name}`;
    default:
      return `${intent.target_construct} ${intent.target_name}`;
  }
}

function lookupSignalType(src: string, name: string): string | null {
  const match = new RegExp(`^SIGNAL\\s+(\\S+)\\s+${escapeRegex(name)}\\s*\\{`, "m").exec(src);
  return match?.[1] ?? null;
}

function lookupPluginType(src: string, name: string): string | null {
  const match = new RegExp(`^PLUGIN\\s+${escapeRegex(name)}\\s+(\\S+)\\s*\\{`, "m").exec(src);
  return match?.[1] ?? null;
}

function lookupBackendType(src: string, name: string): string | null {
  const match = new RegExp(`^BACKEND\\s+(\\S+)\\s+${escapeRegex(name)}\\s*\\{`, "m").exec(src);
  return match?.[1] ?? null;
}

function normalizeDsl(src: string): string {
  const normalized = src.replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  return normalized ? `${normalized}\n` : "";
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function hasOwn(value: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function asOptionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asConditionNode(value: unknown): ConditionNode | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  return value as ConditionNode;
}

function asModelIntents(value: unknown): ModelIntent[] {
  return Array.isArray(value) ? (value as ModelIntent[]) : [];
}

function asAlgorithmIntent(value: unknown): AlgorithmIntent | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  return value as AlgorithmIntent;
}

function asPluginRefIntents(value: unknown): PluginRefIntent[] {
  return Array.isArray(value) ? (value as PluginRefIntent[]) : [];
}

function validatePlannerResultAgainstSchema(
  result: NLPlannerResult,
  schema: NLSchemaManifest | null,
): string | null {
  if (!schema) {
    return null;
  }

  switch (result.status) {
    case "ready":
      return validateReadyResult(result, schema);
    case "needs_clarification":
      return validateClarificationResult(result);
    case "unsupported":
    case "error":
      return null;
    default:
      return `Planner returned unsupported status ${(result as { status?: string }).status ?? "unknown"}.`;
  }
}

function validateReadyResult(
  result: NLPlannerResult,
  schema: NLSchemaManifest,
): string | null {
  if (!result.intentIr) {
    return "Ready planner result is missing intentIr.";
  }
  if (!result.intentIr.intents.length) {
    return "Ready planner result is missing intents.";
  }

  const index = buildSchemaIndex(schema);
  for (const [intentIndex, intent] of result.intentIr.intents.entries()) {
    const rawIntent = intent as unknown as Record<string, unknown>;
    const intentType = asString(intent.type);
    if (!intentType) {
      return `Intent ${intentIndex + 1} is missing type.`;
    }

    switch (intentType) {
      case "signal": {
        const message = validateTypedFieldsIntent(
          rawIntent,
          "signal_type",
          index.signals,
          "signal",
        );
        if (message) {
          return `Intent ${intentIndex + 1}: ${message}`;
        }
        break;
      }
      case "plugin_template": {
        const message = validateTypedFieldsIntent(
          rawIntent,
          "plugin_type",
          index.plugins,
          "plugin",
        );
        if (message) {
          return `Intent ${intentIndex + 1}: ${message}`;
        }
        break;
      }
      case "backend": {
        const message = validateTypedFieldsIntent(
          rawIntent,
          "backend_type",
          index.backends,
          "backend",
        );
        if (message) {
          return `Intent ${intentIndex + 1}: ${message}`;
        }
        break;
      }
      case "route": {
        const message = validateRouteIntent(rawIntent, index);
        if (message) {
          return `Intent ${intentIndex + 1}: ${message}`;
        }
        break;
      }
      case "global":
        if (!isObject(rawIntent.fields)) {
          return `Intent ${intentIndex + 1}: global intent is missing fields.`;
        }
        break;
      case "modify": {
        const message = validateModifyIntent(rawIntent, index);
        if (message) {
          return `Intent ${intentIndex + 1}: ${message}`;
        }
        break;
      }
      default:
        return `Intent ${intentIndex + 1}: unsupported intent type ${intentType}.`;
    }
  }

  return null;
}

function validateClarificationResult(result: NLPlannerResult): string | null {
  if (!result.clarification) {
    return "Clarification result is missing clarification payload.";
  }
  if (!result.clarification.question.trim()) {
    return "Clarification result is missing a question.";
  }
  if (!result.clarification.options.length) {
    return "Clarification result has no options.";
  }
  return null;
}

function validateTypedFieldsIntent(
  intent: Record<string, unknown>,
  typeKey: "signal_type" | "plugin_type" | "backend_type",
  allowed: Map<string, NLTypeSchemaEntry>,
  label: string,
): string | null {
  const typeName = asString(intent[typeKey]);
  if (!typeName) {
    return `${label} intent is missing ${typeKey}.`;
  }
  const entry = allowed.get(typeName);
  if (!entry) {
    return `unsupported ${label} type ${typeName}.`;
  }
  if (!asString(intent.name)) {
    return `${label} intent is missing name.`;
  }
  const fields = asObject(intent.fields);
  if (!fields) {
    return `${label} intent is missing fields.`;
  }
  return validateFieldMap(fields, entry, true);
}

function validateRouteIntent(
  intent: Record<string, unknown>,
  index: ReturnType<typeof buildSchemaIndex>,
): string | null {
  if (!asString(intent.name)) {
    return "route intent is missing name.";
  }

  const models = asArray(intent.models);
  if (!models?.length) {
    return "route intent requires at least one model.";
  }
  for (const [modelIndex, modelEntry] of models.entries()) {
    const model = asObject(modelEntry);
    if (!model || !asString(model.model)) {
      return `route model ${modelIndex + 1} is missing model.`;
    }
  }

  const algorithm = asObject(intent.algorithm);
  if (algorithm) {
    const algoType = asString(algorithm.algo_type);
    if (!algoType) {
      return "route algorithm is missing algo_type.";
    }
    const schema = index.algorithms.get(algoType);
    if (!schema) {
      return `unsupported algorithm type ${algoType}.`;
    }
    const params = asObject(algorithm.params);
    if (params) {
      const message = validateFieldMap(params, schema, true);
      if (message) {
        return `route algorithm params: ${message}`;
      }
    }
  }

  const condition = asObject(intent.condition);
  if (condition) {
    return validateCondition(condition);
  }

  return null;
}

function validateModifyIntent(
  intent: Record<string, unknown>,
  index: ReturnType<typeof buildSchemaIndex>,
): string | null {
  const action = asString(intent.action);
  const targetConstruct = asString(intent.target_construct);
  if (!action) {
    return "modify intent is missing action.";
  }
  if (!targetConstruct) {
    return "modify intent is missing target_construct.";
  }
  if (!asString(intent.target_name)) {
    return "modify intent is missing target_name.";
  }
  if (action === "delete") {
    return null;
  }
  if (action !== "add" && action !== "update") {
    return `modify intent uses unsupported action ${action}.`;
  }

  switch (targetConstruct) {
    case "signal":
      return validateModifyTypedFields(
        intent,
        action,
        "target_signal_type",
        index.signals,
        "signal",
      );
    case "plugin":
      return validateModifyTypedFields(
        intent,
        action,
        "target_plugin_type",
        index.plugins,
        "plugin",
      );
    case "backend":
      return validateModifyTypedFields(
        intent,
        action,
        "target_backend_type",
        index.backends,
        "backend",
      );
    case "route": {
      const changes = asObject(intent.changes);
      if (!changes) {
        return "modify route intent is missing changes.";
      }
      return validateModifyRouteChanges(changes, index);
    }
    case "global":
      return isObject(intent.changes)
        ? null
        : "modify global intent is missing changes.";
    default:
      return `modify intent uses unsupported target_construct ${targetConstruct}.`;
  }
}

function validateModifyTypedFields(
  intent: Record<string, unknown>,
  action: string,
  typeKey: "target_signal_type" | "target_plugin_type" | "target_backend_type",
  allowed: Map<string, NLTypeSchemaEntry>,
  label: string,
): string | null {
  const typeName = asString(intent[typeKey]);
  if (!typeName) {
    return `modify ${label} intent is missing ${typeKey}.`;
  }
  const entry = allowed.get(typeName);
  if (!entry) {
    return `unsupported ${label} type ${typeName}.`;
  }
  const changes = asObject(intent.changes);
  if (!changes) {
    return `modify ${label} intent is missing changes.`;
  }
  return validateFieldMap(changes, entry, action === "add");
}

function validateModifyRouteChanges(
  changes: Record<string, unknown>,
  index: ReturnType<typeof buildSchemaIndex>,
): string | null {
  if (Object.keys(changes).length === 0) {
    return "modify route intent is missing supported changes.";
  }

  for (const key of Object.keys(changes)) {
    if (!["description", "priority", "condition", "models", "algorithm", "plugins"].includes(key)) {
      return `modify route intent does not support field ${key}.`;
    }
  }

  if (hasOwn(changes, "description") && changes.description !== null && typeof changes.description !== "string") {
    return "modify route description must be a string or null.";
  }

  if (hasOwn(changes, "priority") && changes.priority !== null && typeof changes.priority !== "number") {
    return "modify route priority must be a number or null.";
  }

  if (hasOwn(changes, "models")) {
    const models = asArray(changes.models);
    if (!models) {
      return "modify route models must be an array.";
    }
    if (!models.length) {
      return "modify route models cannot be empty.";
    }
    for (const [modelIndex, modelEntry] of models.entries()) {
      const model = asObject(modelEntry);
      if (!model || !asString(model.model)) {
        return `modify route model ${modelIndex + 1} is missing model.`;
      }
    }
  }

  if (hasOwn(changes, "algorithm") && changes.algorithm !== null) {
    const message = validateRouteIntent(
      {
        type: "route",
        name: "draft_route",
        models: [{ model: "placeholder-model" }],
        algorithm: changes.algorithm,
      },
      index,
    );
    if (message) {
      return message.replace("route ", "modify route ");
    }
  }

  if (hasOwn(changes, "condition") && changes.condition !== null) {
    const condition = asObject(changes.condition);
    if (!condition) {
      return "modify route condition must be an object or null.";
    }
    const message = validateCondition(condition);
    if (message) {
      return message;
    }
  }

  if (hasOwn(changes, "plugins")) {
    const plugins = asArray(changes.plugins);
    if (!plugins) {
      return "modify route plugins must be an array.";
    }
    for (const [pluginIndex, pluginEntry] of plugins.entries()) {
      const plugin = asObject(pluginEntry);
      if (!plugin || !asString(plugin.name)) {
        return `modify route plugin ${pluginIndex + 1} is missing name.`;
      }
      if (hasOwn(plugin, "overrides") && plugin.overrides !== null && !asObject(plugin.overrides)) {
        return `modify route plugin ${pluginIndex + 1} overrides must be an object or null.`;
      }
    }
  }

  return null;
}

function validateCondition(node: Record<string, unknown>): string | null {
  const op = asString(node.op);
  if (!op) {
    return "route condition is missing op.";
  }
  switch (op) {
    case "SIGNAL_REF":
      if (!asString(node.signal_type) || !asString(node.signal_name)) {
        return "route condition signal_ref is missing signal_type or signal_name.";
      }
      return null;
    case "NOT": {
      const operand = asObject(node.operand);
      return operand ? validateCondition(operand) : "route condition NOT is missing operand.";
    }
    case "AND":
    case "OR": {
      const operands = asArray(node.operands);
      if (!operands?.length) {
        return `route condition ${op} requires operands.`;
      }
      for (const operand of operands) {
        const child = asObject(operand);
        if (!child) {
          return `route condition ${op} contains a non-object operand.`;
        }
        const message = validateCondition(child);
        if (message) {
          return message;
        }
      }
      return null;
    }
    default:
      return `route condition uses unsupported op ${op}.`;
  }
}

function validateFieldMap(
  fields: Record<string, unknown>,
  entry: NLTypeSchemaEntry,
  requireRequiredFields: boolean,
): string | null {
  const fieldIndex = new Map((entry.fields ?? []).map((field) => [field.key, field]));
  for (const [key, value] of Object.entries(fields)) {
    const field = fieldIndex.get(key);
    if (!field) {
      return `field ${key} is not declared in schema.`;
    }
    const message = validateFieldValue(value, field.type, field.options ?? []);
    if (message) {
      return `field ${key}: ${message}`;
    }
  }

  for (const field of entry.fields ?? []) {
    if (!requireRequiredFields || !field.required) {
      continue;
    }
    const value = fields[field.key];
    if (value === undefined || isEmptyRequiredValue(value)) {
      return `missing required field ${field.key}.`;
    }
  }

  return null;
}

function validateFieldValue(
  value: unknown,
  type: NLFieldType,
  options: string[],
): string | null {
  switch (type) {
    case "string":
    case "select":
      if (typeof value !== "string") {
        return "expected string.";
      }
      if (type === "select" && options.length > 0 && !options.includes(value)) {
        return `expected one of ${options.join(", ")}.`;
      }
      return null;
    case "number":
      return typeof value === "number" ? null : "expected number.";
    case "boolean":
      return typeof value === "boolean" ? null : "expected boolean.";
    case "string[]":
      return Array.isArray(value) && value.every((item) => typeof item === "string")
        ? null
        : "expected string array.";
    case "number[]":
      return Array.isArray(value) && value.every((item) => typeof item === "number")
        ? null
        : "expected number array.";
    case "json":
      return null;
    default:
      return `unsupported field type ${String(type)}.`;
  }
}

function buildSchemaIndex(schema: NLSchemaManifest) {
  return {
    signals: new Map(schema.signals.map((entry) => [entry.typeName, entry])),
    plugins: new Map(schema.plugins.map((entry) => [entry.typeName, entry])),
    algorithms: new Map(schema.algorithms.map((entry) => [entry.typeName, entry])),
    backends: new Map(schema.backends.map((entry) => [entry.typeName, entry])),
  };
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function asObject(value: unknown): Record<string, unknown> | null {
  return isObject(value) ? value : null;
}

function asArray(value: unknown): unknown[] | null {
  return Array.isArray(value) ? value : null;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function isEmptyRequiredValue(value: unknown): boolean {
  if (typeof value === "string") {
    return !value.trim();
  }
  if (Array.isArray(value)) {
    return value.length === 0;
  }
  return value === null || value === undefined;
}
