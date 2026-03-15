import type { ASTProgram } from "@/types/dsl";
import type {
  ConditionNode,
  Intent,
  ModifyIntent,
  NLPlannerResult,
  PluginRefIntent,
  RouteIntent,
} from "@/types/nl";
import { astRouteToInput } from "./builderPageRouteInputSupport";

export interface NLReviewCard {
  id: string;
  kicker: string;
  title: string;
  summary: string;
  details: string[];
}

export function buildIntentReviewCards(
  result: NLPlannerResult | null,
  baseAst: ASTProgram | null = null,
): NLReviewCard[] {
  if (!result?.intentIr || result.status !== "ready") {
    return [];
  }

  return result.intentIr.intents.map((intent, index) =>
    buildIntentReviewCard(intent, index, baseAst),
  );
}

function buildIntentReviewCard(
  intent: Intent,
  index: number,
  baseAst: ASTProgram | null,
): NLReviewCard {
  switch (intent.type) {
    case "signal":
      return {
        id: `signal-${intent.name}-${index}`,
        kicker: "Signal draft",
        title: `${intent.signal_type} ${intent.name}`,
        summary: "Create or update a signal block in Builder DSL.",
        details: summarizeFieldMap(intent.fields, [
          ["keywords", "Keywords"],
          ["operator", "Operator"],
          ["threshold", "Threshold"],
          ["description", "Description"],
        ]),
      };
    case "plugin_template":
      return {
        id: `plugin-${intent.name}-${index}`,
        kicker: "Plugin template",
        title: `${intent.plugin_type} ${intent.name}`,
        summary: "Create or update a reusable plugin block.",
        details: summarizeFieldMap(intent.fields, [
          ["message", "Message"],
          ["enabled", "Enabled"],
          ["capture_request_body", "Capture request body"],
          ["capture_response_body", "Capture response body"],
        ]),
      };
    case "backend":
      return {
        id: `backend-${intent.name}-${index}`,
        kicker: "Backend",
        title: `${intent.backend_type} ${intent.name}`,
        summary: "Create or update a backend block used by routes.",
        details: summarizeFieldMap(intent.fields, [
          ["address", "Address"],
          ["model", "Model"],
          ["timeout_seconds", "Timeout"],
          ["enabled", "Enabled"],
        ]),
      };
    case "route":
      return buildRouteCard(intent, index);
    case "global":
      return {
        id: `global-${index}`,
        kicker: "Global settings",
        title: "Router-wide config",
        summary: "Apply changes to the GLOBAL block.",
        details: summarizeFieldMap(intent.fields, []),
      };
    case "modify":
      return buildModifyCard(intent, index, baseAst);
    default:
      return {
        id: `intent-${index}`,
        kicker: "Draft change",
        title: "Planner change",
        summary: "Review the canonical draft for details.",
        details: [],
      };
  }
}

function buildRouteCard(intent: RouteIntent, index: number): NLReviewCard {
  const details: string[] = [];
  details.push(`Model refs: ${intent.models.map((model) => model.model).join(", ")}`);
  details.push(`Priority: ${intent.priority ?? 100}`);
  details.push(
    `Condition: ${intent.condition ? describeCondition(intent.condition) : "None"}`,
  );
  details.push(`Plugins: ${formatPluginRefs(intent.plugins)}`);
  if (intent.algorithm) {
    details.push(`Algorithm: ${intent.algorithm.algo_type}`);
  }
  if (intent.description) {
    details.push(`Description: ${intent.description}`);
  }
  return {
    id: `route-${intent.name}-${index}`,
    kicker: "Route",
    title: intent.name,
    summary: "Create or update a route block in Builder DSL.",
    details,
  };
}

function buildModifyCard(
  intent: ModifyIntent,
  index: number,
  baseAst: ASTProgram | null,
): NLReviewCard {
  const targetName = intent.target_construct === "global" ? "GLOBAL" : intent.target_name;
  const title = `${capitalize(intent.action)} ${intent.target_construct} ${targetName}`.trim();

  if (intent.action === "delete") {
    return {
      id: `modify-${intent.target_construct}-${targetName}-${index}`,
      kicker: "Delete",
      title,
      summary: "Remove this construct from the working Builder DSL.",
      details: ["Removes the existing block from the next reviewed draft."],
    };
  }

  if (intent.target_construct === "route") {
    return {
      id: `modify-route-${targetName}-${index}`,
      kicker: `${capitalize(intent.action)} route`,
      title: targetName,
      summary: "Update only the route fields described below.",
      details: summarizeRouteChanges(
        intent.changes,
        resolveBaseRouteDetails(targetName, baseAst),
      ),
    };
  }

  if (intent.target_construct === "signal") {
    return {
      id: `modify-signal-${targetName}-${index}`,
      kicker: `${capitalize(intent.action)} signal`,
      title: `${intent.target_signal_type ?? "signal"} ${targetName}`,
      summary: "Apply targeted field updates to this signal block.",
      details: summarizeFieldMap(intent.changes ?? {}, []),
    };
  }

  if (intent.target_construct === "plugin") {
    return {
      id: `modify-plugin-${targetName}-${index}`,
      kicker: `${capitalize(intent.action)} plugin`,
      title: `${intent.target_plugin_type ?? "plugin"} ${targetName}`,
      summary: "Apply targeted field updates to this plugin block.",
      details: summarizeFieldMap(intent.changes ?? {}, []),
    };
  }

  if (intent.target_construct === "backend") {
    return {
      id: `modify-backend-${targetName}-${index}`,
      kicker: `${capitalize(intent.action)} backend`,
      title: `${intent.target_backend_type ?? "backend"} ${targetName}`,
      summary: "Apply targeted field updates to this backend block.",
      details: summarizeFieldMap(intent.changes ?? {}, []),
    };
  }

  return {
    id: `modify-global-${index}`,
    kicker: `${capitalize(intent.action)} global`,
    title: "GLOBAL settings",
    summary: "Apply targeted field updates to the global block.",
    details: summarizeFieldMap(intent.changes ?? {}, []),
  };
}

function summarizeRouteChanges(
  changes: Record<string, unknown> | undefined,
  baseRoute: ReturnType<typeof resolveBaseRouteDetails>,
): string[] {
  if (!changes) {
    return ["No explicit field changes were provided."];
  }

  const details: string[] = [];
  if (Object.prototype.hasOwnProperty.call(changes, "models")) {
    const models = asArray(changes.models)
      .map((entry) => asRecord(entry)?.model)
      .filter((value): value is string => typeof value === "string");
    details.push(
      describeFieldTransition(
        "Model refs",
        baseRoute?.models ?? null,
        models.length ? models.join(", ") : "Cleared",
      ),
    );
  } else if (baseRoute?.models) {
    details.push(`Model refs: ${baseRoute.models} (unchanged)`);
  }
  if (Object.prototype.hasOwnProperty.call(changes, "priority")) {
    details.push(
      describeFieldTransition(
        "Priority",
        baseRoute?.priority ?? null,
        formatValue(changes.priority),
      ),
    );
  } else if (baseRoute?.priority) {
    details.push(`Priority: ${baseRoute.priority} (unchanged)`);
  }
  if (Object.prototype.hasOwnProperty.call(changes, "condition")) {
    const condition = asRecord(changes.condition);
    details.push(
      describeFieldTransition(
        "Condition",
        baseRoute?.condition ?? null,
        condition ? describeCondition(condition as ConditionNode) : "Cleared",
      ),
    );
  } else if (baseRoute?.condition) {
    details.push(`Condition: ${baseRoute.condition} (unchanged)`);
  }
  if (Object.prototype.hasOwnProperty.call(changes, "plugins")) {
    details.push(
      describeFieldTransition(
        "Plugins",
        baseRoute?.plugins ?? null,
        formatPluginRefs(asPluginRefs(changes.plugins)),
      ),
    );
  } else if (baseRoute?.plugins) {
    details.push(`Plugins: ${baseRoute.plugins} (unchanged)`);
  }
  if (Object.prototype.hasOwnProperty.call(changes, "algorithm")) {
    const algorithm = asRecord(changes.algorithm);
    details.push(
      describeFieldTransition(
        "Algorithm",
        baseRoute?.algorithm ?? null,
        typeof algorithm?.algo_type === "string" ? algorithm.algo_type : "Updated",
      ),
    );
  } else if (baseRoute?.algorithm) {
    details.push(`Algorithm: ${baseRoute.algorithm} (unchanged)`);
  }
  if (Object.prototype.hasOwnProperty.call(changes, "description")) {
    details.push(
      describeFieldTransition(
        "Description",
        baseRoute?.description ?? null,
        formatValue(changes.description),
      ),
    );
  } else if (baseRoute?.description) {
    details.push(`Description: ${baseRoute.description} (unchanged)`);
  }

  details.push("Unspecified route settings stay unchanged.");
  return details;
}

function resolveBaseRouteDetails(
  routeName: string,
  baseAst: ASTProgram | null,
): {
  description: string | null;
  priority: string | null;
  condition: string | null;
  models: string | null;
  algorithm: string | null;
  plugins: string | null;
} | null {
  const route = baseAst?.routes?.find((candidate) => candidate.name === routeName);
  if (!route) {
    return null;
  }

  const input = astRouteToInput(route);
  return {
    description: input.description ?? null,
    priority:
      typeof input.priority === "number" ? String(input.priority) : null,
    condition: input.when ?? null,
    models: input.models.length
      ? input.models.map((model) => model.model).join(", ")
      : null,
    algorithm: input.algorithm?.algoType ?? null,
    plugins: input.plugins.length
      ? input.plugins.map((plugin) => plugin.name).join(", ")
      : null,
  };
}

function summarizeFieldMap(
  fields: Record<string, unknown>,
  preferredKeys: Array<[string, string]>,
): string[] {
  const details: string[] = [];
  const usedKeys = new Set<string>();

  for (const [key, label] of preferredKeys) {
    if (!Object.prototype.hasOwnProperty.call(fields, key)) {
      continue;
    }
    usedKeys.add(key);
    details.push(`${label}: ${formatValue(fields[key])}`);
  }

  for (const [key, value] of Object.entries(fields)) {
    if (usedKeys.has(key)) {
      continue;
    }
    details.push(`${humanizeKey(key)}: ${formatValue(value)}`);
    if (details.length >= 5) {
      break;
    }
  }

  return details.length ? details : ["No field-level details were provided."];
}

function describeCondition(condition: ConditionNode): string {
  switch (condition.op) {
    case "SIGNAL_REF":
      return `${condition.signal_type}(${condition.signal_name})`;
    case "NOT":
      return `NOT ${describeCondition(condition.operand)}`;
    case "AND":
    case "OR":
      return condition.operands.map(describeCondition).join(` ${condition.op} `);
    default:
      return "Custom condition";
  }
}

function formatPluginRefs(plugins: PluginRefIntent[] | undefined): string {
  if (!plugins || plugins.length === 0) {
    return "None";
  }
  return plugins.map((plugin) => plugin.name).join(", ");
}

function describeFieldTransition(
  label: string,
  before: string | null,
  after: string,
): string {
  if (!before || before === after) {
    return `${label}: ${after}`;
  }
  return `${label}: ${before} -> ${after}`;
}

function asPluginRefs(value: unknown): PluginRefIntent[] | undefined {
  const entries = asArray(value)
    .map((entry) => asRecord(entry))
    .filter((entry): entry is Record<string, unknown> => Boolean(entry))
    .map((entry) => ({
      name: typeof entry.name === "string" ? entry.name : "plugin",
      overrides:
        entry.overrides && typeof entry.overrides === "object"
          ? (entry.overrides as Record<string, unknown>)
          : undefined,
    }));

  return entries.length ? entries : undefined;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function formatValue(value: unknown): string {
  if (value === null) {
    return "Cleared";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    const rendered = value
      .map((entry) => {
        if (typeof entry === "string" || typeof entry === "number") {
          return String(entry);
        }
        const record = asRecord(entry);
        if (record?.model && typeof record.model === "string") {
          return record.model;
        }
        if (record?.name && typeof record.name === "string") {
          return record.name;
        }
        return "item";
      })
      .join(", ");
    return rendered || "None";
  }
  const record = asRecord(value);
  if (record) {
    if (typeof record.name === "string") {
      return record.name;
    }
    if (typeof record.model === "string") {
      return record.model;
    }
    return "Updated";
  }
  return String(value);
}

function humanizeKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
