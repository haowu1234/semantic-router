import {
  serializeBoolExpr,
  type RouteAlgoInput,
  type RouteInput,
  type RouteModelInput,
  type RoutePluginInput,
} from "@/lib/dslMutations";
import type {
  ASTAlgoSpec,
  ASTModelRef,
  ASTPluginRef,
  ASTRouteDecl,
} from "@/types/dsl";

export function astModelToInput(model: ASTModelRef): RouteModelInput {
  return {
    model: model.model,
    reasoning: model.reasoning,
    effort: model.effort,
    lora: model.lora,
    paramSize: model.paramSize,
    weight: model.weight,
    reasoningFamily: model.reasoningFamily,
  };
}

export function astAlgoToInput(algorithm?: ASTAlgoSpec): RouteAlgoInput | undefined {
  if (!algorithm) {
    return undefined;
  }

  return {
    algoType: algorithm.algoType,
    fields: { ...algorithm.fields },
  };
}

export function astPluginRefToInput(plugin: ASTPluginRef): RoutePluginInput {
  return {
    name: plugin.name,
    fields: plugin.fields ? { ...plugin.fields } : undefined,
  };
}

export function astRouteToInput(route: ASTRouteDecl): RouteInput {
  return {
    description: route.description,
    priority: route.priority,
    when: route.when
      ? serializeBoolExpr(route.when as unknown as Record<string, unknown>)
      : undefined,
    models: route.models.map(astModelToInput),
    algorithm: astAlgoToInput(route.algorithm),
    plugins: route.plugins.map(astPluginRefToInput),
  };
}
