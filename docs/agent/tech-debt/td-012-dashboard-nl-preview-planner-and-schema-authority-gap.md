# TD012: Dashboard NL Preview Still Uses a Bounded Planner and Duplicated Schema Authority

## Status

Open

## Scope

dashboard NL authoring planner backend, planner tool governance, and schema ownership

## Summary

The dashboard Builder still ships a bounded NL authoring preview by default, but the backend now has server-owned `structured-llm` and `tool-calling-llm` backends behind the stable planner seam, plus a planner-owned registry that can host readonly Builder-domain tools, an optional builtin web tool source, and allowlisted MCP tools. The backend continues to validate planner output against the NL schema manifest before the draft reaches Builder apply flows, and the repo now derives planner capability and shared frontend schema data from a canonical generated manifest pipeline. Partial route modify drafts already merge against the current Builder AST so route updates preserve existing conditions and plugin refs, and backend planner validation treats typed-nil object fields as absent instead of misclassifying valid route drafts as planner errors. The repo also now has planner turn/tool observability hooks, broader create/modify/fix fixtures, a model-backed Builder E2E path, and a dedicated rollout playbook for backend and tool-source promotion. Even with those improvements, the implementation still diverges from the desired end state in two durable ways. First, the new model-backed backends remain bounded to the preview support subset and are not yet the production-intended default planner. Second, some presentation-oriented UI tables and non-canonical config helpers still sit outside the generated catalog, so they can drift if they evolve independently.

## Evidence

- [dashboard/backend/nlauthor/planner.go](../../../dashboard/backend/nlauthor/planner.go)
- [dashboard/backend/nlauthor/planner_support.go](../../../dashboard/backend/nlauthor/planner_support.go)
- [dashboard/backend/nlauthor/planner_result_validation.go](../../../dashboard/backend/nlauthor/planner_result_validation.go)
- [dashboard/backend/nlauthor/catalog/schema_manifest.json](../../../dashboard/backend/nlauthor/catalog/schema_manifest.json)
- [dashboard/backend/nlauthor/catalog/generate_frontend_catalog.py](../../../dashboard/backend/nlauthor/catalog/generate_frontend_catalog.py)
- [dashboard/backend/nlauthor/openai_provider.go](../../../dashboard/backend/nlauthor/openai_provider.go)
- [dashboard/backend/nlauthor/planner_factory.go](../../../dashboard/backend/nlauthor/planner_factory.go)
- [dashboard/backend/nlauthor/preview_planner.go](../../../dashboard/backend/nlauthor/preview_planner.go)
- [dashboard/backend/nlauthor/structured_llm_planner.go](../../../dashboard/backend/nlauthor/structured_llm_planner.go)
- [dashboard/backend/nlauthor/tool_calling_planner.go](../../../dashboard/backend/nlauthor/tool_calling_planner.go)
- [dashboard/backend/nlauthor/planner_tools.go](../../../dashboard/backend/nlauthor/planner_tools.go)
- [dashboard/backend/nlauthor/planner_tools_builder.go](../../../dashboard/backend/nlauthor/planner_tools_builder.go)
- [dashboard/backend/nlauthor/planner_tools_builtin_web.go](../../../dashboard/backend/nlauthor/planner_tools_builtin_web.go)
- [dashboard/backend/nlauthor/planner_tools_mcp.go](../../../dashboard/backend/nlauthor/planner_tools_mcp.go)
- [dashboard/backend/nlauthor/schema.go](../../../dashboard/backend/nlauthor/schema.go)
- [dashboard/backend/router/core_routes.go](../../../dashboard/backend/router/core_routes.go)
- [dashboard/backend/mcp/manager.go](../../../dashboard/backend/mcp/manager.go)
- [dashboard/frontend/src/lib/generatedNlSchemaManifest.ts](../../../dashboard/frontend/src/lib/generatedNlSchemaManifest.ts)
- [dashboard/frontend/src/lib/nlSchemaCatalog.ts](../../../dashboard/frontend/src/lib/nlSchemaCatalog.ts)
- [dashboard/frontend/src/pages/builderPageNLDraft.ts](../../../dashboard/frontend/src/pages/builderPageNLDraft.ts)
- [dashboard/frontend/src/types/nl.ts](../../../dashboard/frontend/src/types/nl.ts)
- [docs/agent/adr/adr-0003-dashboard-nl-authoring-architecture.md](../adr/adr-0003-dashboard-nl-authoring-architecture.md)
- [docs/agent/plans/pl-0003-dashboard-nl-authoring-roadmap.md](../plans/pl-0003-dashboard-nl-authoring-roadmap.md)
- [docs/agent/plans/pl-0004-dashboard-nl-llm-planner-roadmap.md](../plans/pl-0004-dashboard-nl-llm-planner-roadmap.md)

## Why It Matters

- Issue [#1511](https://github.com/vllm-project/semantic-router/issues/1511) explicitly asked for a coding-agent-backed authoring path, so it is not enough to merely expose additional backends behind config flags; the repo still needs confidence that the model-backed backends are the safe default path.
- The repo now has a planner-owned registry plus rollout and observability guidance, but promotion still depends on proving that the bounded support subset, tool policy, and validation path hold up as coverage expands.
- If the remaining dashboard surfaces continue to own local presentation tables or broader type inventories outside the shared catalog, the product can still show configuration affordances that do not line up with NL-specific manifest semantics.
- The current preview is appropriate for bounded generation, clarification, and repair, but it will cap feature growth and raise maintenance cost if more construct coverage is added without finishing the schema-authority convergence first.

## Desired End State

- The dashboard backend exposes a stable planner seam that can host a guarded model-backed or coding-agent-backed implementation without changing Builder UI contracts.
- Tool calling, when introduced, runs through a planner-owned registry that can mount Builder-domain tools first and only later allow builtin web tools or MCP-managed tools behind explicit source policy.
- At least one model-backed planner backend is hardened enough, tested enough, and observable enough to serve as the production-intended default instead of a preview-only alternate backend.
- NL schema capability and field authority derive from one canonical source, or from an explicitly generated artifact, instead of parallel hand-maintained definitions.
- The frontend canonicalizer remains deterministic, but it consumes schema metadata that is guaranteed to stay aligned with planner support and DSL mutation constraints.
- The remaining dashboard documentation and add/edit affordances that rely on type descriptions or typed field editors also consume the same generated catalog instead of local hand-maintained tables.

## Exit Criteria

- A production-intended planner backend exists behind `dashboard/backend/nlauthor/planner.go` that is more capable than the preview ruleset and is governed by explicit tool and policy limits.
- The repo has a planner-owned tool registry that can host readonly Builder-domain tools and, if enabled, adapted builtin web tools or allowlisted MCP tools behind a common policy layer.
- Regression coverage and rollout documentation demonstrate when `structured-llm` and `tool-calling-llm` can be enabled safely beyond preview-only environments and when the repo may promote one of them as the default planner.
- The repo has one authoritative schema source, or one generated manifest pipeline, for NL planner capability, frontend Builder affordances, and deterministic DSL application rules.
- The main Builder field-schema exports, user-visible DSL guide text, and shared topology signal type inventories are fully sourced from that generated catalog, or any intentional exception is tracked separately.
- Regression coverage demonstrates that newly added NL constructs cannot be advertised by the planner unless the canonical Builder path can validate and apply them.
