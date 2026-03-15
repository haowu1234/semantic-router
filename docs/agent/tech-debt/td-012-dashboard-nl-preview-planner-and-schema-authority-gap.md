# TD012: Dashboard NL Preview Still Uses a Bounded Planner and Duplicated Schema Authority

## Status

Open

## Scope

dashboard NL authoring planner backend and schema ownership

## Summary

The dashboard Builder now ships a bounded NL authoring preview that stays on the canonical DSL path, and the repo now validates planner output against the NL schema manifest before the draft reaches Builder apply flows. The backend also derives advertised planner support from the planner's own support profile instead of a broad static capability claim. A new single-source schema catalog now exists as repo data, with the backend loading it directly and the frontend consuming a generated catalog module for Builder field schemas, DSL guide text, and shared topology signal type inventories. Partial route modify drafts now merge against the current Builder AST so route updates preserve existing conditions and plugin refs, and backend planner validation now treats typed-nil object fields as absent instead of misclassifying valid route drafts as planner errors. Even with those guards, the preview architecture still diverges from the desired end state in two durable ways. First, the backend planner is still a constrained `preview-rulebased` implementation instead of a guarded coding-agent-backed planner interface. Second, some presentation-oriented UI tables and non-canonical config helpers still sit outside the generated catalog, so they can drift if they evolve independently.

## Evidence

- [dashboard/backend/nlauthor/planner.go](../../../dashboard/backend/nlauthor/planner.go)
- [dashboard/backend/nlauthor/planner_support.go](../../../dashboard/backend/nlauthor/planner_support.go)
- [dashboard/backend/nlauthor/planner_result_validation.go](../../../dashboard/backend/nlauthor/planner_result_validation.go)
- [dashboard/backend/nlauthor/catalog/schema_manifest.json](../../../dashboard/backend/nlauthor/catalog/schema_manifest.json)
- [dashboard/backend/nlauthor/catalog/generate_frontend_catalog.py](../../../dashboard/backend/nlauthor/catalog/generate_frontend_catalog.py)
- [dashboard/backend/nlauthor/preview_planner.go](../../../dashboard/backend/nlauthor/preview_planner.go)
- [dashboard/backend/nlauthor/schema.go](../../../dashboard/backend/nlauthor/schema.go)
- [dashboard/frontend/src/lib/generatedNlSchemaManifest.ts](../../../dashboard/frontend/src/lib/generatedNlSchemaManifest.ts)
- [dashboard/frontend/src/lib/nlSchemaCatalog.ts](../../../dashboard/frontend/src/lib/nlSchemaCatalog.ts)
- [dashboard/frontend/src/pages/builderPageNLDraft.ts](../../../dashboard/frontend/src/pages/builderPageNLDraft.ts)
- [dashboard/frontend/src/types/nl.ts](../../../dashboard/frontend/src/types/nl.ts)
- [docs/agent/adr/adr-0003-dashboard-nl-authoring-architecture.md](../adr/adr-0003-dashboard-nl-authoring-architecture.md)
- [docs/agent/plans/pl-0003-dashboard-nl-authoring-roadmap.md](../plans/pl-0003-dashboard-nl-authoring-roadmap.md)

## Why It Matters

- Issue [#1511](https://github.com/vllm-project/semantic-router/issues/1511) explicitly asked for a coding-agent-backed authoring path, so the bounded preview planner is a deliberate but durable scope cut, not the final target architecture.
- If the remaining dashboard surfaces continue to own local presentation tables or broader type inventories outside the shared catalog, the product can still show configuration affordances that do not line up with NL-specific manifest semantics.
- The current preview is appropriate for bounded generation, clarification, and repair, but it will cap feature growth and raise maintenance cost if more construct coverage is added without finishing the schema-authority convergence first.

## Desired End State

- The dashboard backend exposes a stable planner seam that can host a guarded coding-agent-backed implementation without changing Builder UI contracts.
- NL schema capability and field authority derive from one canonical source, or from an explicitly generated artifact, instead of parallel hand-maintained definitions.
- The frontend canonicalizer remains deterministic, but it consumes schema metadata that is guaranteed to stay aligned with planner support and DSL mutation constraints.
- The remaining dashboard documentation and add/edit affordances that rely on type descriptions or typed field editors also consume the same generated catalog instead of local hand-maintained tables.

## Exit Criteria

- A production-intended planner backend exists behind `dashboard/backend/nlauthor/planner.go` that is more capable than the preview ruleset and is governed by explicit tool and policy limits.
- The repo has one authoritative schema source, or one generated manifest pipeline, for NL planner capability, frontend Builder affordances, and deterministic DSL application rules.
- The main Builder field-schema exports, user-visible DSL guide text, and shared topology signal type inventories are fully sourced from that generated catalog, or any intentional exception is tracked separately.
- Regression coverage demonstrates that newly added NL constructs cannot be advertised by the planner unless the canonical Builder path can validate and apply them.
