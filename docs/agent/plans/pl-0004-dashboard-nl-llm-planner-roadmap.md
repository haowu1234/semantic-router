# Dashboard NL LLM Planner Execution Plan

This document tracks the next implementation loop for dashboard NL authoring after the preview Builder flow shipped. The goal is to evolve the backend planner from a bounded ruleset into a guarded model-backed planner without changing the Builder's canonical DSL review/apply contract.

## Goal

- Add a server-owned model-backed planner backend behind the existing NL authoring planner seam.
- Fold tool calling into that planner roadmap as a bounded backend capability, not as a separate user-facing agent surface.
- Keep Builder UI contracts stable while improving modify, fix, and long-context planning quality.

## Scope

- `dashboard/backend/nlauthor/**` planner backend selection, provider abstraction, tool registry, and policy gates
- Builder-scoped NL capabilities and schema endpoints as they relate to planner backend rollout
- planner-owned tool discovery and invocation for builtin Builder tools, optional builtin web tools, and allowlisted MCP tools
- structured planner output, clarification, repair, and fallback behavior behind the existing `PlannerResult` contract
- planner evaluation, backend coverage, and affected Builder E2E updates for the new backend paths

## Out of Scope for This Loop

- changing the frontend NL authoring contract away from `PlannerResult -> Intent IR -> deterministic DSL`
- exposing a general-purpose coding agent directly to dashboard users
- arbitrary frontend model, provider, endpoint, or API-key overrides
- automatic deploy without Builder review, apply, and the existing deploy confirmation flow
- unrestricted shell, repo-write, or free-form network tool access from planner turns

## Exit Criteria

- The backend can run either the existing `preview-rulebased` planner or a server-owned model-backed planner behind the same `Planner` interface.
- The first production-intended model-backed planner emits strict structured `PlannerResult` payloads that validate against the existing NL contract before reaching Builder apply flows.
- Tool calling, when enabled, runs through a planner-owned registry with explicit source allowlists, readonly limits, timeouts, and turn budgets.
- The first tool-backed implementation uses Builder-domain read-only tools before any external or MCP-backed tools are enabled.
- The repo documents rollout and fallback rules for `preview-rulebased`, `structured-llm`, and future `tool-calling-llm` backends.
- Validation coverage demonstrates create, modify, fix, clarification, fallback, and tool-policy behavior for the new planner backends.

## Current State Snapshot

- The shipped Builder preview is still backed by the bounded `preview-rulebased` planner by default, but the backend can now also select `structured-llm` and `tool-calling-llm` behind the same `Planner` interface.
- The frontend still consumes planner output through deterministic `Intent IR -> DSL` application and canonical validation, so the backend planner seam remains stable.
- The backend now owns planner backend selection, provider configuration, an OpenAI-compatible provider adapter, and first-pass `StructuredLLMPlanner` and `ToolCallingPlanner` implementations.
- The planner runtime now has a planner-owned registry that can mount readonly Builder-domain tools, an optional builtin web tool source, and an allowlisted MCP source behind the same registry interface.
- The current model-backed backends still use the preview support subset, and the repo still lacks the broader evaluation, observability, and rollout hardening needed to treat those backends as the production default.

## Task List

- [ ] `L001` Record the next-stage implementation plan and align debt tracking.
  - Done when this plan is indexed, `TD012` reflects the model-backed planner and planner-tool-registry target, and the preview roadmap links to this loop.
- [x] `L002` Add backend planner backend selection and provider configuration.
  - Done when `nlauthor` can select `preview-rulebased`, `structured-llm`, or later `tool-calling-llm` from server-owned config, without frontend provider overrides.
- [x] `L003` Introduce a provider abstraction for structured planner calls.
  - Done when the backend has a provider interface plus an initial OpenAI-compatible implementation that can request strict JSON planner output with backend-owned retry and timeout policy.
- [x] `L004` Implement `StructuredLLMPlanner` behind the existing planner seam.
  - Done when Builder NL API turns can run through a model-backed planner that returns validated `PlannerResult` payloads for the current preview construct subset and falls back safely when unavailable.
- [x] `L005` Add a planner-owned tool registry and builtin Builder-domain tool source.
  - Done when the planner can discover and invoke readonly Builder tools such as symbol lookup, schema subset lookup, diagnostics lookup, relevant DSL snippet lookup, and candidate `Intent IR` validation through one registry interface.
- [x] `L006` Implement bounded `ToolCallingPlanner` turns.
  - Done when modify/fix-capable planner turns can use the planner-owned registry with explicit tool budgets, readonly enforcement, timeout handling, transcript limits, and final structured-output validation.
- [x] `L007` Add optional planner tool sources for builtin web tools and allowlisted MCP tools.
  - Done when the planner registry can mount additional tool sources behind explicit source policies and allowlists, without forcing planner code to call HTTP handlers or depend directly on `mcp.Manager` semantics.
- [x] `L008` Add planner evaluation, observability, and affected test coverage.
  - Done when fixtures cover create/modify/fix/clarification/fallback cases, tool-policy behavior is unit-tested, and affected Builder E2E coverage exists for the model-backed planner path.
- [x] `L009` Document rollout, fallback, and production guardrails.
  - Done when canonical docs explain how the repo stages `preview-rulebased`, `structured-llm`, and `tool-calling-llm`, including when external tool sources may be enabled.

## Current Loop

- Loop status: completed on 2026-03-16 after the preview Builder NL workflow shipped and the model-backed planner seam, tool registry, observability, and rollout playbook landed.
- Current focus:
  - keep the frontend contract fixed while future planner coverage expands behind the now-complete backend seam
  - keep Builder-domain readonly tools as the primary tool-calling path while external sources stay opt-in
  - use the rollout playbook before promoting `structured-llm` or `tool-calling-llm` beyond preview-only deployments
- Completion rule for this loop:
  - no planner backend should bypass `PlannerResult` validation, Builder review/apply gates, or server-owned tool policy.

## Decision Log

- This loop extends `ADR 0003`; it does not replace the canonical DSL Builder contract or reopen the decision to keep NL authoring out of a parallel config path.
- The first model-backed backend should be a structured-output planner, not a user-facing coding agent.
- Tool calling belongs inside the same backend planner roadmap as a bounded enhancement phase, not as a separate product surface.
- The planner should not call dashboard HTTP handlers or `/api/mcp/tools` endpoints directly; instead it should use a planner-owned registry that can adapt builtin services and allowlisted MCP tools behind a common policy layer.
- The first planner tool source should be Builder-domain readonly tools, because they carry the highest value for modify/fix flows and the lowest policy risk.
- Builtin web tools and MCP tools may be adapted later, but only behind explicit allowlists and server-owned source policy.
- A 2026-03-16 implementation slice completed `L002` and `L003` by adding dashboard config-driven planner backend selection, an OpenAI-compatible provider seam, router wiring into `NewServiceFromRuntimeConfig`, and the first `StructuredLLMPlanner` implementation that keeps the Builder contract unchanged.
- A follow-up 2026-03-16 implementation slice completed `L004` through `L007` by adding the first `StructuredLLMPlanner` and `ToolCallingPlanner` implementations, a planner-owned tool registry, readonly Builder-domain tools, an optional builtin web tool source, and an allowlisted MCP source adapter.
- A follow-up 2026-03-16 implementation slice completed `L008` and `L009` by adding planner turn/tool observability hooks, broader create/modify/fix fixtures, a model-backed Builder E2E path, and a canonical rollout playbook for backend selection and tool-source policy.

## Follow-up Debt / ADR Links

- [../adr/adr-0003-dashboard-nl-authoring-architecture.md](../adr/adr-0003-dashboard-nl-authoring-architecture.md)
- [../plans/pl-0003-dashboard-nl-authoring-roadmap.md](pl-0003-dashboard-nl-authoring-roadmap.md)
- [../playbooks/dashboard-nl-planner-rollout.md](../playbooks/dashboard-nl-planner-rollout.md)
- [../tech-debt/td-012-dashboard-nl-preview-planner-and-schema-authority-gap.md](../tech-debt/td-012-dashboard-nl-preview-planner-and-schema-authority-gap.md)
