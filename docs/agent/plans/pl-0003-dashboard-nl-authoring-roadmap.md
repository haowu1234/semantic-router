# Dashboard NL Authoring Execution Plan

This document tracks the long-horizon workstream for delivering natural-language authoring in the dashboard Builder without creating a parallel config path.

## Goal

- Deliver a reviewable NL authoring flow in the dashboard Builder that stays on the canonical DSL contract.
- Keep the work resumable across multiple implementation loops, contributors, and release phases.

## Scope

- Builder `nl` mode UX, routing, and review flow
- backend NL authoring APIs, planner orchestration, session lifecycle, and policy gates
- shared `Intent IR` and schema-manifest contract used by planner and Builder integration
- canonical DSL validation, compile, deploy-preview, and readonly behavior integration
- unit, integration, and E2E coverage for the NL Builder path

## Out of Scope for the Initial Preview

- direct dashboard exposure of a general-purpose coding agent
- automatic deploy without human review and existing deploy confirmation
- free-form frontend selection of arbitrary model endpoints or API keys
- replacing the DSL Builder with a chat-only config surface
- production-grade multi-agent orchestration beyond a bounded planner interface

## Exit Criteria

- A user can enter Builder NL mode, describe a new config or a modification, and receive a reviewable draft.
- The draft is represented as structured planning output and is deterministically applied into DSL using the canonical Builder path.
- The resulting DSL can be validated, compiled, and sent through the existing deploy-preview flow without a parallel config writer.
- Ambiguous prompts surface clarification instead of silently guessing.
- `readonly` mode permits inspection-only NL workflows and blocks apply and deploy.
- Relevant frontend, backend, and Builder E2E coverage exists for the shipped behavior.
- ADR and implementation state remain aligned, or any intentional preview gap is captured in indexed technical debt.

## Current State Snapshot

- Builder now exposes a preview `nl` mode in the main dashboard path, backed by Builder-scoped `/api/builder/nl/*` endpoints and a bounded preview planner.
- The dashboard applies planner output through deterministic `Intent IR -> DSL` logic and then reuses the existing DSL validation, compile, and deploy-preview flow.
- Partial route modify drafts now merge against the current Builder AST so NL updates can preserve existing route conditions, plugin refs, and other untouched fields instead of rewriting the route block from scratch.
- The backend now owns a canonical NL schema manifest as repo data, and the frontend consumes a generated catalog derived from that source for Builder field schemas, DSL guide text, and shared type inventories such as topology signal enums.
- The preview still carries a deliberate residual gap: the planner is not yet a true coding-agent backend, and a few non-canonical UI presentation tables still sit outside the generated schema catalog. That gap remains tracked in `TD012`.

## Task List

- [x] `N001` Record the durable architecture decision and execution plan before implementation starts.
  - Done when the repo has a canonical ADR and an indexed execution plan for this workstream.
- [x] `N002` Define the shared NL contract.
  - Done when `Intent IR`, clarification payloads, planner result types, and schema-manifest ownership are versioned and documented in code-facing modules.
- [x] `N003` Implement the backend NL authoring surface.
  - Done when the dashboard backend exposes Builder-specific NL session and planning APIs with server-controlled provider policy.
- [x] `N004` Integrate Builder NL mode without growing hotspot debt.
  - Done when `BuilderPage.tsx` stays a thin orchestrator and NL mode UI, review panels, and API clients live in adjacent modules.
- [x] `N005` Reuse the canonical DSL pipeline end to end.
  - Done when NL drafts are applied through deterministic `Intent IR` to DSL logic plus the existing DSL store compile, validate, and deploy-preview actions.
- [x] `N006` Add clarification, repair, and readonly guardrails.
  - Done when ambiguous prompts require explicit clarification, repair loops consume canonical validator output, and readonly mode blocks state-changing actions.
- [x] `N007` Add tests and affected gates for the user-visible flow.
  - Done when unit and integration coverage exists for planner and Builder logic, and at least one Builder E2E flow covers NL authoring behavior.
- [x] `N008` Decide whether the preview architecture leaves durable residual debt.
  - Done when any intentionally deferred gap, such as a later true coding-agent planner backend or schema-authority cleanup, is either retired or recorded in indexed technical debt.

## Current Loop

- Loop status: preview implementation slice completed on 2026-03-13.
- Current focus:
  - hand off the shipped preview with residual architecture gaps recorded in `TD012`
  - keep future planner expansion behind the existing Builder-scoped NL contracts
- Completion rule for this loop:
  - the implementation should not start by copying `feat/nl` wholesale; each adopted piece must be mapped to the ADR's ownership boundaries first.

## Decision Log

- This plan follows `ADR 0003`, which keeps NL authoring on the canonical DSL Builder contract.
- The repo should mine `feat/nl` for reusable ideas and tests, but not treat the branch as the merge target.
- The first shipped preview should use a constrained planner backend behind a stable interface instead of directly exposing a general-purpose coding agent.
- `N002` is satisfied by backend-owned contract types and schema/capabilities endpoints under `/api/builder/nl/*`, plus matching frontend-facing types and API helpers.
- `N003` is satisfied by a Builder-scoped backend surface under `/api/builder/nl/*`, server-owned in-memory session state, strict rejection of client-supplied provider overrides, and a bounded preview planner behind the stable planner interface.
- `N004/N005` are satisfied by a thin Builder NL shell, a deterministic frontend `Intent IR -> DSL` applier that reuses existing `dslMutations`, and `Open in DSL` / apply hooks that stay inside the existing Builder store flow.
- `N006` is satisfied by fix-mode repair turns on the existing session API, validator-driven clarification for undefined signals, unknown construct types, threshold constraints, corrected ready-draft repairs for threshold fixes and simple undefined-signal route repairs, and readonly apply gating in Builder NL mode.
- `N007` is satisfied by backend unit coverage, handler coverage, frontend type-check, and Builder E2E coverage for `ready -> review -> Open in DSL`, readonly apply gating, `invalid draft -> Retry with diagnostics -> repair clarification -> corrected draft`, and `Apply draft -> Deploy preview`.
- `N008` is satisfied by `TD012`, which records the intentional preview gap between the bounded planner shipped now and the desired future coding-agent-backed planner plus schema-authority cleanup.
- Follow-up hardening on 2026-03-15 added backend planner-result normalization against the NL schema manifest, frontend draft preparation through a single schema-aware `preparePlannerDraft` seam, and planner-derived capability profiles so preview capability responses no longer advertise unsupported type options such as `pii` signal creation.
- Follow-up hardening later on 2026-03-15 moved the NL schema manifest to a canonical backend-owned JSON catalog with a generated frontend catalog module, then rewired Builder field-schema exports and topology signal type inventories to consume that shared data instead of hand-maintained copies.
- Follow-up hardening later on 2026-03-15 also made partial route modify intents merge with the current Builder AST, extended preview route planning and E2E coverage for route updates, and closed a backend planner-validation bug where typed-nil route conditions were being normalized into false `error` results.
- If implementation loops discover a durable mismatch between the ADR and the delivered preview, the repo should add or update an indexed technical debt entry in the same change.

## Follow-up Debt / ADR Links

- [../adr/adr-0003-dashboard-nl-authoring-architecture.md](../adr/adr-0003-dashboard-nl-authoring-architecture.md)
- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
