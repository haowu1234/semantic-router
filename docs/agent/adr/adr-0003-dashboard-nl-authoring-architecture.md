# ADR 0003: Keep Dashboard NL Authoring on the Canonical DSL Builder Contract

## Status

Accepted

## Context

Issue `#1511` proposes coding-agent-backed natural-language authoring for the dashboard Builder. The current mainline Builder already has the canonical authoring and deployment path for this surface:

- Builder state is rooted in DSL source plus the existing WASM compile, validate, decompile, and format flow.
- loading current router config, deploy preview, and deploy already exist as stable dashboard behaviors.
- `EditorMode` already reserves an `nl` mode, but the UI is still a placeholder.

The exploratory `feat/nl` branch proved several useful ideas:

- a natural-language chat-style Builder workflow is viable.
- an intermediate `Intent IR` is safer than direct LLM-to-DSL text generation.
- deterministic modify flows can reuse existing DSL mutation helpers.

The same branch also showed architectural drift that should not become the long-term design:

- NL orchestration, schema mirrors, repair logic, and model routing were pushed primarily into the frontend.
- backend support was a generic LLM proxy rather than a Builder-specific planning surface.
- repair logic drifted from canonical DSL syntax.
- the prototype was not actually wired into the main Builder mode or toolbar flow.

This feature crosses `dashboard_config_ui`, `dashboard_console_backend`, and `dsl_crd`, so the repo needs one durable architectural decision before implementation starts.

## Decision

Use the following architecture for dashboard NL authoring:

- NL authoring is a Builder mode, not a new Config page and not a parallel config-editing surface.
- The canonical contract remains DSL source plus the existing Builder compile, validate, deploy-preview, and deploy flow.
- The backend owns NL planning, session state, clarification handling, policy checks, and provider/model selection.
- The backend returns structured planning outputs such as `Intent IR`, clarifications, warnings, and repair guidance, not final persisted router config.
- The frontend owns the Builder review experience and deterministic `Intent IR` to DSL application against the existing DSL store and mutation helpers.
- Validation authority remains the canonical DSL compiler and validator path; NL-specific repair must operate on structured planning state first and only use canonical validator output as feedback.
- The repo should prefer one shared schema or manifest source derived from canonical Builder or compiler knowledge over separate frontend-only NL schema tables.
- `BuilderPage.tsx` remains a route-level orchestrator only; NL-specific UI, transport, and helper logic belong in adjacent modules.
- The initial `v0.3` preview uses a constrained planner backend behind a stable interface. It does not expose a general-purpose coding agent directly to the dashboard.
- If the repo later adds a true coding-agent-backed planner, it must fit behind the same backend planner interface and remain bounded by Builder-specific policy and review gates.

## Consequences

- The repo should add a dedicated NL authoring backend surface rather than extend the generic config handlers or merge the `feat/nl` LLM proxy as-is.
- The frontend may reuse concepts from `feat/nl`, especially `Intent IR`, review UX, and deterministic mutation patterns, but should not merge the branch wholesale.
- Arbitrary frontend-supplied upstream endpoints or API keys are out of scope for the canonical implementation; planner backends should be server-controlled and policy-gated.
- Apply and deploy remain separate actions: NL mode can produce a draft and apply it into Builder state, but deployment still uses the existing preview and confirmation path.
- `readonly` mode can support draft generation and inspection, but it must not allow draft application or deploy.
- A user-visible NL Builder feature requires E2E coverage in addition to unit and integration tests.
- If implementation phases ship a narrower preview than this ADR's end state, any residual architecture gap should be recorded as durable technical debt instead of living only in chat or PR text.
