# Dashboard NL Planner Rollout Playbook

Use this playbook when enabling or validating model-backed Builder NL planner backends under `dashboard/backend/nlauthor/**`.

## Stable Contract

- Keep the frontend contract fixed at `PlannerResult -> Intent IR -> deterministic DSL`.
- Keep Builder review/apply gates intact. Planner backends may propose drafts; they must not bypass canonical validation, apply, or deploy preview.
- Treat planner backend selection, provider choice, and tool policy as server-owned config. Do not add frontend endpoint, model, or API-key overrides.

## Backends

- `preview-rulebased`
  - default safe fallback
  - use when provider config is missing, external tools are disallowed, or a release needs the bounded subset only
- `structured-llm`
  - first production-intended model-backed path
  - use when strict JSON `PlannerResult` output, clarification, and repair behavior are validated for the targeted prompt subset
- `tool-calling-llm`
  - bounded enhancement for modify/fix and long-context turns
  - use only after readonly Builder-domain tools are stable and tool budget / allowlist behavior is covered by tests

## Config Surface

Relevant backend flags and env vars live in [dashboard/backend/config/config.go](../../../dashboard/backend/config/config.go).

- `NL_PLANNER_BACKEND`
- `NL_PLANNER_PROVIDER`
- `NL_PLANNER_BASE_URL`
- `NL_PLANNER_API_KEY`
- `NL_PLANNER_MODEL`
- `NL_PLANNER_TIMEOUT_MS`
- `NL_PLANNER_MAX_OUTPUT_TOKENS`
- `NL_PLANNER_TOOL_BUDGET`
- `NL_PLANNER_ALLOW_WEB_TOOLS`
- `NL_PLANNER_ALLOW_MCP_TOOLS`
- `NL_PLANNER_ALLOWED_MCP_TOOLS`

Recommended staging order:

1. start with `preview-rulebased`
2. enable `structured-llm` in local or staging with server-owned provider config
3. keep `tool-calling-llm` disabled until Builder-domain readonly tools are validated
4. opt into builtin web tools or MCP tools only behind explicit allowlists

For `vllm-sr serve` local container workflows:

- export `NL_PLANNER_*` in the same shell before `vllm-sr serve`
- use the local image flow (`make vllm-sr-dev ...` plus `--image-pull-policy never`) when validating local planner changes
- use container DNS names for sibling services on `vllm-sr-network`, or `host.docker.internal` when the planner targets a host-local provider

## Tool Policy

Planner tool calling must stay behind the planner-owned registry in [dashboard/backend/nlauthor/planner_tools.go](../../../dashboard/backend/nlauthor/planner_tools.go).

- always enable `builtin_builder` tools first
- keep `builtin_web` off by default
- keep `mcp` off by default
- require explicit allowlists for MCP tool names
- use bounded budgets and timeouts; never allow unbounded tool loops
- do not let planner turns call shell, git, repo-write, or deploy actions

## Fallback Rules

- if provider config is incomplete, expose the configured backend as unavailable and keep Builder review-only behavior intact
- if provider calls fail, return structured planner errors; do not silently apply drafts
- if a model-backed backend is unstable in an environment, revert `NL_PLANNER_BACKEND` to `preview-rulebased`
- do not promote external tool sources ahead of Builder-domain readonly tools

## Validation Before Promotion

Before promoting a backend beyond preview-only environments:

- run backend unit coverage for config, provider, planner, and tool policy
- run affected NL handler tests
- run Builder Playwright coverage for the model-backed planner path
- confirm planner output still validates through canonical DSL review/apply
- confirm logs or equivalent observability show backend, status, duration, and tool-call outcomes per turn

Recommended commands:

- `go test ./config ./nlauthor ./router -run 'TestLoadConfigAppliesNLPlannerFlags|TestOpenAICompatibleProviderGenerateStructured|TestOpenAICompatibleProviderGenerateToolCalls|TestStructuredLLMPlannerParsesProviderJSON|TestStructuredLLMPlannerRecordsObserverEvent|TestStructuredLLMPlannerFixtures|TestToolCallingPlannerUsesRegistryThenParsesFinalJSON|TestToolCallingPlannerRecordsToolObserverEvents|TestToolCallingPlannerFixtures|TestNewPlannerFromRuntimeConfigMarksMissingStructuredProviderUnavailable|TestNewNLAuthoringServiceUsesConfiguredStructuredPlanner'`
- `go test ./handlers -run 'TestNLAuthoring'`
- `npm run type-check`
- `npx playwright test e2e/builder-nl.spec.ts`
- `make agent-validate`

## Production Guardrails

- keep `preview-rulebased` as a documented escape hatch until a model-backed backend is proven stable
- avoid widening planner support subsets without matching deterministic DSL application coverage
- keep schema authority tied to the generated manifest pipeline
- record durable rollout gaps in [docs/agent/tech-debt/td-012-dashboard-nl-preview-planner-and-schema-authority-gap.md](../tech-debt/td-012-dashboard-nl-preview-planner-and-schema-authority-gap.md)
