# Multi-turn Session Affinity Productionization Plan

This execution plan tracks issue `#1513` and the related multi-turn routing stability work discussed in `#1439` and `#1458`. It is the durable work loop for turning session affinity from scattered selector/session primitives into a production-safe routing contract.

## Goal

- Productionize multi-turn session affinity so routed conversations remain stable across turns by default.
- Allow controlled model switching only when explicit signals, policy, or runtime availability justify it.
- Make the session-affinity contract visible through router state, headers, replay, metrics, dashboard surfaces, and E2E coverage.

## Scope

- `src/semantic-router/pkg/extproc/**` request routing flow and request context
- `src/semantic-router/pkg/selection/**` selection-context threading and selector integration
- `src/semantic-router/pkg/responseapi/**` and `src/semantic-router/pkg/responsestore/**` session identity touchpoints
- New `pkg/sessionaffinity/**` state, arbitration, and store abstractions
- `src/semantic-router/pkg/headers/**`, router replay, and selection/affinity metrics
- `dashboard/frontend/**` affinity visibility surfaces
- Local/E2E coverage for multi-turn stable routing and controlled switching

Out of scope for this plan:

- Replacing the existing selector families with a new routing algorithm
- Extending the public OpenAI-compatible request schema with new top-level affinity fields
- Redesigning memory extraction beyond the session-identity points needed for affinity

## Exit Criteria

- Requests with a trusted user identity and stable session key keep the same model within the same decision scope by default.
- Controlled switches occur only for named reasons such as decision changes, missing candidates, negative feedback, explicit bypass/reset, threshold-based overrides, or runtime availability failures.
- `/v1/responses` and chat-completions flows both have an explicit session-affinity contract and a tested disable path when trust requirements are not met.
- Multi-replica deployments can preserve affinity state through Redis without changing routing semantics.
- Response headers, router replay, metrics, dashboard surfaces, and E2E cases can all explain whether a turn stuck, switched, bypassed, or reset.

## Task List

- [ ] `SA001` Define the global session-affinity config and trusted identity contract.
  - Done when `IntelligentRouting` has a dedicated `session_affinity` block and the repo documents which session keys are trusted, namespaced, or rejected.
- [ ] `SA002` Thread `UserID`, `SessionID`, and `ConversationHistory` into selection context on the main extproc path.
  - Done when `selectModelFromCandidates(...)` builds a fully populated `selection.SelectionContext` instead of only passing query/category/candidates.
- [ ] `SA003` Add the `pkg/sessionaffinity` state model and arbitration interface.
  - Done when the repo has one canonical state shape, affinity key builder, arbitration result type, and rule evaluation entrypoint.
- [ ] `SA004` Implement `memory` and `redis` affinity stores with TTL, reset, and per-session scope indexing semantics.
  - Done when the store lifecycle can support `default`, `bypass`, `refresh`, and `reset` flows without orphaned state.
- [ ] `SA005` Insert affinity arbitration into the extproc auto-routing flow.
  - Done when selector output feeds a global affinity manager before endpoint routing, and the final outcome carries primary and fallback model choices.
- [ ] `SA006` Preserve availability by adding endpoint and provider fallback after affinity decisions.
  - Done when a stale or unavailable sticky model can gracefully fall back to another candidate instead of turning affinity into a hard failure.
- [ ] `SA007` Add request-scoped escape hatches and feedback-driven release rules.
  - Done when request headers or metadata can trigger `bypass`, `refresh`, and `reset`, and negative feedback signals can release a sticky binding under policy.
- [ ] `SA008` Expose affinity action and reason through router context, headers, replay, and metrics.
  - Done when operators can answer why a request stuck or switched without reading logs only.
- [ ] `SA009` Update dashboard reveal/display surfaces for affinity visibility.
  - Done when playground and header-driven UI surfaces can show affinity action/reason alongside selected model/decision data.
- [ ] `SA010` Add unit and integration coverage for keying, arbitration, feedback release, and availability fallback.
  - Done when targeted tests protect the state contract and the extproc routing path from regression.
- [ ] `SA011` Add or update E2E coverage for stable multi-turn routing and controlled switching.
  - Done when at least one affected E2E profile exercises multi-turn stability and at least one escape/switch path.
- [ ] `SA012` Close the loop with config/docs/reporting validation and decide whether a dedicated ADR or debt entry is required.
  - Done when the implementation lands with aligned docs, validation output, and any unresolved architecture gap recorded in the canonical governance layer.

## Current Loop

- Loop status: active as of `2026-03-13`.
- Current focus:
  - `SA001` lock the trust boundary and config surface before store or routing code lands
  - `SA002` thread session/user/history into the live selection path
  - `SA003` define the canonical arbitration/state interfaces before backend-specific store work
- Planned implementation order:
  1. Land the identity/config contract first so the feature cannot silently depend on untrusted session keys.
  2. Thread full selection context next so existing selector families can benefit from session-aware inputs immediately.
  3. Land arbitration and observability in the same loop so sticky behavior never ships as an opaque side effect.
  4. Add Redis-backed persistence and E2E coverage after the in-process path is deterministic.

## Decision Log

- Session affinity is a global routing-arbitration layer above selectors, not a selector-family-specific heuristic.
- The affinity scope is `{trusted_user_id, session_id, decision_name}` rather than one conversation-wide `bound_model`; decision scopes must not overwrite each other.
- `/v1/responses` keeps `conversation_id` as the canonical session identifier; the plan does not add new top-level public API fields for affinity.
- Non-Response API affinity requires a trusted user identity; if that trust boundary is absent, affinity must disable itself rather than guess.
- Availability outranks stickiness; endpoint/provider/user-key failures are valid switch reasons and must produce explainable fallbacks.
- Affinity state is tracked in a dedicated store abstraction instead of piggybacking implicitly on response retention internals.
- If implementation shows that affinity-state ownership versus response-store ownership is a durable architecture decision, record that in an ADR; if it remains an unresolved mismatch, record it in the indexed debt register.

## Follow-up Debt / ADR Links

- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../adr/README.md](../adr/README.md)
