# vLLM-SR Runtime Topology Split Execution Plan

This document tracks the long-horizon execution loop for issue `#1508`, which proposes splitting `vllm-sr` into a router-only runtime with explicit external service boundaries for dashboard, database, and Envoy.

## Goal

- Converge the local `vllm-sr` runtime packaging and orchestration model on an explicit multi-service topology instead of the current all-in-one supervisor container.
- Make the router runtime, dashboard runtime, database persistence, and Envoy deployment boundaries explicit enough that they can be versioned, validated, and operated independently.
- Preserve the user-facing `vllm-sr serve` product experience where practical while retiring the monolithic runtime assumptions behind it.

## Scope

- Local Docker or Podman runtime orchestration under `src/vllm-sr/**`
- `vllm-sr` image responsibilities, default image selection, and local developer build flow
- Dashboard backend and frontend contracts that currently assume a managed single container
- Runtime config generation and propagation for router and Envoy
- Docker build or publish workflows, docs, smoke coverage, and affected E2E coverage for the new topology
- Alignment with the already split Kubernetes and operator deployment model where that path can serve as the reference architecture

Out of scope for this plan unless required by the split:

- Routing algorithm changes
- Dashboard UX redesign unrelated to runtime topology
- Non-topology router behavior changes
- Replacing existing Kubernetes deployment patterns that already model router, Envoy, and dashboard as separate services

## Exit Criteria

- The default local runtime no longer depends on one `vllm-sr-container` that embeds router, Envoy, dashboard, and control logic under `supervisord`.
- The router runtime image has a single dominant responsibility and no longer carries dashboard assets or an embedded Envoy runtime as product defaults.
- Dashboard functionality that currently works in the all-in-one layout still works in the split topology, including setup mode, config editing, status, logs, and OpenClaw behavior or a documented equivalent.
- The database contract is explicit: the repository defines what service or persistence model owns dashboard auth and evaluation state, and whether any interim file-backed mode remains supported.
- `vllm-sr serve`, `status`, `logs`, `dashboard`, and `stop` operate against the split topology with stable service discovery, health, and teardown behavior.
- Build or publish automation, smoke flows, CLI integration tests, and affected E2E coverage validate the new topology as the canonical local path.
- Existing technical debt items about config fragmentation, config portability, and Python CLI versus Kubernetes workflow drift are retired or materially narrowed by the same workstream.

## Task List

- [x] `R001` Freeze the target topology and image naming contract.
  - Done when the repo defines the canonical local runtime graph, names the owned images, and documents compatibility behavior for the existing `vllm-sr` and `extproc` artifacts.
- [x] `R002` Decide the database ownership and persistence contract.
  - Done via ADR `0004`, which defines dashboard-owned persistence, Postgres as the canonical split-runtime database backend, and SQLite as compatibility-only.
- [x] `R003` Define the config-generation ownership model for router and Envoy.
  - Done via ADR `0005`, which assigns compilation and apply responsibilities to a runtime-control layer rather than dashboard container internals or the monolithic runtime.
- [ ] `R004` Refactor the local CLI orchestration model to manage multiple services.
  - Done when `vllm-sr serve`, `stop`, `status`, `logs`, and `dashboard` work against separate router, Envoy, dashboard, and database services with shared networks and volumes.
- [ ] `R005` Decouple dashboard control-plane features from single-container assumptions.
  - Done when config deploy or rollback, runtime status, runtime logs, and service health no longer depend on `supervisorctl`, `/var/log/supervisor/*`, or the hard-coded `vllm-sr-container`.
- [ ] `R006` Preserve or deliberately reshape setup mode and OpenClaw behavior in the split topology.
  - Done when first-run setup, dashboard-only boot, and OpenClaw runtime integration either work through the new topology or are narrowed behind explicit product decisions and documentation.
- [ ] `R007` Converge build, publish, and local developer workflows on the split topology.
  - Done when Dockerfiles, `tools/make/docker.mk`, image defaults, and release workflows all reflect one intentional ownership model instead of parallel monolith and split-product paths.
- [ ] `R008` Add validation and rollout coverage for the new topology.
  - Done when local smoke, CLI integration coverage, and affected E2E checks validate the split topology as the default path and the docs no longer present the all-in-one runtime as the primary behavior.
- [ ] `R009` Retire or materially narrow the related debt entries and record any new durable architecture decisions.
  - Done when the debt register and ADR set reflect the new steady-state architecture instead of leaving the topology split only in issue text or chat.

## Current Loop

- Loop status: active as of 2026-03-13.
- Completed in this loop: repository analysis, gap inventory, creation of this execution plan, acceptance of ADR `0003` for the split local runtime topology and naming contract, acceptance of ADR `0004` for the dashboard database contract, acceptance of ADR `0005` for config-compilation ownership, and the first dashboard backend runtime-control extraction under `dashboard/backend/runtimecontrol`.
- Active focus for the first implementation loop:
  - `R004` local multi-service orchestration seam design
  - `R005` dashboard control-plane decoupling from single-container assumptions
- Expected first code-loop output:
  - a durable decision on how config generation moves out of the monolithic container without regressing dashboard deploy or rollback behavior
- Known blockers:
  - the repo still has both `extproc` and `vllm-sr` image families with overlapping implementations that now need migration sequencing

## Decision Log

- Use the existing split Kubernetes and operator topology as the architectural reference rather than extending the current local all-in-one container.
- ADR `0003` now records the durable topology and naming decision for this workstream.
- ADR `0004` now records dashboard-owned persistence, Postgres as the target split-runtime database backend, and SQLite as compatibility-only.
- ADR `0005` now records that config compilation and runtime apply belong to a dedicated runtime-control layer rather than dashboard-managed container internals.
- The first code slice now routes dashboard config propagation, service-log access, setup restarts, and container-status checks through `dashboard/backend/runtimecontrol` instead of embedding that command logic directly in multiple handlers.
- Treat image-boundary cleanup and config-contract unification as related but separate concerns; a router-only image does not automatically remove the need for Python-backed config generation.
- Prefer preserving the user-facing `vllm-sr serve` workflow while changing the implementation behind it; a product-level topology split should not force an unnecessary CLI UX split.
- Avoid broadening dashboard runtime privileges beyond what is already required for OpenClaw unless the repo makes that tradeoff explicit.

## Follow-up Debt / ADR Links

- [issue #1508](https://github.com/vllm-project/semantic-router/issues/1508)
- [../adr/adr-0003-vllm-sr-runtime-topology-split.md](../adr/adr-0003-vllm-sr-runtime-topology-split.md)
- [../adr/adr-0004-vllm-sr-dashboard-database-contract.md](../adr/adr-0004-vllm-sr-dashboard-database-contract.md)
- [../adr/adr-0005-vllm-sr-runtime-control-and-config-ownership.md](../adr/adr-0005-vllm-sr-runtime-control-and-config-ownership.md)
- [../tech-debt/td-001-config-surface-fragmentation.md](../tech-debt/td-001-config-surface-fragmentation.md)
- [../tech-debt/td-002-config-portability-gap-local-vs-k8s.md](../tech-debt/td-002-config-portability-gap-local-vs-k8s.md)
- [../tech-debt/td-004-python-cli-kubernetes-workflow-separation.md](../tech-debt/td-004-python-cli-kubernetes-workflow-separation.md)
- [../tech-debt/td-005-dashboard-enterprise-console-foundations.md](../tech-debt/td-005-dashboard-enterprise-console-foundations.md)
- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../adr/README.md](../adr/README.md)
