# ADR 0003: Adopt a Split Local Runtime Topology for vLLM-SR

## Status

Accepted

## Context

The repository currently has two conflicting runtime stories:

- the local `vllm-sr` path centers on one all-in-one image and one managed container that embeds router, Envoy, dashboard assets, and control logic
- the Kubernetes and operator paths already model router, Envoy, and dashboard as separate services or containers

Issue `#1508` asks the repository to split `vllm-sr` into a router-only runtime with external service boundaries for dashboard, database, and Envoy.

Without a durable architectural decision, implementation work would risk solving this several different ways at once:

- keeping `vllm-sr` as a monolith while adding more sidecar exceptions
- treating `extproc` and `vllm-sr` as separate long-term product surfaces with overlapping ownership
- preserving dashboard control-plane features by continuing to `docker exec` into a managed container even after the topology is nominally split

The repository needs one explicit target topology and naming contract before changing Dockerfiles, CLI orchestration, dashboard control paths, or release automation.

## Decision

Adopt the following runtime-topology decisions for the `vllm-sr` workstream:

- The canonical local runtime architecture is a split-service topology, not a supervisor-managed all-in-one container.
- `vllm-sr serve` remains the primary end-user entrypoint, but it must orchestrate explicit peer services for:
  - router runtime
  - Envoy
  - dashboard
  - dashboard persistence through an explicit database or persistence boundary
  - optional observability services
- `vllm-sr` remains the canonical user-facing product and CLI name.
- The `vllm-sr` image family must converge to router-only responsibility. New local-runtime work must not depend on a distinct all-in-one `vllm-sr` product image.
- `extproc` may remain temporarily as a compatibility or deployment-facing alias while manifests and downstream consumers migrate, but it is not the source of truth for a separate long-term product surface.
- Dashboard integration with router and Envoy must use explicit service contracts, shared volumes, or dedicated control adapters instead of assuming:
  - `supervisorctl`
  - `/var/log/supervisor/*`
  - hard-coded `docker exec` into `vllm-sr-container`
- The router-only image boundary and the config-contract simplification are separate tracks.
  - Phase 1 may retain Python-backed config generation where still required.
  - Phase 1 may not preserve that generation by hiding it inside a monolithic runtime container.
- Setup mode may start a reduced subset of services, but it must still follow the split topology and may not re-establish the all-in-one image as the primary local architecture.

## Consequences

- CLI lifecycle code, dashboard status or logs or deploy behavior, Docker build or publish flows, and smoke or E2E coverage all need coordinated updates.
- The existing split Kubernetes and operator deployment model becomes the reference architecture for local-runtime convergence.
- Long-term duplication between `extproc` and `vllm-sr` now counts as architecture drift unless it exists for bounded compatibility reasons.
- This ADR intentionally does not choose the final database engine or migration path.
  - That remains execution-plan work under `R002`.
  - What is fixed here is that persistence must be an explicit boundary rather than an accidental side effect of the router image.
- This ADR intentionally does not require phase 1 to eliminate Python-backed config generation.
  - What is fixed here is that config generation can no longer rely on the all-in-one managed container model as the long-term architecture.
