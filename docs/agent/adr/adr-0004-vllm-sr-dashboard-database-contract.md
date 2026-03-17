# ADR 0004: Use a Dedicated Dashboard Database Contract in the Split Runtime

## Status

Accepted

## Context

Issue `#1508` requires the split `vllm-sr` topology to make dashboard, database, and Envoy explicit services or boundaries instead of hiding them inside the current all-in-one runtime image.

Today the dashboard persists several different kinds of state in one implicit local filesystem surface:

- auth users, roles, and audit records in `./data/auth.db`
- evaluation tasks and history in `./data/evaluations.db`
- evaluation exports in `./data/results`
- ML pipeline artifacts in `./data/ml-pipeline`
- OpenClaw registry and workspace state under `./data/openclaw`

That state is currently coupled to the monolithic local runtime:

- local `vllm-sr serve` bind-mounts `.vllm-sr/dashboard-data` into the all-in-one container at `/app/data`
- the dashboard backend only supports direct SQLite file paths for auth and evaluation state
- the Kubernetes and OpenShift dashboard manifests do not yet model persistent auth or evaluation storage as a first-class runtime dependency

Without a durable decision, the repository risks splitting router, dashboard, and Envoy while leaving the database requirement ambiguous or accidental:

- treating SQLite files inside the dashboard container as if they already satisfy the "database service" requirement
- trying to move all dashboard artifacts into a database even when they are better represented as files
- allowing local and Kubernetes paths to diverge again on persistence ownership

## Decision

Adopt the following dashboard database contract for the `vllm-sr` split-runtime workstream:

- The dashboard owns its own persistence boundary. Router and Envoy do not own dashboard auth or evaluation state.
- Dashboard state is divided into two classes:
  - relational control-plane state, which belongs in a database
  - file artifacts, which remain on a writable dashboard-owned filesystem surface
- The relational control-plane state includes:
  - auth users
  - roles and permissions
  - auth audit records
  - evaluation task metadata
  - evaluation result metadata and history
- The file-artifact surface includes:
  - evaluation exports and raw result bundles
  - ML pipeline inputs and outputs
  - OpenClaw registry, team, room, and workspace files
- The canonical split runtime must treat the dashboard database as an explicit peer service, not an implicit file inside the router image.
- Postgres is the target database backend for the canonical split runtime.
  - local `vllm-sr serve` should eventually provision a `dashboard-db` peer service
  - Kubernetes and OpenShift paths should eventually point the dashboard at an in-cluster or externally managed Postgres service
- SQLite remains a bounded compatibility mode only.
  - it may continue to exist for unit tests, migration import, or short-lived single-process development
  - it is not the steady-state database contract for issue `#1508`
  - it must not be presented as the canonical split-topology database story
- The runtime contract must expose database connectivity through an explicit configuration surface rather than hard-coded file paths.
  - the implementation may choose the exact variable names later
  - phase `R004` and `R005` must wire the CLI, dashboard runtime, and manifests to that contract
- Existing `.vllm-sr/dashboard-data/auth.db` and `.vllm-sr/dashboard-data/evaluations.db` files are migration inputs, not long-term architecture.
  - the implementation loop must provide migration, import, or an explicit user-visible reset decision before claiming full closure

## Consequences

- Dashboard backend code needs a storage seam so auth and evaluation features can move off direct SQLite-only assumptions.
- The local CLI orchestration model now needs a real database service lifecycle in addition to router, Envoy, and dashboard.
- The current Kubernetes and OpenShift dashboard manifests are now explicitly incomplete for the desired steady state because they do not model durable dashboard database storage.
- File artifacts remain outside the database by design, so the split runtime still needs a writable dashboard data volume in addition to the database service.
- This ADR narrows TD005 by choosing the target persistence model, but TD005 remains open until the repo implements the new database and session architecture.
