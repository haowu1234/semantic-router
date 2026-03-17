# ADR 0005: Assign Config Compilation and Runtime Apply to a Runtime-Control Layer

## Status

Accepted

## Context

The current local configuration flow is split across several incompatible owners:

- the dashboard edits the user-facing `config.yaml`
- dashboard handlers directly invoke Python CLI helpers to regenerate `.vllm-sr/router-config.yaml`
- dashboard handlers also generate Envoy config and then use `supervisorctl` or `docker exec` into `vllm-sr-container` to restart services
- the router already has its own `/config/deploy`, `/config/rollback`, and fsnotify-based reload path, but that path operates on the flat router runtime config rather than the user-facing `config.yaml`

This is survivable inside the current all-in-one container because the dashboard, Python CLI, router, and Envoy all live in one runtime image. It breaks down once issue `#1508` splits router, dashboard, database, and Envoy into explicit services:

- the current dashboard image does not contain the Python CLI used for config regeneration
- `docker exec ... vllm-sr-container` and `supervisorctl` stop making sense once there is no single managed container
- router and dashboard risk competing to become the control-plane owner for config deploy or rollback

The repository needs one durable ownership model for how user config becomes runtime artifacts and how those artifacts are applied to router and Envoy in the split topology.

## Decision

Adopt the following ownership model for configuration generation and apply:

- The user-facing source of truth remains the workspace config surface:
  - `config.yaml`
  - `.vllm-sr/router-defaults.yaml` where applicable
- The runtime artifacts are separate derived outputs:
  - `.vllm-sr/router-config.yaml` for router runtime consumption
  - generated Envoy config for Envoy runtime consumption
- Translation from the user-facing workspace config into runtime artifacts belongs to a dedicated runtime-control layer owned by `vllm-sr serve`.
- The dashboard is a control-plane client, not the owner of config compilation or runtime service actions.
  - it may write or request changes to the user-facing config
  - it must call an explicit control contract to compile or apply those changes
  - it must not rely on `docker exec`, `supervisorctl`, or a hard-coded managed container
- The router runtime owns consumption and hot-reload of `router-config.yaml`, but it does not own translation from the user-facing workspace config.
- Envoy owns serving traffic from its generated config, but it does not own translation from the user-facing workspace config.
- Phase 1 may keep the existing Python CLI translation logic, but only inside the runtime-control layer.
  - Python-backed generation is an implementation detail of that control layer
  - it is not a justification for keeping the all-in-one runtime image
- The runtime-control layer must write compiled artifacts into a shared writable surface mounted into router and Envoy.
- The runtime-control layer must own the apply semantics for generated Envoy config.
  - the implementation may choose the exact reload mechanism later
  - restart, hot-reload, or rolling replacement behavior must be exposed through an explicit contract instead of container-local supervisor commands
- The router API's existing `/config/deploy` and `/config/rollback` endpoints remain valid for already-compiled router runtime config and for cluster-native flows that choose to use them.
  - they are not the authoritative local split-topology interface for dashboard-driven `config.yaml` changes

## Consequences

- Local runtime orchestration now needs an explicit runtime-control seam in addition to router, Envoy, dashboard, and database services.
- Dashboard handlers should converge away from direct Python CLI subprocess execution and managed-container restart logic.
- Router and Envoy can be kept narrower because they no longer need to absorb the user-config translation problem just to support dashboard deploy flows.
- The shared-volume contract for generated artifacts becomes a first-class part of the local topology.
- This ADR narrows TD001 and TD004 by clarifying control ownership, but both debt items remain open until the repo implements the new control layer and removes the legacy split behavior.
