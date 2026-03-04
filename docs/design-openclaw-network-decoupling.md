# Design: OpenClaw Network Decoupling from vllm-sr Container

## Status

Implemented (except Delete enhancement and uninstall command — deferred)

## Problem Statement

OpenClaw agent containers currently use `container:vllm-sr-container` network mode, sharing the
vllm-sr-container's network namespace. This creates a hard lifecycle coupling:

- **`vllm-sr serve` (restart)**: The old vllm-sr-container is removed, destroying the shared network
  namespace. OpenClaw containers lose all network connectivity (no external access, no reachability
  from the new Dashboard). They become orphaned on a dead network namespace.
- **`vllm-sr stop` + `vllm-sr serve`**: Same outcome — OpenClaw containers are stranded.
- **No recovery path**: There is no mechanism to reconnect orphaned OpenClaw containers after a
  vllm-sr restart.

### Current Architecture (Broken on Restart)

```
┌── vllm-sr-container (NetworkNamespace owner) ─────────────┐
│                                                            │
│   Dashboard (:8700)          Envoy (:50051)                │
│   Router API (:8080)                                       │
│                                                            │
│   ┌─ Shared NS (localhost) ──────────────────────────────┐ │
│   │  claude (:18788)   moss (:18789)   iris (:18790)     │ │
│   │  ↑ container:vllm-sr-container network mode          │ │
│   └──────────────────────────────────────────────────────┘ │
│                                                            │
│   eth0 → vllm-sr-network (bridge)                          │
└────────────────────────────────────────────────────────────┘
```

When vllm-sr-container is removed, the entire namespace (including OpenClaw containers' network
stack) is destroyed.

## Proposed Solution

Migrate OpenClaw containers from `container:vllm-sr-container` network mode to `vllm-sr-network`
bridge network mode. All containers (vllm-sr, observability, OpenClaw) share a single bridge
network with DNS-based service discovery.

### Target Architecture

```
┌────── vllm-sr-network (bridge) ──────────────────────────┐
│                                                           │
│  vllm-sr-container  (172.18.0.2)  ← Dashboard/Router     │
│  vllm-sr-jaeger     (172.18.0.3)                          │
│  vllm-sr-prometheus (172.18.0.4)                          │
│  vllm-sr-grafana    (172.18.0.5)                          │
│  claude             (172.18.0.6)  ← OpenClaw agent        │
│  moss               (172.18.0.7)  ← OpenClaw agent        │
│  iris               (172.18.0.8)  ← OpenClaw agent        │
│                                                           │
│  DNS: claude → 172.18.0.6, moss → 172.18.0.7, etc.       │
└───────────────────────────────────────────────────────────┘
```

### Key Properties

1. **OpenClaw containers have their own network namespace** — removing vllm-sr-container does not
   affect them.
2. **Docker DNS** resolves container names within the bridge network — no hardcoded IPs needed.
3. **Single network** — no need for a second `openclaw-network`; simpler to manage.

## Lifecycle Design

### `vllm-sr serve` (Start / Restart)

```
vllm-sr serve
  │
  ├── 1. Check existing vllm-sr-container → stop + rm if found
  │
  ├── 2. docker network create vllm-sr-network   (idempotent, skips if exists)
  │
  ├── 3. Start observability stack (stop+rm old → run new)
  │      Jaeger, Prometheus, Grafana → --network vllm-sr-network
  │
  ├── 4. docker run vllm-sr-container --network vllm-sr-network
  │
  └── 5. Recover OpenClaw containers (NEW)
         for each entry in containers.json:
           if container exists and stopped:
             docker network connect vllm-sr-network {name}   (idempotent)
             docker start {name}
           if container exists and running:
             docker network connect vllm-sr-network {name}   (idempotent, no-op if already connected)
           if container does not exist:
             mark as deleted in registry
```

**Direct `vllm-sr serve` (without prior stop):**
- Step 2 is idempotent — network already exists, skip.
- OpenClaw containers are already on `vllm-sr-network` — `docker network connect` is a no-op.
- Result: OpenClaw containers remain fully connected throughout. **Zero downtime for agents.**

### `vllm-sr stop`

```
vllm-sr stop
  │
  ├── 1. Stop OpenClaw containers (NEW) — do NOT remove
  │      for each entry in containers.json:
  │        docker stop {name}
  │        docker network disconnect vllm-sr-network {name}
  │
  ├── 2. Stop + remove vllm-sr-container (existing)
  │
  ├── 3. Stop + remove observability containers (existing)
  │      Jaeger, Prometheus, Grafana
  │
  └── 4. docker network rm vllm-sr-network (existing, now clean)
```

**Why disconnect before removing the network?**
- `docker network rm` fails if active endpoints exist.
- Disconnecting stopped OpenClaw containers first ensures clean removal.
- On next `vllm-sr serve`, containers are reconnected (Step 5 above).

### `vllm-sr stop` + `vllm-sr serve` (Full Restart)

```
stop:
  ├── stop OpenClaw → disconnect from vllm-sr-network
  ├── rm vllm-sr + observability
  └── rm vllm-sr-network                    ← clean

serve:
  ├── create vllm-sr-network (new)
  ├── run vllm-sr + observability
  └── reconnect + start OpenClaw containers  ← automatic recovery
```

OpenClaw containers are preserved (stopped but not removed). Data, volumes, and workspace
directories are intact. Users see their agents automatically restored after restart.

### Dashboard Provision (New Agent)

```
POST /api/openclaw/provision
  │
  ├── Prepare workspace, config, identity files
  │
  ├── docker run -d --name {name} --network vllm-sr-network \  ← CHANGED
  │     -v workspace:/config -v state-vol:/state ...
  │
  └── Health check → return status
```

### Dashboard Delete (Remove Agent)

```
DELETE /api/openclaw/containers/{name}
  │
  ├── docker rm -f {name}
  ├── Remove workspace directory
  ├── docker volume rm openclaw-state-{name}
  └── Remove from containers.json registry
```

### `vllm-sr uninstall` / `vllm-sr purge` (Full Cleanup)

A new command for complete teardown:

```
vllm-sr uninstall
  │
  ├── 1. Delete all OpenClaw containers + workspace + volumes
  │      for each in containers.json:
  │        docker rm -f {name}
  │        rm -rf {dataDir}/containers/{name}/
  │        docker volume rm openclaw-state-{name}
  │
  ├── 2. Standard stop logic
  │      rm vllm-sr-container + observability
  │      rm vllm-sr-network
  │
  ├── 3. Clean registry
  │      rm containers.json, teams.json
  │
  └── 4. Optional: rm -rf {dataDir}/
```

## Operation Matrix

| Operation | vllm-sr-container | Observability | OpenClaw Containers | vllm-sr-network | Data |
|-----------|:-:|:-:|:-:|:-:|:-:|
| `vllm-sr serve` | Create | Create | **Recover** | Create (idempotent) | Preserve |
| `vllm-sr stop` | Remove | Remove | **Stop + Disconnect** | **Remove** | Preserve |
| `vllm-sr serve` (no prior stop) | Recreate | Recreate | **Untouched** | Exists (skip) | Preserve |
| Dashboard Provision | — | — | Create | — | Create |
| Dashboard Stop | — | — | Stop | — | Preserve |
| Dashboard Start | — | — | Start | — | Preserve |
| Dashboard Delete | — | — | Remove | — | **Clean** |
| `vllm-sr uninstall` | Remove | Remove | **Remove** | **Remove** | **Clean** |

## Code Changes Required

### 1. CLI Layer — `docker_cli.py`

**Change `OPENCLAW_DEFAULT_NETWORK_MODE`** from container mode to network name:

```python
# Before (line 443-447):
env_vars.setdefault(
    "OPENCLAW_DEFAULT_NETWORK_MODE",
    f"container:{VLLM_SR_DOCKER_NAME}",
)

# After:
env_vars.setdefault(
    "OPENCLAW_DEFAULT_NETWORK_MODE",
    network_name or "vllm-sr-network",
)
```

**Add function to load OpenClaw registry** (for stop/serve lifecycle):

```python
def load_openclaw_registry(data_dir: str) -> list[dict]:
    """Load OpenClaw container entries from containers.json."""
    registry_path = os.path.join(data_dir, "containers.json")
    if not os.path.exists(registry_path):
        return []
    with open(registry_path) as f:
        return json.load(f)
```

**Add `docker_network_disconnect` function:**

```python
def docker_network_disconnect(network_name: str, container_name: str) -> tuple:
    """Disconnect a container from a Docker network."""
    runtime = get_container_runtime()
    return run_command([runtime, "network", "disconnect", network_name, container_name])
```

**Add `docker_network_connect` function:**

```python
def docker_network_connect(network_name: str, container_name: str) -> tuple:
    """Connect a container to a Docker network (idempotent)."""
    runtime = get_container_runtime()
    return run_command([runtime, "network", "connect", network_name, container_name])
```

### 2. CLI Layer — `core.py`

**`stop_vllm_sr()`** — Add OpenClaw graceful shutdown before network removal:

```python
def stop_vllm_sr():
    # ... existing container status check ...

    # NEW: Stop and disconnect OpenClaw containers
    openclaw_entries = load_openclaw_registry(openclaw_data_dir)
    for entry in openclaw_entries:
        name = entry.get("containerName") or entry.get("name")
        if not name:
            continue
        status = docker_container_status(name)
        if status == "running":
            log.info(f"Stopping OpenClaw container: {name}")
            docker_stop_container(name)
        if status != "not found":
            log.info(f"Disconnecting {name} from vllm-sr-network")
            docker_network_disconnect("vllm-sr-network", name)

    # ... existing: stop + rm vllm-sr-container ...
    # ... existing: stop + rm observability containers ...
    # ... existing: docker network rm vllm-sr-network ...
```

**`start_vllm_sr()`** — Add OpenClaw recovery after container startup:

```python
def start_vllm_sr():
    # ... existing: create network, start observability, start vllm-sr-container ...
    # ... existing: health check ...

    # NEW: Recover OpenClaw containers
    openclaw_entries = load_openclaw_registry(openclaw_data_dir)
    for entry in openclaw_entries:
        name = entry.get("containerName") or entry.get("name")
        if not name:
            continue
        status = docker_container_status(name)
        if status == "not found":
            log.warning(f"OpenClaw container {name} no longer exists, skipping")
            continue
        # Reconnect to network (idempotent)
        rc, _, _ = docker_network_connect("vllm-sr-network", name)
        if rc == 0:
            log.info(f"Connected {name} to vllm-sr-network")
        # Start if stopped
        if status != "running":
            log.info(f"Starting OpenClaw container: {name}")
            docker_start_container(name)
```

### 3. Backend — `openclaw_provision.go`

**Change network mode interpretation:**

```go
// Before (line 49-59): Uses "container:xxx" as network mode
// After: Uses "vllm-sr-network" (bridge network name)

preferredNetworkMode := os.Getenv("OPENCLAW_DEFAULT_NETWORK_MODE")
// Value is now "vllm-sr-network" instead of "container:vllm-sr-container"
```

**Change `docker run` command (line 180-216):**

```go
// Before:
//   docker run -d --name X --network container:vllm-sr-container ...
// After:
//   docker run -d --name X --network vllm-sr-network ...
```

No structural code change needed here — the `--network` flag already uses the `NetworkMode`
variable. The change is purely in the environment variable value passed from the CLI.

### 4. Backend — `openclaw_status.go`

**Change gateway host resolution to use container names:**

```go
// Before (line 61-91): Candidates are 127.0.0.1, host.docker.internal, etc.
// After: Primary candidate is the container name itself (DNS resolution via bridge network)

func (h *OpenClawHandler) gatewayHostCandidates(containerName string) []string {
    candidates := []string{}

    // First priority: container name (DNS within bridge network)
    if containerName != "" {
        candidates = append(candidates, containerName)
    }

    // Fallback: environment variable overrides
    if host := os.Getenv("OPENCLAW_GATEWAY_HOST"); host != "" {
        candidates = append(candidates, host)
    }

    // Fallback: localhost (for development / non-Docker environments)
    candidates = append(candidates, "127.0.0.1")

    return candidates
}
```

**Update `resolveGatewayHost()` and `gatewayReachable()` to accept container name parameter.**

### 5. Backend — `openclaw_lifecycle.go`

**Update `TargetBaseForContainer()`** to use container name as hostname:

```go
// Before (line 137-159): Uses gatewayBaseURL(port) which resolves to 127.0.0.1:port
// After: Uses container name directly

func (h *OpenClawHandler) TargetBaseForContainer(containerName string) (string, error) {
    port, err := h.PortForContainer(containerName)
    if err != nil {
        return "", err
    }
    return fmt.Sprintf("http://%s:%d", containerName, port), nil
}
```

### 6. Backend — `openclaw_helpers.go`

**Update `generateDockerRunCmd()` and `generateComposeYAML()`:**

The `NetworkMode` field will now contain `vllm-sr-network` instead of `container:xxx`. The
generated commands will naturally produce:

```bash
docker run -d --name claude --network vllm-sr-network ...
```

```yaml
services:
  claude:
    network_mode: vllm-sr-network  # bridge network, not container:xxx
```

Note: For compose YAML, `network_mode` should be replaced with `networks:` syntax:

```yaml
services:
  claude:
    networks:
      - vllm-sr-network
networks:
  vllm-sr-network:
    external: true
```

### 7. Backend — `router.go`

**WebSocket proxy target resolution (line 381-416):**

No structural change needed. `TargetBaseForContainer()` now returns
`http://{containerName}:{port}` instead of `http://127.0.0.1:{port}`. The proxy handler
works the same way.

### 8. Backend — `openclaw_lifecycle.go` — Delete Enhancement

**Add workspace and volume cleanup to `deleteContainerByName()`:**

```go
func (h *OpenClawHandler) deleteContainerByName(name string) error {
    // Existing: docker rm -f
    h.containerRun("rm", "-f", name)

    // NEW: Clean workspace directory
    workDir := filepath.Join(h.dataDir, "containers", name)
    os.RemoveAll(workDir)

    // NEW: Clean Docker volume
    h.containerRun("volume", "rm", "-f", fmt.Sprintf("openclaw-state-%s", name))

    // Existing: remove from registry
    return h.removeFromRegistry(name)
}
```

## Port Exposure Consideration

With bridge network mode, OpenClaw containers have their own network namespace. If external access
to agent ports is needed (e.g., for debugging), explicit port mapping is required:

```bash
# Container mode (before): All ports visible on localhost (shared NS)
docker run --network container:vllm-sr-container ...

# Bridge mode (after): Ports are internal to the bridge network
# Dashboard accesses via DNS (http://claude:18788) — no host port mapping needed
# For debugging, optionally expose:
docker run --network vllm-sr-network -p 18788:18788 ...
```

In normal operation, **no port mapping is needed** — Dashboard and the WebSocket proxy access
agents via the bridge network DNS. Port mapping is only needed if users want to access agents
directly from the host.

## Migration Strategy

### For Existing Deployments

On first `vllm-sr serve` after upgrade:

1. Detect existing OpenClaw containers with `container:*` network mode.
2. Stop them, recreate with `--network vllm-sr-network`.
3. Alternatively, document that existing agents need re-provisioning after upgrade.

### For New Deployments

No migration needed — new agents are created with bridge network mode from the start.

## Testing Plan

1. **Unit tests**: Update `openclaw_test.go` to verify bridge network mode in generated commands.
2. **Integration tests**:
   - Provision agent → verify on `vllm-sr-network`.
   - `vllm-sr stop` + `vllm-sr serve` → verify agent auto-recovery.
   - Direct `vllm-sr serve` (restart) → verify agent connectivity preserved.
   - Dashboard → agent WebSocket proxy works via container name DNS.
   - Agent → external API (LLM providers) works via bridge NAT.
3. **E2E tests**: Full workflow — provision, chat, restart vllm-sr, chat again.
