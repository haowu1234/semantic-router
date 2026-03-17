# vLLM-SR CLI Tests

End-to-end tests for the `vllm-sr` command-line interface.

## Quick Start

```bash
# From project root:
make vllm-sr-test              # Unit tests only (fast)
make vllm-sr-test-integration  # Unit + Integration tests
```

## Make Targets

| Target | Description | Requires |
|--------|-------------|----------|
| `make vllm-sr-test` | Run unit tests only | Python (bootstraps local `vllm-sr` CLI) |
| `make vllm-sr-test-integration` | Run unit + integration tests | Docker image (builds automatically) |

## Test Files

| File | Type | Description |
|------|------|-------------|
| `test_unit_serve.py` | Unit | Tests `serve` flags |
| `test_unit_lifecycle.py` | Unit | Tests `status/logs/stop/dashboard/config` flags |
| `test_integration.py` | **Integration** | Real container tests (strong validation) |
| `test_split_runtime_topology.py` | **Integration** | Split runtime communication topology tests |
| `cli_test_base.py` | Helper | Base class with utilities |
| `run_cli_tests.py` | Helper | Test runner |

## Integration Tests (Strong Validation)

These tests start real containers and verify with `docker inspect`:

| Test | What it verifies |
|------|------------------|
| `test_running_container_contracts` | canonical config → serve → container running → health |
| `test_env_var_passed_to_container` | HF_TOKEN inside container |
| `test_volume_mounting` | config.yaml + models/ mounted |
| `test_status_shows_running_container` | `status` reports running |
| `test_logs_retrieves_container_logs` | `logs` gets actual output |
| `test_stop_terminates_container` | `stop` actually stops container |
| `test_image_pull_policy_never_fails_with_missing_image` | `never` policy rejects missing image |
| `test_image_pull_policy_always_attempts_pull` | `always` policy attempts pull |

## Split Runtime Topology Tests

These tests validate the communication links between containers in split runtime mode:

| Test | What it verifies |
|------|------------------|
| `test_all_containers_on_same_network` | All containers connected to `vllm-sr-network` |
| `test_envoy_to_router_grpc_connectivity` | Envoy → Router extproc gRPC (port 50051) |
| `test_dashboard_to_router_api_connectivity` | Dashboard → Router API (port 8080) |
| `test_dashboard_to_envoy_connectivity` | Dashboard → Envoy listener/admin (ports 8899/9901) |
| `test_openclaw_model_base_url_points_to_envoy` | `OPENCLAW_MODEL_BASE_URL` → Envoy (not Dashboard) |
| `test_dashboard_to_postgres_connectivity` | Dashboard → PostgreSQL (port 5432) |
| `test_router_startup_before_envoy` | Router ready before Envoy starts |
| `test_envoy_listener_health` | Envoy listener is healthy and responding |
| `test_full_communication_chain` | End-to-end: External → Envoy → Router |

**Communication topology diagram:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Docker Network (vllm-sr-network)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  External ──▶ Envoy:8899 ──extproc──▶ Router:50051                  │
│                  │                        │                          │
│                  │                        │ API:8080                 │
│                  │                        │ Metrics:9190             │
│                  │                        ▼                          │
│                  └──────────────────▶ Dashboard:3000                │
│                                           │                          │
│                                           ▼                          │
│  OpenClaw ──▶ Envoy:8899/v1           PostgreSQL:5432               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Unit Tests (Flag Validation)

| Command | Options Tested |
|---------|----------------|
| `serve` | `--config`, `--image`, `--image-pull-policy`, `--readonly-dashboard` |
| `status` | `all`, `envoy`, `router`, `dashboard` |
| `logs` | `envoy`, `router`, `dashboard`, `-f/--follow` |
| `stop` | default |
| `dashboard` | default, `--no-open` |
| `config` | `envoy`, `router` |

## Running Tests

```bash
cd e2e/testing/vllm-sr-cli

# All unit tests
python run_cli_tests.py --verbose

# Include integration tests
python run_cli_tests.py --verbose --integration

# Filter by pattern
python run_cli_tests.py --pattern lifecycle
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUN_INTEGRATION_TESTS` | Set to `true` to enable integration tests |
| `CONTAINER_RUNTIME` | Override runtime (`docker` or `podman`) |
