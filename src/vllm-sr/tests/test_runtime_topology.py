from types import SimpleNamespace

from cli.consts import TOPOLOGY_MONOLITH, TOPOLOGY_SPLIT
from cli import runtime_stack
from cli.runtime_topology import (
    detect_runtime_topology,
    is_monolith_image,
    managed_container_names,
    split_service_names,
)


def test_detect_runtime_topology_defaults_to_split(monkeypatch):
    monkeypatch.delenv("VLLM_SR_RUNTIME_TOPOLOGY", raising=False)
    monkeypatch.delenv("VLLM_SR_IMAGE", raising=False)

    assert detect_runtime_topology() == TOPOLOGY_SPLIT


def test_detect_runtime_topology_uses_legacy_monolith_image(monkeypatch):
    monkeypatch.delenv("VLLM_SR_RUNTIME_TOPOLOGY", raising=False)

    assert (
        detect_runtime_topology("ghcr.io/vllm-project/semantic-router/vllm-sr:latest")
        == TOPOLOGY_MONOLITH
    )
    assert is_monolith_image("vllm-sr:dev")


def test_split_service_names_respect_runtime_modes():
    assert split_service_names(minimal=True) == ["router", "envoy"]
    assert split_service_names(setup_mode=True) == ["dashboard-db", "dashboard"]
    assert managed_container_names(minimal=True) == ["vllm-sr-router", "vllm-sr-envoy"]


def test_resume_split_runtime_after_setup_starts_missing_router_and_envoy(monkeypatch):
    calls = []
    network_connects = []
    started = {"vllm-sr-dashboard", "vllm-sr-dashboard-db"}
    state = SimpleNamespace(
        images=SimpleNamespace(
            router="router-image", envoy="envoy-image", dashboard_db="db-image"
        ),
        layout=SimpleNamespace(first_listener_port=8899),
    )

    monkeypatch.setattr(
        runtime_stack,
        "load_config",
        lambda path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(runtime_stack, "prepare_split_runtime", lambda **kwargs: state)
    monkeypatch.setattr(runtime_stack, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        runtime_stack,
        "docker_container_status",
        lambda name: "running" if name in started else "not found",
    )
    monkeypatch.setattr(
        runtime_stack,
        "_start_dashboard_db",
        lambda *args, **kwargs: (
            calls.append("dashboard-db"),
            started.add("vllm-sr-dashboard-db"),
        ),
    )
    monkeypatch.setattr(
        runtime_stack, "_wait_for_dashboard_db", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_stack,
        "_start_router",
        lambda *args, **kwargs: (calls.append("router"), started.add("vllm-sr-router")),
    )
    monkeypatch.setattr(
        runtime_stack,
        "_start_envoy",
        lambda *args, **kwargs: (calls.append("envoy"), started.add("vllm-sr-envoy")),
    )
    monkeypatch.setattr(
        runtime_stack,
        "docker_network_connect",
        lambda network_name, container_name: (
            network_connects.append((network_name, container_name)) or (0, "", "")
        ),
    )

    resumed = runtime_stack.resume_split_runtime_after_setup(
        "/tmp/config.yaml",
        env_vars={},
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
    )

    assert resumed is state
    assert calls == ["router", "envoy"]
    assert network_connects == [
        ("vllm-sr-network", "vllm-sr-dashboard"),
        ("vllm-sr-network", "vllm-sr-dashboard-db"),
        ("vllm-sr-network", "vllm-sr-router"),
        ("vllm-sr-network", "vllm-sr-envoy"),
    ]


def test_resume_split_runtime_after_setup_reconnects_running_services(monkeypatch):
    network_connects = []
    state = SimpleNamespace(
        images=SimpleNamespace(
            router="router-image", envoy="envoy-image", dashboard_db="db-image"
        ),
        layout=SimpleNamespace(first_listener_port=8899),
    )

    monkeypatch.setattr(
        runtime_stack,
        "load_config",
        lambda path: {
            "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}]
        },
    )
    monkeypatch.setattr(runtime_stack, "prepare_split_runtime", lambda **kwargs: state)
    monkeypatch.setattr(runtime_stack, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        runtime_stack, "docker_container_status", lambda _name: "running"
    )
    monkeypatch.setattr(
        runtime_stack,
        "_start_dashboard_db",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("db should not start")
        ),
    )
    monkeypatch.setattr(
        runtime_stack,
        "_wait_for_dashboard_db",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("db wait should not run")
        ),
    )
    monkeypatch.setattr(
        runtime_stack,
        "_start_router",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("router should not start")
        ),
    )
    monkeypatch.setattr(
        runtime_stack,
        "_start_envoy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("envoy should not start")
        ),
    )
    monkeypatch.setattr(
        runtime_stack,
        "docker_network_connect",
        lambda network_name, container_name: (
            network_connects.append((network_name, container_name)) or (0, "", "")
        ),
    )

    resumed = runtime_stack.resume_split_runtime_after_setup(
        "/tmp/config.yaml",
        env_vars={},
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
    )

    assert resumed is state
    assert network_connects == [
        ("vllm-sr-network", "vllm-sr-dashboard"),
        ("vllm-sr-network", "vllm-sr-dashboard-db"),
        ("vllm-sr-network", "vllm-sr-router"),
        ("vllm-sr-network", "vllm-sr-envoy"),
    ]


def test_start_dashboard_preserves_local_user_identity(monkeypatch):
    captured = {}
    state = SimpleNamespace(
        images=SimpleNamespace(
            router="router-image",
            envoy="envoy-image",
            dashboard="dashboard-image",
            dashboard_db="db-image",
        ),
        layout=SimpleNamespace(
            config_path="/tmp/workspace/config.yaml",
            config_dir="/tmp/workspace",
            dashboard_data_dir="/tmp/workspace/.vllm-sr/dashboard-data",
            first_listener_port=8899,
        ),
    )

    monkeypatch.setattr(runtime_stack, "_local_user_spec", lambda: "1234:4321")
    monkeypatch.setattr(
        runtime_stack,
        "_build_run_command",
        lambda **kwargs: captured.setdefault("kwargs", kwargs) or ["docker", "run"],
    )
    monkeypatch.setattr(runtime_stack, "_run_detached", lambda *args, **kwargs: None)

    runtime_stack._start_dashboard(
        "docker",
        state,
        {},
        network_name="vllm-sr-network",
        openclaw_network_name="vllm-sr-network",
    )

    kwargs = captured["kwargs"]
    assert kwargs["user"] == "1234:4321"
    assert kwargs["env_vars"]["VLLM_SR_PRESERVE_RUN_USER"] == "1"
    assert (
        kwargs["env_vars"]["VLLM_SR_HOST_CONFIG_PATH"] == "/tmp/workspace/config.yaml"
    )
    assert kwargs["env_vars"]["VLLM_SR_ENVOY_EXTPROC_HOST"] == "vllm-sr-router"
    assert kwargs["env_vars"]["VLLM_SR_ENVOY_ROUTER_API_HOST"] == "vllm-sr-router"
    assert kwargs["env_vars"]["TARGET_ROUTER_API_URL"] == "http://vllm-sr-router:8080"
    assert (
        kwargs["env_vars"]["TARGET_ROUTER_METRICS_URL"]
        == "http://vllm-sr-router:9190/metrics"
    )
    assert kwargs["env_vars"]["TARGET_ENVOY_URL"] == "http://vllm-sr-envoy:8899"
    assert "/tmp/workspace:/tmp/workspace:z" in kwargs["volumes"]
    assert "-router_api=http://vllm-sr-router:8080" in kwargs["command"]
    assert "-router_metrics=http://vllm-sr-router:9190/metrics" in kwargs["command"]
    assert "-envoy=http://vllm-sr-envoy:8899" in kwargs["command"]


def test_prepare_split_runtime_targets_envoy_at_router_service(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.1\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n"
    )

    captured = {}
    images = SimpleNamespace(
        router="router-image",
        envoy="envoy-image",
        dashboard="dashboard-image",
        dashboard_db="db-image",
    )

    monkeypatch.setattr(
        runtime_stack, "copy_defaults_reference", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runtime_stack, "generate_router_config", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runtime_stack, "parse_user_config", lambda *_args, **_kwargs: "parsed-config"
    )
    monkeypatch.setattr(
        runtime_stack, "resolve_split_runtime_images", lambda **_kwargs: images
    )
    monkeypatch.setattr(
        runtime_stack,
        "generate_envoy_config_from_user_config",
        lambda user_config, output_file, **kwargs: captured.update(
            {
                "user_config": user_config,
                "output_file": output_file,
                "kwargs": kwargs,
            }
        ),
    )

    state = runtime_stack.prepare_split_runtime(
        str(config_path),
        env_vars={},
        listeners=[{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        setup_mode=False,
    )

    assert state.images is images
    assert captured["user_config"] == "parsed-config"
    assert captured["kwargs"]["extproc_host"] == "vllm-sr-router"
    assert captured["kwargs"]["router_api_host"] == "vllm-sr-router"
