from pathlib import Path

from cli.config_generator import generate_envoy_config_from_user_config
from cli.models import (
    Decision,
    Endpoint,
    Listener,
    Model,
    ModelRef,
    Providers,
    Rules,
    UserConfig,
)


def _build_user_config() -> UserConfig:
    return UserConfig(
        version="v0.1",
        listeners=[Listener(name="http-8899", address="0.0.0.0", port=8899)],
        decisions=[
            Decision(
                name="default-route",
                description="default route",
                priority=100,
                rules=Rules(),
                modelRefs=[ModelRef(model="test-model")],
            )
        ],
        providers=Providers(
            models=[
                Model(
                    name="test-model",
                    endpoints=[
                        Endpoint(
                            name="endpoint1",
                            weight=1,
                            endpoint="example.com:8000",
                            protocol="http",
                        )
                    ],
                )
            ],
            default_model="test-model",
        ),
    )


def _build_api_only_user_config() -> UserConfig:
    return UserConfig(
        version="v0.1",
        listeners=[Listener(name="http-8899", address="0.0.0.0", port=8899)],
        decisions=[
            Decision(
                name="default-route",
                description="default route",
                priority=100,
                rules=Rules(),
                modelRefs=[ModelRef(model="anthropic-model")],
            )
        ],
        providers=Providers(
            models=[
                Model(
                    name="anthropic-model",
                    api_format="anthropic",
                    endpoints=[],
                )
            ],
            default_model="anthropic-model",
        ),
    )


def test_generate_envoy_config_defaults_to_loopback_extproc(tmp_path):
    output_path = tmp_path / "envoy-config.yaml"

    generate_envoy_config_from_user_config(_build_user_config(), str(output_path))

    rendered = output_path.read_text()
    assert "type: STATIC" in rendered.split("extproc_service", 1)[1]
    assert "address: 127.0.0.1" in rendered
    assert "port_value: 50051" in rendered


def test_generate_envoy_config_uses_split_router_host(tmp_path):
    output_path = tmp_path / "envoy-config.yaml"

    generate_envoy_config_from_user_config(
        _build_user_config(),
        str(output_path),
        extproc_host="vllm-sr-router",
        router_api_host="vllm-sr-router",
    )

    rendered = output_path.read_text()
    extproc_cluster = rendered.split("extproc_service", 1)[1]
    assert "type: LOGICAL_DNS" in extproc_cluster
    assert 'hostname: "vllm-sr-router"' in extproc_cluster
    assert "address: vllm-sr-router" in rendered
    assert "port_value: 50051" in rendered
    assert "address: 127.0.0.1" not in rendered.split("extproc_service", 1)[1]


def test_generate_envoy_config_uses_split_router_host_for_api_only_configs(tmp_path):
    output_path = tmp_path / "envoy-config.yaml"

    generate_envoy_config_from_user_config(
        _build_api_only_user_config(),
        str(output_path),
        extproc_host="vllm-sr-router",
        router_api_host="vllm-sr-router",
    )

    rendered = output_path.read_text()
    fallback_cluster = rendered.split("vllm_static_cluster", 1)[1]
    assert "type: LOGICAL_DNS" in fallback_cluster
    assert 'hostname: "vllm-sr-router"' in fallback_cluster
    assert "port_value: 8080" in rendered
    assert "address: vllm-sr-router" in rendered
