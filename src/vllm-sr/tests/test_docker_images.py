from cli import docker_images


def test_resolve_split_runtime_images_defaults_blank_pull_policy(monkeypatch):
    recorded = []

    monkeypatch.setattr(
        docker_images,
        "_resolve_router_image",
        lambda image, normalized_platform: "router-image",
    )
    monkeypatch.setattr(
        docker_images,
        "_ensure_image_available",
        lambda image, pull_policy: recorded.append((image, pull_policy)),
    )

    docker_images.resolve_split_runtime_images(
        pull_policy="",
        include_router=True,
        include_dashboard=False,
        include_envoy=True,
        include_dashboard_db=False,
    )

    assert recorded == [
        ("router-image", docker_images.DEFAULT_IMAGE_PULL_POLICY),
        (
            docker_images.VLLM_SR_ENVOY_IMAGE_DEFAULT,
            docker_images.DEFAULT_IMAGE_PULL_POLICY,
        ),
    ]
