"""Container image resolution helpers for vLLM Semantic Router."""

import os
import sys
from dataclasses import dataclass

from cli.consts import (
    DEFAULT_IMAGE_PULL_POLICY,
    IMAGE_PULL_POLICY_ALWAYS,
    IMAGE_PULL_POLICY_IF_NOT_PRESENT,
    IMAGE_PULL_POLICY_NEVER,
    PLATFORM_AMD,
    VLLM_SR_DASHBOARD_DB_IMAGE_DEFAULT,
    VLLM_SR_DASHBOARD_IMAGE_DEFAULT,
    VLLM_SR_DOCKER_IMAGE_DEFAULT,
    VLLM_SR_DOCKER_IMAGE_ROCM,
    VLLM_SR_ENVOY_IMAGE_DEFAULT,
    VLLM_SR_MONOLITH_IMAGE_DEFAULT,
    VLLM_SR_MONOLITH_IMAGE_ROCM,
)
from cli.docker_runtime import (
    docker_image_exists,
    docker_pull_image,
    get_container_runtime,
)
from cli.utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class SplitRuntimeImages:
    """Container images used by the split local runtime."""

    router: str
    dashboard: str
    envoy: str
    dashboard_db: str


def _normalize_platform(platform):
    """Normalize platform input for comparisons."""
    if platform is None:
        return ""
    return str(platform).strip().lower()


def _resolve_platform_hint(platform):
    return _normalize_platform(platform) or _normalize_platform(
        os.getenv("VLLM_SR_PLATFORM")
    )


def _is_rocm_image(image_name):
    """Return True when the image name appears to be a ROCm image variant."""
    if not image_name:
        return False
    return "rocm" in image_name.lower()


def _derive_rocm_variant(image_name, base_image):
    """Return a ROCm variant for official image references."""
    if not image_name:
        return ""

    image_name = image_name.strip()
    default_repo = base_image.rsplit(":", 1)[0]
    if not image_name.startswith(default_repo):
        return ""

    if image_name == default_repo:
        return f"{default_repo}-rocm:latest"
    if image_name.startswith(f"{default_repo}:"):
        tag = image_name.split(":")[-1]
        return f"{default_repo}-rocm:{tag}"
    return ""


def _maybe_upgrade_to_rocm_image(
    selected_image, normalized_platform, source_name, default_image
):
    if normalized_platform != PLATFORM_AMD or _is_rocm_image(selected_image):
        return selected_image

    rocm_variant = _derive_rocm_variant(selected_image, default_image)
    if rocm_variant:
        log.warning(
            f"Platform 'amd' selected with non-ROCm official {source_name}. "
            f"Switching to ROCm image: {rocm_variant}"
        )
        return rocm_variant

    log.warning(
        f"Platform 'amd' selected but {source_name} does not look like a ROCm image. "
        "GPU acceleration may not be enabled. Prefer a '*-rocm' image."
    )
    return selected_image


def _ensure_image_available(selected_image, pull_policy):
    image_exists = docker_image_exists(selected_image)
    if pull_policy == IMAGE_PULL_POLICY_ALWAYS:
        _pull_or_exit(selected_image)
        return
    if pull_policy == IMAGE_PULL_POLICY_IF_NOT_PRESENT:
        if image_exists:
            log.info(f"Image exists locally: {selected_image}")
            return
        log.info("Image not found locally, pulling...")
        _pull_or_exit(selected_image, show_not_found=True)
        return
    if pull_policy == IMAGE_PULL_POLICY_NEVER and not image_exists:
        log.error(f"Image not found locally: {selected_image}")
        log.error("Pull policy is 'never', cannot pull image")
        _show_image_not_found_error(selected_image)
        sys.exit(1)
    if image_exists:
        log.info(f"Image exists locally: {selected_image}")


def _pull_or_exit(selected_image, show_not_found=False):
    if docker_pull_image(selected_image):
        return
    log.error(f"Failed to pull image: {selected_image}")
    if show_not_found:
        _show_image_not_found_error(selected_image)
    sys.exit(1)


def _resolve_router_image(image, normalized_platform):
    if image:
        log.info(f"Using specified router image: {image}")
        selected = image
    else:
        env_image = os.getenv("VLLM_SR_IMAGE")
        if env_image:
            log.info(f"Using router image from VLLM_SR_IMAGE: {env_image}")
            selected = env_image
        elif normalized_platform == PLATFORM_AMD:
            amd_image = os.getenv(
                "VLLM_SR_IMAGE_AMD", VLLM_SR_DOCKER_IMAGE_ROCM
            ).strip()
            selected = amd_image or VLLM_SR_DOCKER_IMAGE_ROCM
            log.info(
                f"Platform '{normalized_platform}' detected, using AMD router image: "
                f"{selected}"
            )
        else:
            selected = VLLM_SR_DOCKER_IMAGE_DEFAULT
            log.info(f"Using default router image: {selected}")

    source_name = "router image"
    if image:
        source_name = "specified router image"
    elif os.getenv("VLLM_SR_IMAGE"):
        source_name = "router image in VLLM_SR_IMAGE"
    return _maybe_upgrade_to_rocm_image(
        selected, normalized_platform, source_name, VLLM_SR_DOCKER_IMAGE_DEFAULT
    )


def get_docker_image(image=None, pull_policy=None, platform=None):
    """
    Resolve the split-runtime router image and ensure it is available locally.
    """
    if not pull_policy:
        pull_policy = DEFAULT_IMAGE_PULL_POLICY
    normalized_platform = _resolve_platform_hint(platform)
    selected_image = _resolve_router_image(image, normalized_platform)
    _ensure_image_available(selected_image, pull_policy)
    return selected_image


def get_monolith_image(image=None, pull_policy=None, platform=None):
    """
    Resolve the legacy all-in-one runtime image and ensure it is available locally.
    """
    if not pull_policy:
        pull_policy = DEFAULT_IMAGE_PULL_POLICY

    normalized_platform = _resolve_platform_hint(platform)
    if image:
        log.info(f"Using specified legacy monolith image: {image}")
        selected_image = image
    else:
        env_image = os.getenv("VLLM_SR_IMAGE")
        if env_image:
            log.info(f"Using legacy monolith image from VLLM_SR_IMAGE: {env_image}")
            selected_image = env_image
        elif normalized_platform == PLATFORM_AMD:
            selected_image = (
                os.getenv(
                    "VLLM_SR_MONOLITH_IMAGE_AMD", VLLM_SR_MONOLITH_IMAGE_ROCM
                ).strip()
                or VLLM_SR_MONOLITH_IMAGE_ROCM
            )
            log.info(
                f"Platform '{normalized_platform}' detected, using AMD monolith image: "
                f"{selected_image}"
            )
        else:
            selected_image = VLLM_SR_MONOLITH_IMAGE_DEFAULT
            log.info(f"Using default legacy monolith image: {selected_image}")

    selected_image = _maybe_upgrade_to_rocm_image(
        selected_image,
        normalized_platform,
        "legacy monolith image",
        VLLM_SR_MONOLITH_IMAGE_DEFAULT,
    )
    _ensure_image_available(selected_image, pull_policy)
    return selected_image


def resolve_split_runtime_images(
    image=None,
    pull_policy=None,
    platform=None,
    include_router=True,
    include_dashboard=True,
    include_envoy=True,
    include_dashboard_db=True,
):
    """
    Resolve all images required by the split local runtime.
    """
    if not pull_policy:
        pull_policy = DEFAULT_IMAGE_PULL_POLICY

    normalized_platform = _resolve_platform_hint(platform)
    router_image = _resolve_router_image(image, normalized_platform)

    dashboard_image = (
        os.getenv("VLLM_SR_DASHBOARD_IMAGE", VLLM_SR_DASHBOARD_IMAGE_DEFAULT).strip()
        or VLLM_SR_DASHBOARD_IMAGE_DEFAULT
    )
    envoy_image = (
        os.getenv("VLLM_SR_ENVOY_IMAGE", VLLM_SR_ENVOY_IMAGE_DEFAULT).strip()
        or VLLM_SR_ENVOY_IMAGE_DEFAULT
    )
    dashboard_db_image = (
        os.getenv(
            "VLLM_SR_DASHBOARD_DB_IMAGE", VLLM_SR_DASHBOARD_DB_IMAGE_DEFAULT
        ).strip()
        or VLLM_SR_DASHBOARD_DB_IMAGE_DEFAULT
    )

    if include_router:
        _ensure_image_available(router_image, pull_policy)
    for enabled, selected_image in (
        (include_dashboard, dashboard_image),
        (include_envoy, envoy_image),
        (include_dashboard_db, dashboard_db_image),
    ):
        if not enabled:
            continue
        _ensure_image_available(selected_image, pull_policy)

    return SplitRuntimeImages(
        router=router_image,
        dashboard=dashboard_image,
        envoy=envoy_image,
        dashboard_db=dashboard_db_image,
    )


def _show_image_not_found_error(image_name):
    """Show helpful error message when an image is not found."""
    runtime = get_container_runtime()
    log.error("=" * 70)
    log.error("Container image not found!")
    log.error("=" * 70)
    log.error("")
    log.error(f"Image: {image_name}")
    log.error("")
    log.error("Options:")
    log.error("")
    log.error("  1. Pull the image:")
    log.error(f"     {runtime} pull {image_name}")
    log.error("")
    log.error("  2. Use custom image:")
    log.error("     vllm-sr serve --image your-image:tag")
    log.error("")
    log.error("  3. Change pull policy to always:")
    log.error("     vllm-sr serve --image-pull-policy always")
    log.error("")
    log.error("=" * 70)
