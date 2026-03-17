"""Runtime topology helpers for local CLI orchestration."""

import os

from cli.consts import (
    SERVICE_NAME_DASHBOARD,
    SERVICE_NAME_DASHBOARD_DB,
    SERVICE_NAME_ENVOY,
    SERVICE_NAME_ROUTER,
    TOPOLOGY_MONOLITH,
    TOPOLOGY_SPLIT,
    VLLM_SR_DASHBOARD_CONTAINER_NAME,
    VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
    VLLM_SR_DOCKER_NAME,
    VLLM_SR_ENVOY_CONTAINER_NAME,
    VLLM_SR_ROUTER_CONTAINER_NAME,
)

RUNTIME_TOPOLOGY_ENV = "VLLM_SR_RUNTIME_TOPOLOGY"


def normalize_topology(value):
    """Normalize runtime topology input."""
    if value is None:
        return ""
    normalized = str(value).strip().lower()
    if normalized in {TOPOLOGY_MONOLITH, TOPOLOGY_SPLIT}:
        return normalized
    return ""


def is_monolith_image(image_name):
    """Return True when an image name refers to the legacy all-in-one runtime."""
    if not image_name:
        return False

    repository = str(image_name).strip().split("@", 1)[0].split(":", 1)[0]
    return repository.endswith("/vllm-sr") or repository in {
        "vllm-sr",
        "semantic-router",
    }


def detect_runtime_topology(image_name=None):
    """Resolve the effective runtime topology from env vars and image hints."""
    explicit = normalize_topology(os.getenv(RUNTIME_TOPOLOGY_ENV))
    if explicit:
        return explicit

    selected_image = image_name or os.getenv("VLLM_SR_IMAGE")
    if is_monolith_image(selected_image):
        return TOPOLOGY_MONOLITH

    return TOPOLOGY_SPLIT


def container_name_for_service(service_name):
    """Return the managed container name for a split-runtime service."""
    mapping = {
        SERVICE_NAME_ROUTER: VLLM_SR_ROUTER_CONTAINER_NAME,
        SERVICE_NAME_ENVOY: VLLM_SR_ENVOY_CONTAINER_NAME,
        SERVICE_NAME_DASHBOARD: VLLM_SR_DASHBOARD_CONTAINER_NAME,
        SERVICE_NAME_DASHBOARD_DB: VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
    }
    return mapping.get(service_name, "")


def split_service_names(minimal=False, setup_mode=False):
    """Return the split-runtime services that should be present for this mode."""
    if setup_mode:
        return [SERVICE_NAME_DASHBOARD_DB, SERVICE_NAME_DASHBOARD]
    if minimal:
        return [SERVICE_NAME_ROUTER, SERVICE_NAME_ENVOY]
    return [
        SERVICE_NAME_ROUTER,
        SERVICE_NAME_ENVOY,
        SERVICE_NAME_DASHBOARD_DB,
        SERVICE_NAME_DASHBOARD,
    ]


def managed_container_names(minimal=False, setup_mode=False):
    """Return the managed container names for the current split-runtime mode."""
    names = [
        container_name_for_service(name)
        for name in split_service_names(minimal, setup_mode)
    ]
    return [name for name in names if name]


def all_known_runtime_containers():
    """Return all local-runtime container names across both topologies."""
    return [
        VLLM_SR_DOCKER_NAME,
        VLLM_SR_ROUTER_CONTAINER_NAME,
        VLLM_SR_ENVOY_CONTAINER_NAME,
        VLLM_SR_DASHBOARD_CONTAINER_NAME,
        VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
    ]
