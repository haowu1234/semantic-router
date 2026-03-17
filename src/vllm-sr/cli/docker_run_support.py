"""Shared helpers for building container run commands."""

import os
import shutil
import socket
from contextlib import suppress

from cli.consts import (
    DEFAULT_NOFILE_LIMIT,
    MIN_NOFILE_LIMIT,
    PLATFORM_AMD,
)
from cli.docker_images import _normalize_platform
from cli.utils import getLogger

log = getLogger(__name__)


def resolve_platform(env_vars):
    """Resolve the effective runtime platform hint."""
    platform = (
        env_vars.get("DASHBOARD_PLATFORM")
        or env_vars.get("VLLM_SR_PLATFORM")
        or os.getenv("VLLM_SR_PLATFORM")
    )
    return _normalize_platform(platform)


def resolve_nofile_limit():
    """Resolve and validate the effective nofile ulimit."""
    nofile_limit = int(os.getenv("VLLM_SR_NOFILE_LIMIT", DEFAULT_NOFILE_LIMIT))
    if nofile_limit < MIN_NOFILE_LIMIT:
        log.warning(
            f"File descriptor limit {nofile_limit} is below minimum {MIN_NOFILE_LIMIT}. "
            "Using minimum value."
        )
        return MIN_NOFILE_LIMIT
    if nofile_limit != DEFAULT_NOFILE_LIMIT:
        log.info(f"Using custom file descriptor limit: {nofile_limit}")
    return nofile_limit


def append_amd_gpu_passthrough(cmd, normalized_platform):
    """Attach AMD GPU devices when the selected platform expects ROCm."""
    if normalized_platform != PLATFORM_AMD:
        return

    passthrough_enabled = os.getenv("VLLM_SR_AMD_GPU_PASSTHROUGH", "1").lower()
    if passthrough_enabled in ["0", "false", "no", "off"]:
        log.info(
            "AMD GPU passthrough disabled by VLLM_SR_AMD_GPU_PASSTHROUGH environment variable"
        )
        return

    required_devices = ["/dev/kfd", "/dev/dri"]
    mounted_devices = []
    missing_devices = []
    for device in required_devices:
        if os.path.exists(device):
            cmd.extend(["--device", device])
            mounted_devices.append(device)
        else:
            missing_devices.append(device)

    if mounted_devices:
        cmd.extend(["--group-add", "video"])
        cmd.extend(["--cap-add", "SYS_PTRACE"])
        cmd.extend(["--security-opt", "seccomp=unconfined"])
        log.info(
            f"AMD GPU passthrough enabled with devices: {', '.join(mounted_devices)}"
        )

    if missing_devices:
        log.warning(
            "Platform 'amd' selected but missing AMD GPU devices on host: "
            f"{', '.join(missing_devices)}. Container may fall back to CPU."
        )


def append_host_gateway(cmd, runtime):
    """Attach a host-gateway alias for local-container to host communication."""
    if runtime == "docker":
        cmd.append("--add-host=host.docker.internal:host-gateway")
        return

    try:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.connect(("8.8.8.8", 80))
        host_ip = udp_socket.getsockname()[0]
        udp_socket.close()
        cmd.append(f"--add-host=host.docker.internal:{host_ip}")
        log.info(f"Using host IP for Podman: {host_ip}")
    except Exception as exc:
        log.warning(f"Could not detect host IP for Podman: {exc}")
        log.info("Podman will use host.containers.internal by default")


def attach_docker_socket(cmd, user_gid=None):
    """Mount a host container-runtime socket into the target container when available."""
    docker_socket = _normalize_socket_candidate(
        os.getenv("VLLM_SR_CONTAINER_RUNTIME_SOCKET")
        or os.getenv("VLLM_SR_DOCKER_SOCKET")
        or os.getenv("DOCKER_HOST")
        or os.getenv("CONTAINER_HOST")
    )
    socket_candidates = []
    if docker_socket:
        socket_candidates.append(docker_socket)
    else:
        socket_candidates.append("/var/run/docker.sock")
        xdg_runtime_dir = os.getenv("XDG_RUNTIME_DIR")
        if xdg_runtime_dir:
            socket_candidates.append(os.path.join(xdg_runtime_dir, "docker.sock"))
            socket_candidates.append(
                os.path.join(xdg_runtime_dir, "podman", "podman.sock")
            )
        with suppress(Exception):
            socket_candidates.append(f"/run/user/{os.getuid()}/docker.sock")
            socket_candidates.append(f"/run/user/{os.getuid()}/podman/podman.sock")
        socket_candidates.append("/run/podman/podman.sock")

    resolved_socket = next(
        (
            candidate
            for candidate in socket_candidates
            if candidate and os.path.exists(candidate)
        ),
        None,
    )
    if resolved_socket:
        cmd.extend(["-v", f"{resolved_socket}:/var/run/docker.sock"])
        _append_socket_group(cmd, resolved_socket, user_gid=user_gid)
        cmd.extend(["-e", "DOCKER_HOST=unix:///var/run/docker.sock"])
        cmd.extend(["-e", "CONTAINER_HOST=unix:///var/run/docker.sock"])
        log.info(f"Mounting container runtime socket: {resolved_socket}")
        return True

    log.warning(
        "Container runtime socket not found (checked: "
        f"{', '.join(socket_candidates)}); container-managed peers may be unavailable"
    )
    return False


def attach_docker_cli(cmd):
    """Point in-container peer-management code at the bundled Docker CLI."""
    runtime_command = os.getenv("VLLM_SR_CONTAINER_RUNTIME") or "docker"
    cmd.extend(["-e", f"OPENCLAW_CONTAINER_RUNTIME={runtime_command}"])
    cmd.extend(["-e", f"VLLM_SR_CONTAINER_RUNTIME={runtime_command}"])
    cmd.extend(["-e", f"CONTAINER_RUNTIME={runtime_command}"])
    return shutil.which("docker") is not None


def _normalize_socket_candidate(candidate):
    """Normalize a socket candidate that may be a unix:// URL."""
    if not candidate:
        return ""
    normalized = candidate.strip()
    if normalized.startswith("unix://"):
        return normalized[len("unix://") :]
    return normalized


def _append_socket_group(cmd, socket_path, user_gid=None):
    """Grant access to the mounted runtime socket for non-root local users."""
    if user_gid is None:
        return

    try:
        socket_gid = os.stat(socket_path).st_gid
    except OSError as exc:
        log.warning(f"Could not inspect runtime socket group for {socket_path}: {exc}")
        return

    if user_gid is not None and socket_gid == user_gid:
        return

    cmd.extend(["--group-add", str(socket_gid)])
