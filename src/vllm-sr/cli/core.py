"""Core management functions for vLLM Semantic Router."""

import os
import subprocess

from cli.consts import DEFAULT_API_PORT, DEFAULT_ENVOY_PORT, TOPOLOGY_SPLIT
from cli.docker_cli import (
    docker_container_status,
    docker_exec,
    docker_network_disconnect,
    docker_remove_container,
    docker_remove_network,
    docker_start_vllm_sr,
    docker_stop_container,
    load_openclaw_registry,
)
from cli.docker_split_start import (
    docker_start_split_runtime,
    stop_split_runtime_containers,
)
from cli.logo import print_vllm_logo
from cli.runtime_lifecycle import (
    connect_runtime_container,
    ensure_clean_runtime_container,
    ensure_runtime_container_not_exited,
    ensure_shared_network,
    log_runtime_summary,
    log_startup_banner,
    maybe_finish_setup_mode,
    recover_openclaw_containers,
    resolve_openclaw_data_dir,
    start_observability_stack,
    wait_for_router_health,
)
from cli.runtime_stack import resolve_runtime_stack
from cli.runtime_topology import (
    all_known_runtime_containers,
    detect_runtime_topology,
)
from cli.utils import get_logger, load_config

log = get_logger(__name__)

SERVICE_LOG_PATTERNS = {
    "router": r'"caller"|spawned: \'router\'|success: router|cli\.commands',
    "dashboard": r"dashboard|Dashboard|spawned: 'dashboard'|success: dashboard|:8700",
    "envoy": (
        r"\[2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].*\]\[.*\]"
        r"|spawned: 'envoy'|success: envoy"
    ),
}


def start_vllm_sr(
    config_file,
    env_vars=None,
    image=None,
    pull_policy=None,
    enable_observability=True,
    source_config_file=None,
    runtime_config_file=None,
):
    """Start vLLM Semantic Router."""
    if env_vars is None:
        env_vars = {}
    stack_layout = resolve_runtime_stack()

    print_vllm_logo()
    source_config_file = source_config_file or config_file
    runtime_config_file = runtime_config_file or config_file
    listeners = load_config(source_config_file).get("listeners", [])
    if not listeners:
        log.error("No listeners configured in config.yaml")
        raise SystemExit(1)

    # Detect runtime topology based on image and environment
    topology = detect_runtime_topology(image)
    log.info(f"Detected runtime topology: {topology}")

    log_startup_banner(source_config_file, listeners, stack_layout)

    dashboard_disabled = env_vars.get("DISABLE_DASHBOARD") == "true"
    setup_mode = str(env_vars.get("VLLM_SR_SETUP_MODE", "")).lower() == "true"

    if topology == TOPOLOGY_SPLIT:
        # Split runtime: start multiple independent containers
        _start_split_runtime(
            config_file=config_file,
            source_config_file=source_config_file,
            runtime_config_file=runtime_config_file,
            env_vars=env_vars,
            image=image,
            pull_policy=pull_policy,
            enable_observability=enable_observability,
            stack_layout=stack_layout,
            listeners=listeners,
            dashboard_disabled=dashboard_disabled,
            setup_mode=setup_mode,
        )
    else:
        # Monolith runtime: start single container (legacy mode)
        _start_monolith_runtime(
            config_file=config_file,
            source_config_file=source_config_file,
            runtime_config_file=runtime_config_file,
            env_vars=env_vars,
            image=image,
            pull_policy=pull_policy,
            enable_observability=enable_observability,
            stack_layout=stack_layout,
            listeners=listeners,
            dashboard_disabled=dashboard_disabled,
            setup_mode=setup_mode,
        )


def _start_split_runtime(
    config_file,
    source_config_file,
    runtime_config_file,
    env_vars,
    image,
    pull_policy,
    enable_observability,
    stack_layout,
    listeners,
    dashboard_disabled,
    setup_mode,
):
    """Start split runtime with separate containers for each service."""
    from cli.consts import VLLM_SR_ROUTER_CONTAINER_NAME

    # Clean up any existing split runtime containers
    for container_name in all_known_runtime_containers():
        ensure_clean_runtime_container(container_name)

    config_dir = os.path.dirname(os.path.abspath(source_config_file))
    shared_network_name = stack_layout.network_name
    ensure_shared_network(shared_network_name)

    network_name = start_observability_stack(
        enable_observability and not setup_mode,
        shared_network_name,
        config_dir,
        env_vars,
        stack_layout,
    )

    return_code, _stdout, stderr = docker_start_split_runtime(
        source_config_file,
        env_vars,
        listeners,
        image=image,
        pull_policy=pull_policy,
        network_name=network_name or shared_network_name,
        openclaw_network_name=shared_network_name,
        minimal=dashboard_disabled,
        setup_mode=setup_mode,
        stack_layout=stack_layout,
        state_root_dir=config_dir,
        runtime_config_file=runtime_config_file,
    )
    if return_code != 0:
        log.error(f"Failed to start split runtime: {stderr}")
        raise SystemExit(1)

    log.info("Split runtime containers started successfully")

    if maybe_finish_setup_mode(setup_mode, dashboard_disabled, stack_layout):
        return

    # Wait for router health (using the router container name)
    _wait_for_split_router_health(stack_layout)
    ensure_runtime_container_not_exited(VLLM_SR_ROUTER_CONTAINER_NAME)
    recover_openclaw_containers(config_dir, env_vars, shared_network_name)
    log_runtime_summary(
        listeners, stack_layout, dashboard_disabled, enable_observability
    )


def _start_monolith_runtime(
    config_file,
    source_config_file,
    runtime_config_file,
    env_vars,
    image,
    pull_policy,
    enable_observability,
    stack_layout,
    listeners,
    dashboard_disabled,
    setup_mode,
):
    """Start monolith runtime with single all-in-one container (legacy mode)."""
    ensure_clean_runtime_container(stack_layout.container_name)

    shared_network_name = stack_layout.network_name
    config_dir = os.path.dirname(os.path.abspath(source_config_file))
    ensure_shared_network(shared_network_name)
    network_name = start_observability_stack(
        enable_observability,
        shared_network_name,
        config_dir,
        env_vars,
        stack_layout,
    )

    return_code, _stdout, stderr = docker_start_vllm_sr(
        source_config_file,
        env_vars,
        listeners,
        image=image,
        pull_policy=pull_policy,
        network_name=network_name,
        openclaw_network_name=shared_network_name,
        minimal=dashboard_disabled,
        stack_layout=stack_layout,
        state_root_dir=config_dir,
        runtime_config_file=runtime_config_file,
    )
    if return_code != 0:
        log.error(f"Failed to start container: {stderr}")
        raise SystemExit(1)

    log.info("vLLM Semantic Router container started successfully")
    connect_runtime_container(shared_network_name, stack_layout)
    if maybe_finish_setup_mode(setup_mode, dashboard_disabled, stack_layout):
        return

    wait_for_router_health(stack_layout)
    ensure_runtime_container_not_exited(stack_layout.container_name)
    recover_openclaw_containers(config_dir, env_vars, shared_network_name)
    log_runtime_summary(
        listeners, stack_layout, dashboard_disabled, enable_observability
    )


def _wait_for_split_router_health(stack_layout):
    """Wait for split runtime router to become healthy."""
    import time

    from cli.consts import (
        HEALTH_CHECK_INTERVAL,
        HEALTH_CHECK_TIMEOUT,
        VLLM_SR_ROUTER_CONTAINER_NAME,
    )
    from cli.docker_cli import docker_logs

    log.info("Waiting for Router to become healthy...")
    log.info(f"Health check timeout: {HEALTH_CHECK_TIMEOUT}s")

    start_time = time.time()
    check_count = 0

    while time.time() - start_time < HEALTH_CHECK_TIMEOUT:
        check_count += 1

        return_code, _stdout, _stderr = docker_exec(
            VLLM_SR_ROUTER_CONTAINER_NAME,
            ["curl", "-f", "-s", f"http://localhost:{DEFAULT_API_PORT}/health"],
        )
        if return_code == 0:
            elapsed = int(time.time() - start_time)
            log.info(f"Router is healthy (after {elapsed}s, {check_count} checks)")
            return

        if check_count % 10 == 0:
            elapsed = int(time.time() - start_time)
            remaining = int(HEALTH_CHECK_TIMEOUT - elapsed)
            log.info(
                f"  ... still waiting ({elapsed}s elapsed, {remaining}s remaining)"
            )

        time.sleep(HEALTH_CHECK_INTERVAL)

    log.error(f"Router failed to become healthy after {HEALTH_CHECK_TIMEOUT}s")
    log.info("Showing router container logs:")
    docker_logs(VLLM_SR_ROUTER_CONTAINER_NAME, follow=False, tail=100)
    raise SystemExit(1)


def stop_vllm_sr():
    """Stop vLLM Semantic Router and observability containers."""
    log.info("Stopping vLLM Semantic Router...")
    stack_layout = resolve_runtime_stack()

    # Check which topology is currently running
    monolith_status = docker_container_status(stack_layout.container_name)
    split_containers_running = _any_split_containers_running()

    if monolith_status == "not found" and not split_containers_running:
        log.info("No containers found. Nothing to stop.")
        return

    openclaw_data_dir = resolve_openclaw_data_dir(os.getcwd())
    network_name = stack_layout.network_name

    # Stop OpenClaw containers first
    openclaw_entries = load_openclaw_registry(openclaw_data_dir)
    for entry in openclaw_entries:
        name = entry.get("name") or entry.get("containerName")
        if not name:
            continue
        openclaw_status = docker_container_status(name)
        if openclaw_status == "not found":
            continue
        if openclaw_status == "running":
            log.info(f"Stopping OpenClaw container: {name}")
            docker_stop_container(name)
        log.info(f"Disconnecting {name} from {network_name}")
        docker_network_disconnect(network_name, name)

    # Stop split runtime containers if running
    if split_containers_running:
        log.info("Stopping split runtime containers...")
        stop_split_runtime_containers()

    # Stop monolith container if running
    if monolith_status != "not found":
        if monolith_status == "running":
            docker_stop_container(stack_layout.container_name)
        docker_remove_container(stack_layout.container_name)
        log.info("Monolith container stopped")

    log.info("vLLM Semantic Router stopped")

    # Stop observability containers
    observability_containers = [
        stack_layout.grafana_container_name,
        stack_layout.prometheus_container_name,
        stack_layout.jaeger_container_name,
    ]
    for container_name in observability_containers:
        service_status = docker_container_status(container_name)
        if service_status == "not found":
            continue
        log.info(f"Stopping {container_name}...")
        if service_status == "running":
            docker_stop_container(container_name)
        docker_remove_container(container_name)
        log.info(f"{container_name} stopped")

    return_code, _stdout, _stderr = docker_remove_network(network_name)
    if return_code == 0:
        log.info(f"Network {network_name} removed")


def _any_split_containers_running():
    """Check if any split runtime containers are running."""
    from cli.consts import (
        VLLM_SR_DASHBOARD_CONTAINER_NAME,
        VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
        VLLM_SR_ENVOY_CONTAINER_NAME,
        VLLM_SR_ROUTER_CONTAINER_NAME,
    )

    split_containers = [
        VLLM_SR_ROUTER_CONTAINER_NAME,
        VLLM_SR_ENVOY_CONTAINER_NAME,
        VLLM_SR_DASHBOARD_CONTAINER_NAME,
        VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
    ]
    for container_name in split_containers:
        status = docker_container_status(container_name)
        if status != "not found":
            return True
    return False


def show_logs(service: str, follow: bool = False):
    """Show logs from a runtime service."""
    from cli.consts import (
        VLLM_SR_DASHBOARD_CONTAINER_NAME,
        VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
        VLLM_SR_ENVOY_CONTAINER_NAME,
        VLLM_SR_ROUTER_CONTAINER_NAME,
    )
    from cli.docker_cli import docker_logs

    stack_layout = resolve_runtime_stack()
    split_running = _any_split_containers_running()

    if split_running:
        # Split runtime: show logs from specific container
        container_map = {
            "router": VLLM_SR_ROUTER_CONTAINER_NAME,
            "envoy": VLLM_SR_ENVOY_CONTAINER_NAME,
            "dashboard": VLLM_SR_DASHBOARD_CONTAINER_NAME,
            "dashboard-db": VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
        }
        container_name = container_map.get(service)
        if not container_name:
            log.error(f"Invalid service: {service}")
            raise SystemExit(1)

        status = docker_container_status(container_name)
        if status == "not found":
            log.error(f"Container {container_name} not found")
            raise SystemExit(1)

        if follow:
            log.info(f"Following {service} logs (Ctrl+C to stop)...")
            log.info("")

        docker_logs(container_name, follow=follow, tail=100 if not follow else None)
    else:
        # Monolith runtime: grep logs from single container
        _validate_runtime_service(service)
        _ensure_runtime_container_available(stack_layout.container_name)

        if follow:
            log.info(f"Following {service} logs (Ctrl+C to stop)...")
            log.info("")

        grep_pattern = SERVICE_LOG_PATTERNS[service]
        command = _log_command(stack_layout.container_name, grep_pattern, follow)
        try:
            if follow:
                subprocess.run(command, shell=True, check=False)
                return

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout:
                print(result.stdout)
            else:
                log.info(f"No recent {service} logs found")
        except KeyboardInterrupt:
            log.info("\nStopped following logs")
        except Exception as exc:
            log.error(f"Failed to get {service} logs: {exc}")
            raise SystemExit(1) from exc


def show_status(service: str = "all"):
    """Show runtime service status."""
    stack_layout = resolve_runtime_stack()

    # Check for split runtime first
    split_running = _any_split_containers_running()
    monolith_status = docker_container_status(stack_layout.container_name)

    if not split_running and monolith_status == "not found":
        log.info("Status: Not running")
        log.info("Start with: vllm-sr serve")
        return

    if split_running:
        _show_split_runtime_status(service, stack_layout)
    else:
        _show_monolith_runtime_status(service, stack_layout, monolith_status)


def _show_split_runtime_status(service: str, stack_layout):
    """Show status for split runtime topology."""
    from cli.consts import (
        VLLM_SR_DASHBOARD_CONTAINER_NAME,
        VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
        VLLM_SR_ENVOY_CONTAINER_NAME,
        VLLM_SR_ROUTER_CONTAINER_NAME,
    )

    log.info("=" * 60)
    log.info("Runtime Topology: Split (separate containers)")
    log.info("")

    containers = {
        "router": VLLM_SR_ROUTER_CONTAINER_NAME,
        "envoy": VLLM_SR_ENVOY_CONTAINER_NAME,
        "dashboard": VLLM_SR_DASHBOARD_CONTAINER_NAME,
        "dashboard-db": VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
    }

    for svc_name in _requested_services_extended(service):
        container_name = containers.get(svc_name)
        if not container_name:
            continue
        status = docker_container_status(container_name)
        if status == "running":
            log.info(f"{svc_name.capitalize()}: Running ({container_name})")
        elif status == "not found":
            log.info(f"{svc_name.capitalize()}: Not started")
        else:
            log.info(f"{svc_name.capitalize()}: {status} ({container_name})")

    log.info("")
    log.info(f"Dashboard URL: {stack_layout.dashboard_url}")
    log.info("")
    log.info("For detailed logs: vllm-sr logs <envoy|router|dashboard>")
    log.info("=" * 60)


def _show_monolith_runtime_status(service: str, stack_layout, monolith_status: str):
    """Show status for monolith runtime topology."""
    if monolith_status == "exited":
        log.info("Status: Container exited (error)")
        log.info("View logs with: vllm-sr logs <envoy|router>")
        return
    if monolith_status != "running":
        log.info(f"Status: {monolith_status}")
        return

    log.info("=" * 60)
    log.info("Runtime Topology: Monolith (single container)")
    log.info("Container Status: Running")
    log.info("")

    for requested_service in _requested_services(service):
        _report_service_status(requested_service, stack_layout)

    log.info("")
    log.info("For detailed logs: vllm-sr logs <envoy|router|dashboard>")
    log.info("=" * 60)


def _validate_runtime_service(service: str) -> None:
    if service in SERVICE_LOG_PATTERNS:
        return
    log.error(f"Invalid service: {service}")
    log.error("Must be 'envoy', 'router', or 'dashboard'")
    raise SystemExit(1)


def _requested_services(service: str) -> list[str]:
    if service == "all":
        return ["router", "envoy", "dashboard"]
    _validate_runtime_service(service)
    return [service]


def _requested_services_extended(service: str) -> list[str]:
    """Return requested services including dashboard-db for split runtime."""
    if service == "all":
        return ["router", "envoy", "dashboard", "dashboard-db"]
    if service == "dashboard-db":
        return ["dashboard-db"]
    return _requested_services(service)


def _ensure_runtime_container_available(container_name: str) -> None:
    if docker_container_status(container_name) != "not found":
        return
    log.error("Container not found. Is vLLM Semantic Router running?")
    log.info("Start it with: vllm-sr serve")
    raise SystemExit(1)


def _log_command(container_name: str, grep_pattern: str, follow: bool) -> str:
    if follow:
        return f'docker logs -f {container_name} 2>&1 | grep -E "{grep_pattern}"'
    return (
        f'docker logs --tail 200 {container_name} 2>&1 | grep -E "{grep_pattern}" '
        "| tail -50"
    )


def _report_service_status(service: str, stack_layout) -> None:
    checkers = {
        "router": ("Router", _check_router_status, None),
        "envoy": ("Envoy", _check_envoy_status, None),
        "dashboard": (
            "Dashboard",
            _check_dashboard_status,
            stack_layout.dashboard_url,
        ),
    }
    label, checker, detail = checkers[service]
    try:
        is_running = checker(stack_layout.container_name)
        _log_service_status(label, is_running, detail if is_running else None)
    except Exception as exc:
        log.error(f"Failed to check {service} status: {exc}")


def _check_router_status(container_name: str) -> bool:
    return_code, _stdout, _stderr = docker_exec(
        container_name,
        ["curl", "-f", "-s", f"http://localhost:{DEFAULT_API_PORT}/health"],
    )
    return return_code == 0


def _check_envoy_status(container_name: str) -> bool:
    return_code, stdout, _stderr = docker_exec(
        container_name,
        [
            "curl",
            "-f",
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            f"http://localhost:{DEFAULT_ENVOY_PORT}/ready",
        ],
    )
    return return_code == 0 and stdout.strip() == "200"


def _check_dashboard_status(container_name: str) -> bool:
    return_code, stdout, _stderr = docker_exec(
        container_name,
        [
            "curl",
            "-f",
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "http://localhost:8700",
        ],
    )
    return return_code == 0 and stdout.strip() in {"200", "301", "302"}


def _log_service_status(
    label: str, is_running: bool, detail: str | None = None
) -> None:
    if not is_running:
        log.info(f"WARNING {label}: Status unknown")
        return
    if detail:
        log.info(f"{label}: Running ({detail})")
        return
    log.info(f"{label}: Running")
