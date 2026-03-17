"""Container startup orchestration for split runtime topology."""

import os
import subprocess

from cli.consts import (
    DEFAULT_API_PORT,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_ENVOY_PORT,
    DEFAULT_LISTENER_PORT,
    DEFAULT_METRICS_PORT,
    DEFAULT_POSTGRES_PORT,
    DEFAULT_ROUTER_PORT,
    SERVICE_NAME_DASHBOARD,
    SERVICE_NAME_DASHBOARD_DB,
    SERVICE_NAME_ENVOY,
    SERVICE_NAME_ROUTER,
    VLLM_SR_DASHBOARD_CONTAINER_NAME,
    VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
    VLLM_SR_ENVOY_CONTAINER_NAME,
    VLLM_SR_ROUTER_CONTAINER_NAME,
)
from cli.docker_images import SplitRuntimeImages, resolve_split_runtime_images
from cli.docker_run_support import (
    append_amd_gpu_passthrough,
    append_host_gateway,
    attach_docker_cli,
    attach_docker_socket,
    resolve_nofile_limit,
    resolve_platform,
)
from cli.docker_runtime import get_container_runtime
from cli.docker_services import docker_container_status, docker_remove_container
from cli.runtime_stack import RuntimeStackLayout
from cli.utils import get_logger

log = get_logger(__name__)


def docker_start_split_runtime(
    config_file,
    env_vars,
    listeners,
    image=None,
    pull_policy=None,
    network_name=None,
    openclaw_network_name=None,
    minimal=False,
    setup_mode=False,
    stack_layout: RuntimeStackLayout | None = None,
    state_root_dir: str | None = None,
    runtime_config_file: str | None = None,
):
    """
    Start split runtime topology with separate containers for each service.

    Returns:
        (return_code, stdout, stderr) - aggregate result from all container starts
    """
    runtime = get_container_runtime()
    env_vars = dict(env_vars or {})
    normalized_platform = resolve_platform(env_vars)

    # Resolve images for all services
    include_dashboard = not minimal and not setup_mode
    images = resolve_split_runtime_images(
        image=image,
        pull_policy=pull_policy,
        platform=normalized_platform,
        include_router=not setup_mode,
        include_dashboard=not minimal,
        include_envoy=not setup_mode,
        include_dashboard_db=not minimal,
    )

    config_dir = (
        os.path.abspath(state_root_dir)
        if state_root_dir
        else os.path.dirname(os.path.abspath(config_file))
    )

    # Start services in dependency order
    services_to_start = _determine_services_to_start(minimal, setup_mode)
    log.info(f"Starting split runtime with services: {services_to_start}")

    all_stdout = []
    all_stderr = []

    for service in services_to_start:
        log.info(f"Starting {service} container...")
        rc, stdout, stderr = _start_service_container(
            service=service,
            runtime=runtime,
            images=images,
            config_file=config_file,
            runtime_config_file=runtime_config_file,
            env_vars=env_vars,
            listeners=listeners,
            network_name=network_name,
            openclaw_network_name=openclaw_network_name,
            stack_layout=stack_layout,
            config_dir=config_dir,
            normalized_platform=normalized_platform,
            source_config_file=config_file,
        )
        if rc != 0:
            log.error(f"Failed to start {service}: {stderr}")
            return (rc, "\n".join(all_stdout), stderr)
        all_stdout.append(stdout)
        all_stderr.append(stderr)
        log.info(f"{service} container started successfully")

    return (0, "\n".join(all_stdout), "\n".join(all_stderr))


def _determine_services_to_start(minimal, setup_mode):
    """Determine which services to start based on mode flags."""
    if setup_mode:
        # Setup mode: dashboard-db + dashboard only
        return [SERVICE_NAME_DASHBOARD_DB, SERVICE_NAME_DASHBOARD]
    if minimal:
        # Minimal mode: router + envoy only
        return [SERVICE_NAME_ROUTER, SERVICE_NAME_ENVOY]
    # Full mode: all services
    return [
        SERVICE_NAME_DASHBOARD_DB,
        SERVICE_NAME_ROUTER,
        SERVICE_NAME_ENVOY,
        SERVICE_NAME_DASHBOARD,
    ]


def _start_service_container(
    service,
    runtime,
    images: SplitRuntimeImages,
    config_file,
    runtime_config_file,
    env_vars,
    listeners,
    network_name,
    openclaw_network_name,
    stack_layout,
    config_dir,
    normalized_platform,
    source_config_file,
):
    """Start a single service container."""
    container_name = _container_name_for_service(service)
    _ensure_container_removed(container_name)

    if service == SERVICE_NAME_ROUTER:
        return _start_router_container(
            runtime=runtime,
            container_name=container_name,
            image=images.router,
            config_file=config_file,
            runtime_config_file=runtime_config_file,
            env_vars=env_vars,
            listeners=listeners,
            network_name=network_name,
            openclaw_network_name=openclaw_network_name,
            stack_layout=stack_layout,
            config_dir=config_dir,
            normalized_platform=normalized_platform,
        )
    if service == SERVICE_NAME_ENVOY:
        return _start_envoy_container(
            runtime=runtime,
            container_name=container_name,
            image=images.envoy,
            listeners=listeners,
            network_name=network_name,
            stack_layout=stack_layout,
            config_dir=config_dir,
        )
    if service == SERVICE_NAME_DASHBOARD:
        return _start_dashboard_container(
            runtime=runtime,
            container_name=container_name,
            image=images.dashboard,
            env_vars=env_vars,
            network_name=network_name,
            openclaw_network_name=openclaw_network_name,
            stack_layout=stack_layout,
            config_dir=config_dir,
            source_config_file=source_config_file,
        )
    if service == SERVICE_NAME_DASHBOARD_DB:
        return _start_dashboard_db_container(
            runtime=runtime,
            container_name=container_name,
            image=images.dashboard_db,
            network_name=network_name,
            stack_layout=stack_layout,
            config_dir=config_dir,
        )
    return (1, "", f"Unknown service: {service}")


def _container_name_for_service(service):
    """Return the container name for a service."""
    mapping = {
        SERVICE_NAME_ROUTER: VLLM_SR_ROUTER_CONTAINER_NAME,
        SERVICE_NAME_ENVOY: VLLM_SR_ENVOY_CONTAINER_NAME,
        SERVICE_NAME_DASHBOARD: VLLM_SR_DASHBOARD_CONTAINER_NAME,
        SERVICE_NAME_DASHBOARD_DB: VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
    }
    return mapping.get(service, "")


def _ensure_container_removed(container_name):
    """Remove existing container if present."""
    status = docker_container_status(container_name)
    if status != "not found":
        log.info(f"Removing existing container: {container_name}")
        docker_remove_container(container_name)


def _start_router_container(
    runtime,
    container_name,
    image,
    config_file,
    runtime_config_file,
    env_vars,
    listeners,
    network_name,
    openclaw_network_name,
    stack_layout,
    config_dir,
    normalized_platform,
):
    """Start the router container."""
    nofile_limit = resolve_nofile_limit()

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        "--ulimit",
        f"nofile={nofile_limit}:{nofile_limit}",
    ]

    if network_name:
        cmd.extend(["--network", network_name])

    append_amd_gpu_passthrough(cmd, normalized_platform)
    append_host_gateway(cmd, runtime)

    # Port mappings for router
    for listener in listeners:
        port = listener.get("port")
        if port:
            cmd.extend(["-p", f"{port + stack_layout.port_offset}:{port}"])

    cmd.extend(["-p", f"{stack_layout.router_port}:{DEFAULT_ROUTER_PORT}"])
    cmd.extend(["-p", f"{stack_layout.metrics_port}:{DEFAULT_METRICS_PORT}"])
    cmd.extend(["-p", f"{stack_layout.api_port}:{DEFAULT_API_PORT}"])

    # Mount config and state directories
    source_config_path = os.path.abspath(config_file)
    cmd.extend(["-v", f"{source_config_path}:/app/config.yaml:z"])

    vllm_sr_dir = os.path.join(config_dir, ".vllm-sr")
    if os.path.exists(vllm_sr_dir):
        cmd.extend(["-v", f"{vllm_sr_dir}:/app/.vllm-sr:z"])

    models_dir = os.path.join(config_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    cmd.extend(["-v", f"{models_dir}:/app/models:z"])

    # Environment variables
    router_env = dict(env_vars)
    if runtime_config_file and runtime_config_file != config_file:
        runtime_container_config = (
            f"/app/.vllm-sr/{os.path.basename(runtime_config_file)}"
        )
        router_env["VLLM_SR_RUNTIME_CONFIG_PATH"] = runtime_container_config

    # Set network name for openclaw discovery
    router_env.setdefault(
        "OPENCLAW_DEFAULT_NETWORK_MODE",
        openclaw_network_name or stack_layout.network_name,
    )

    for key, value in router_env.items():
        cmd.extend(["-e", f"{key}={value}"])

    cmd.append(image)

    log.debug(f"Router container command: {' '.join(cmd)}")
    return _run_container_start(cmd)


def _start_envoy_container(
    runtime,
    container_name,
    image,
    listeners,
    network_name,
    stack_layout,
    config_dir,
):
    """Start the Envoy proxy container."""
    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
    ]

    if network_name:
        cmd.extend(["--network", network_name])

    # Port mapping for Envoy admin
    cmd.extend(["-p", f"{DEFAULT_ENVOY_PORT + stack_layout.port_offset}:{DEFAULT_ENVOY_PORT}"])

    # Map listener ports through Envoy
    for listener in listeners:
        port = listener.get("port", DEFAULT_LISTENER_PORT)
        host_port = port + stack_layout.port_offset
        cmd.extend(["-p", f"{host_port}:{port}"])

    # Mount Envoy config
    envoy_config_dir = os.path.join(config_dir, ".vllm-sr", "envoy")
    os.makedirs(envoy_config_dir, exist_ok=True)
    envoy_config = os.path.join(envoy_config_dir, "envoy.yaml")
    _ensure_envoy_config(envoy_config, stack_layout)
    cmd.extend(["-v", f"{envoy_config}:/etc/envoy/envoy.yaml:ro"])

    cmd.append(image)

    log.debug(f"Envoy container command: {' '.join(cmd)}")
    return _run_container_start(cmd)


def _start_dashboard_container(
    runtime,
    container_name,
    image,
    env_vars,
    network_name,
    openclaw_network_name,
    stack_layout,
    config_dir,
    source_config_file,
):
    """Start the dashboard container."""
    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
    ]

    if network_name:
        cmd.extend(["--network", network_name])

    # Port mapping for dashboard
    cmd.extend(["-p", f"{stack_layout.dashboard_port}:{DEFAULT_DASHBOARD_PORT}"])

    # Mount config file to dashboard container
    abs_config_file = os.path.abspath(source_config_file)
    cmd.extend(["-v", f"{abs_config_file}:/app/config/config.yaml:ro"])

    # Mount config directory for config editing operations
    cmd.extend(["-v", f"{config_dir}:/app/workspace:z"])

    # Mount dashboard data directory
    dashboard_data_dir = os.path.join(config_dir, ".vllm-sr", "dashboard-data")
    os.makedirs(dashboard_data_dir, exist_ok=True)
    cmd.extend(["-v", f"{dashboard_data_dir}:/app/data:z"])

    # Mount Docker socket for OpenClaw management
    attach_docker_socket(cmd)
    attach_docker_cli(cmd)

    # Environment variables for dashboard
    router_api_url = f"http://{VLLM_SR_ROUTER_CONTAINER_NAME}:{DEFAULT_API_PORT}"
    router_metrics_url = f"http://{VLLM_SR_ROUTER_CONTAINER_NAME}:9190/metrics"
    envoy_url = f"http://{VLLM_SR_ENVOY_CONTAINER_NAME}:{DEFAULT_LISTENER_PORT}"

    dashboard_env = {
        # Database configuration (use PostgreSQL instead of SQLite)
        "DATABASE_URL": f"postgres://vllm_sr:vllm_sr@{VLLM_SR_DASHBOARD_DB_CONTAINER_NAME}:{DEFAULT_POSTGRES_PORT}/vllm_sr",
        "DASHBOARD_DB_DRIVER": "postgres",
        "DASHBOARD_DB_URL": f"postgres://vllm_sr:vllm_sr@{VLLM_SR_DASHBOARD_DB_CONTAINER_NAME}:{DEFAULT_POSTGRES_PORT}/vllm_sr",
        "EVALUATION_DB_DRIVER": "postgres",
        "EVALUATION_DB_URL": f"postgres://vllm_sr:vllm_sr@{VLLM_SR_DASHBOARD_DB_CONTAINER_NAME}:{DEFAULT_POSTGRES_PORT}/vllm_sr",
        # Router API configuration (connect to router container, not localhost)
        "TARGET_ROUTER_API_URL": router_api_url,
        "TARGET_ROUTER_METRICS_URL": router_metrics_url,
        "TARGET_ENVOY_URL": envoy_url,
        # Config path
        "ROUTER_CONFIG_PATH": "/app/config/config.yaml",
        # OpenClaw network configuration
        "OPENCLAW_DEFAULT_NETWORK_MODE": openclaw_network_name or stack_layout.network_name,
        # Host config path for split runtime resume
        "VLLM_SR_HOST_CONFIG_PATH": abs_config_file,
    }

    # Add passthrough env vars
    for key in ["DASHBOARD_READONLY", "DISABLE_DASHBOARD_AUTH", "DASHBOARD_PLATFORM"]:
        if key in env_vars:
            dashboard_env[key] = env_vars[key]

    # Add observability URLs if configured
    if stack_layout.grafana_container_name:
        dashboard_env["TARGET_GRAFANA_URL"] = f"http://{stack_layout.grafana_container_name}:3000"
    if stack_layout.prometheus_container_name:
        dashboard_env["TARGET_PROMETHEUS_URL"] = f"http://{stack_layout.prometheus_container_name}:9090"
    if stack_layout.jaeger_container_name:
        dashboard_env["TARGET_JAEGER_URL"] = f"http://{stack_layout.jaeger_container_name}:16686"

    for key, value in dashboard_env.items():
        cmd.extend(["-e", f"{key}={value}"])

    cmd.append(image)

    log.debug(f"Dashboard container command: {' '.join(cmd)}")
    return _run_container_start(cmd)


def _start_dashboard_db_container(
    runtime,
    container_name,
    image,
    network_name,
    stack_layout,
    config_dir,
):
    """Start the dashboard database container."""
    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
    ]

    if network_name:
        cmd.extend(["--network", network_name])

    # Mount postgres data directory for persistence
    postgres_data_dir = os.path.join(config_dir, ".vllm-sr", "postgres-data")
    os.makedirs(postgres_data_dir, exist_ok=True)
    cmd.extend(["-v", f"{postgres_data_dir}:/var/lib/postgresql/data:z"])

    # Environment variables for postgres
    cmd.extend(["-e", "POSTGRES_USER=vllm_sr"])
    cmd.extend(["-e", "POSTGRES_PASSWORD=vllm_sr"])
    cmd.extend(["-e", "POSTGRES_DB=vllm_sr"])

    cmd.append(image)

    log.debug(f"Dashboard DB container command: {' '.join(cmd)}")
    return _run_container_start(cmd)


def _ensure_envoy_config(config_path, stack_layout):
    """Ensure Envoy configuration file exists."""
    if os.path.exists(config_path):
        return

    # Generate minimal Envoy config
    envoy_config = f"""
admin:
  address:
    socket_address:
      address: 0.0.0.0
      port_value: {DEFAULT_ENVOY_PORT}

static_resources:
  listeners:
    - name: listener_0
      address:
        socket_address:
          address: 0.0.0.0
          port_value: {DEFAULT_LISTENER_PORT}
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: ingress_http
                codec_type: AUTO
                route_config:
                  name: local_route
                  virtual_hosts:
                    - name: local_service
                      domains: ["*"]
                      routes:
                        - match:
                            prefix: "/"
                          route:
                            cluster: router_cluster
                http_filters:
                  - name: envoy.filters.http.ext_proc
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
                      grpc_service:
                        envoy_grpc:
                          cluster_name: ext_proc_cluster
                      processing_mode:
                        request_header_mode: SEND
                        response_header_mode: SEND
                        request_body_mode: BUFFERED
                        response_body_mode: BUFFERED
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router

  clusters:
    - name: router_cluster
      connect_timeout: 30s
      type: STRICT_DNS
      lb_policy: ROUND_ROBIN
      load_assignment:
        cluster_name: router_cluster
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: {VLLM_SR_ROUTER_CONTAINER_NAME}
                      port_value: {DEFAULT_API_PORT}

    - name: ext_proc_cluster
      connect_timeout: 30s
      type: STRICT_DNS
      lb_policy: ROUND_ROBIN
      typed_extension_protocol_options:
        envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
          "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
          explicit_http_config:
            http2_protocol_options: {{}}
      load_assignment:
        cluster_name: ext_proc_cluster
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: {VLLM_SR_ROUTER_CONTAINER_NAME}
                      port_value: {DEFAULT_ROUTER_PORT}
"""

    with open(config_path, "w") as f:
        f.write(envoy_config)
    log.info(f"Generated Envoy config: {config_path}")


def _run_container_start(cmd):
    """Execute container start command."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)


def stop_split_runtime_containers():
    """Stop all split runtime containers."""
    from cli.docker_services import docker_stop_container

    containers = [
        VLLM_SR_DASHBOARD_CONTAINER_NAME,
        VLLM_SR_ENVOY_CONTAINER_NAME,
        VLLM_SR_ROUTER_CONTAINER_NAME,
        VLLM_SR_DASHBOARD_DB_CONTAINER_NAME,
    ]

    for container_name in containers:
        status = docker_container_status(container_name)
        if status == "not found":
            continue
        log.info(f"Stopping {container_name}...")
        if status == "running":
            docker_stop_container(container_name)
        docker_remove_container(container_name)
        log.info(f"{container_name} stopped")
