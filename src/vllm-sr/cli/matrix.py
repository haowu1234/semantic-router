"""Matrix communication stack for vLLM Semantic Router.

This module manages the Matrix communication infrastructure:
- Tuwunel (Matrix server)
- Matrix user initialization
- OpenClaw agent Matrix plugin configuration
"""

import os
import time
import json
import secrets
from pathlib import Path
from cli.utils import getLogger
from cli.docker_cli import (
    get_container_runtime,
    docker_container_status,
    docker_stop_container,
    docker_remove_container,
    docker_pull_image,
    docker_image_exists,
)

log = getLogger(__name__)

# Container names
MATRIX_SERVER_NAME = "vllm-sr-matrix"
MATRIX_DOMAIN_DEFAULT = "matrix.vllm-sr.local"

# Image configs - Tuwunel (successor to conduwuit)
# DockerHub: jevolk/tuwunel:latest
# GHCR: ghcr.io/matrix-construct/tuwunel:latest
TUWUNEL_IMAGE = "ghcr.io/matrix-construct/tuwunel:latest"


def generate_matrix_token():
    """Generate a secure registration token for Matrix."""
    return secrets.token_hex(16)


def get_matrix_config(config: dict, config_dir: str) -> dict:
    """
    Extract Matrix configuration from user config.yaml.
    
    Args:
        config: Parsed user config dict
        config_dir: Directory containing config.yaml
        
    Returns:
        Matrix configuration dict with defaults applied
    """
    matrix_config = config.get("matrix", {})
    
    # Apply defaults
    defaults = {
        "enabled": False,
        "domain": os.getenv("MATRIX_DOMAIN", MATRIX_DOMAIN_DEFAULT),
        "port": 6167,
        "registration_token": os.getenv("MATRIX_REGISTRATION_TOKEN") or generate_matrix_token(),
        "admin_user": os.getenv("MATRIX_ADMIN_USER", "admin"),
        "data_dir": os.path.join(config_dir, ".vllm-sr", "matrix-data"),
    }
    
    for key, default in defaults.items():
        if key not in matrix_config:
            matrix_config[key] = default
            
    return matrix_config


def docker_start_matrix_server(network_name: str, matrix_config: dict):
    """
    Start Tuwunel Matrix server container.
    
    Args:
        network_name: Docker network to join
        matrix_config: Matrix configuration dict
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    runtime = get_container_runtime()
    
    domain = matrix_config.get("domain", MATRIX_DOMAIN_DEFAULT)
    port = matrix_config.get("port", 6167)
    registration_token = matrix_config.get("registration_token", "")
    data_dir = matrix_config.get("data_dir", "/tmp/matrix-data")
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Build docker run command
    cmd = [
        runtime,
        "run",
        "-d",
        "--name", MATRIX_SERVER_NAME,
        "--network", network_name,
        # Environment variables (CONDUWUIT_ prefix for Tuwunel)
        "-e", f"CONDUWUIT_SERVER_NAME={domain}",
        "-e", "CONDUWUIT_DATABASE_PATH=/data",
        "-e", f"CONDUWUIT_PORT={port}",
        "-e", "CONDUWUIT_ADDRESS=0.0.0.0",  # Listen on all interfaces (required for Docker)
        "-e", "CONDUWUIT_ALLOW_REGISTRATION=true",
        "-e", f"CONDUWUIT_REGISTRATION_TOKEN={registration_token}",
        "-e", "CONDUWUIT_ALLOW_FEDERATION=false",
        "-e", "CONDUWUIT_LOG=info",
        "-e", "CONDUWUIT_MAX_REQUEST_SIZE=52428800",
        "-e", "CONDUWUIT_ROCKSDB_CACHE_CAPACITY_MB=256",
        # Volume for data persistence
        "-v", f"{data_dir}:/data",
        # Port mapping
        "-p", f"{port}:{port}",
        # Note: Tuwunel is a minimal scratch image with NO shell, NO curl, NO tools
        # Health check must be done externally via wait_for_matrix_healthy()
        "--no-healthcheck",
        # Image
        TUWUNEL_IMAGE,
    ]
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def wait_for_matrix_healthy(timeout: int = 60, port: int = 6167) -> bool:
    """
    Wait for Matrix server to become healthy by checking HTTP endpoint.
    
    Args:
        timeout: Maximum seconds to wait
        port: Matrix server port (default 6167)
        
    Returns:
        True if healthy, False if timeout
    """
    import urllib.request
    import urllib.error
    
    start_time = time.time()
    url = f"http://localhost:{port}/_matrix/client/versions"
    
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    log.info(f"Matrix server is healthy (responded to {url})")
                    return True
        except urllib.error.URLError as e:
            log.debug(f"Matrix server not ready yet: {e}")
        except Exception as e:
            log.debug(f"Health check failed: {e}")
                
        time.sleep(2)
    
    log.warning(f"Matrix server health check timed out after {timeout}s")
    return False


def register_matrix_user(matrix_config: dict, username: str, password: str = None, admin: bool = False):
    """
    Register a user on the Matrix server.
    
    Args:
        matrix_config: Matrix configuration dict
        username: Username to register
        password: Password (auto-generated if not provided)
        admin: Whether to make user an admin
        
    Returns:
        Dict with user_id and access_token, or None on failure
    """
    import subprocess
    import urllib.request
    import urllib.error
    
    domain = matrix_config.get("domain", MATRIX_DOMAIN_DEFAULT)
    port = matrix_config.get("port", 6167)
    registration_token = matrix_config.get("registration_token", "")
    
    if password is None:
        password = secrets.token_urlsafe(16)
        
    user_id = f"@{username}:{domain}"
    
    # Prepare registration request
    url = f"http://localhost:{port}/_matrix/client/v3/register"
    data = {
        "username": username,
        "password": password,
        "auth": {
            "type": "m.login.registration_token",
            "token": registration_token,
        },
        "inhibit_login": False,
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {
                "user_id": result.get("user_id", user_id),
                "access_token": result.get("access_token"),
                "password": password,
            }
    except urllib.error.HTTPError as e:
        # User might already exist, try login
        if e.code == 400:
            return login_matrix_user(matrix_config, username, password)
        log.error(f"Failed to register Matrix user {username}: {e}")
        return None
    except Exception as e:
        log.error(f"Failed to register Matrix user {username}: {e}")
        return None


def login_matrix_user(matrix_config: dict, username: str, password: str):
    """
    Login an existing Matrix user.
    
    Args:
        matrix_config: Matrix configuration dict
        username: Username
        password: Password
        
    Returns:
        Dict with user_id and access_token, or None on failure
    """
    import urllib.request
    import urllib.error
    
    domain = matrix_config.get("domain", MATRIX_DOMAIN_DEFAULT)
    port = matrix_config.get("port", 6167)
    
    user_id = f"@{username}:{domain}"
    
    url = f"http://localhost:{port}/_matrix/client/v3/login"
    data = {
        "type": "m.login.password",
        "identifier": {
            "type": "m.id.user",
            "user": username,
        },
        "password": password,
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
            return {
                "user_id": result.get("user_id", user_id),
                "access_token": result.get("access_token"),
                "password": password,
            }
    except Exception as e:
        log.error(f"Failed to login Matrix user {username}: {e}")
        return None


def initialize_matrix_users(matrix_config: dict, config_dir: str) -> dict:
    """
    Initialize required Matrix users (system, admin, leader).
    
    Args:
        matrix_config: Matrix configuration dict
        config_dir: Directory for storing credentials
        
    Returns:
        Dict of username -> credentials
    """
    credentials_file = os.path.join(config_dir, ".vllm-sr", "matrix-credentials.json")
    
    # Load existing credentials if available
    existing_credentials = {}
    if os.path.exists(credentials_file):
        try:
            with open(credentials_file, "r") as f:
                existing_credentials = json.load(f)
        except Exception:
            pass
    
    users_to_create = [
        {"username": "system", "admin": True},
        {"username": matrix_config.get("admin_user", "admin"), "admin": True},
        {"username": "leader", "admin": False},
    ]
    
    credentials = {}
    
    for user_info in users_to_create:
        username = user_info["username"]
        
        # Use existing password if available
        existing = existing_credentials.get(username, {})
        password = existing.get("password")
        
        result = register_matrix_user(matrix_config, username, password, user_info["admin"])
        
        if result:
            credentials[username] = result
            log.info(f"✓ Matrix user ready: @{username}:{matrix_config.get('domain')}")
        else:
            log.warning(f"✗ Failed to create Matrix user: {username}")
    
    # Save credentials
    os.makedirs(os.path.dirname(credentials_file), exist_ok=True)
    with open(credentials_file, "w") as f:
        json.dump(credentials, f, indent=2)
    os.chmod(credentials_file, 0o600)  # Restrict access
    
    return credentials


def generate_openclaw_matrix_config(matrix_config: dict, credentials: dict, agent_name: str, config_dir: str) -> dict:
    """
    Generate OpenClaw configuration with Matrix channel enabled.
    
    Args:
        matrix_config: Matrix configuration dict
        credentials: Matrix credentials dict
        agent_name: Name of the agent (e.g., "leader", "worker-alice")
        config_dir: Config directory
        
    Returns:
        OpenClaw config dict with Matrix channel
    """
    domain = matrix_config.get("domain", MATRIX_DOMAIN_DEFAULT)
    port = matrix_config.get("port", 6167)
    admin_user = matrix_config.get("admin_user", "admin")
    
    # Get credentials for this agent
    agent_creds = credentials.get(agent_name, credentials.get("leader", {}))
    access_token = agent_creds.get("access_token", "")
    
    # Determine if this is a leader or worker
    is_leader = agent_name == "leader"
    
    matrix_channel_config = {
        "enabled": True,
        "homeserver": f"http://localhost:{port}",
        "accessToken": access_token,
        "dm": {
            "policy": "allowlist",
            "allowFrom": [f"@{admin_user}:{domain}"] if is_leader else [],
        },
        "groupPolicy": "allowlist",
        "groupAllowFrom": [
            f"@{admin_user}:{domain}",
            f"@leader:{domain}",
        ],
        "groups": {
            "*": {
                "allow": True,
                "requireMention": True,
            }
        },
    }
    
    return {
        "channels": {
            "matrix": matrix_channel_config,
        }
    }


def stop_matrix_server():
    """Stop and remove the Matrix server container."""
    status = docker_container_status(MATRIX_SERVER_NAME)
    
    if status == "not found":
        return
        
    if status == "running":
        docker_stop_container(MATRIX_SERVER_NAME)
        
    docker_remove_container(MATRIX_SERVER_NAME)
    log.info(f"Matrix server stopped")


def start_matrix_stack(config: dict, config_dir: str, network_name: str) -> dict:
    """
    Start the complete Matrix communication stack.
    
    Args:
        config: User config dict
        config_dir: Config directory
        network_name: Docker network name
        
    Returns:
        Dict with matrix_config and credentials
    """
    matrix_config = get_matrix_config(config, config_dir)
    
    if not matrix_config.get("enabled"):
        log.info("Matrix communication disabled (enable with matrix.enabled: true)")
        return {"matrix_config": matrix_config, "credentials": {}}
    
    log.info("Starting Matrix communication stack...")
    
    # Check for existing container
    status = docker_container_status(MATRIX_SERVER_NAME)
    if status != "not found":
        log.info(f"Existing Matrix server found (status: {status}), cleaning up...")
        stop_matrix_server()
    
    # Pull image if needed
    if not docker_image_exists(TUWUNEL_IMAGE):
        log.info(f"Pulling Matrix server image: {TUWUNEL_IMAGE}")
        docker_pull_image(TUWUNEL_IMAGE)
    
    # Start Matrix server
    log.info("Starting Tuwunel Matrix server...")
    return_code, stdout, stderr = docker_start_matrix_server(network_name, matrix_config)
    
    if return_code != 0:
        log.error(f"Failed to start Matrix server: {stderr}")
        return {"matrix_config": matrix_config, "credentials": {}}
    
    # Wait for healthy
    log.info("Waiting for Matrix server to become healthy...")
    if not wait_for_matrix_healthy(timeout=60):
        log.error("Matrix server failed to become healthy")
        return {"matrix_config": matrix_config, "credentials": {}}
    
    log.info("✓ Matrix server is healthy")
    
    # Initialize users
    log.info("Initializing Matrix users...")
    credentials = initialize_matrix_users(matrix_config, config_dir)
    
    # Save Matrix config for dashboard
    matrix_env_file = os.path.join(config_dir, ".vllm-sr", "matrix-env.json")
    os.makedirs(os.path.dirname(matrix_env_file), exist_ok=True)
    with open(matrix_env_file, "w") as f:
        json.dump({
            "MATRIX_ENABLED": "true",
            "MATRIX_DOMAIN": matrix_config.get("domain"),
            "MATRIX_INTERNAL_URL": f"http://{MATRIX_SERVER_NAME}:{matrix_config.get('port')}",
            "MATRIX_EXTERNAL_URL": f"http://localhost:{matrix_config.get('port')}",
            "MATRIX_SYSTEM_ACCESS_TOKEN": credentials.get("system", {}).get("access_token", ""),
        }, f, indent=2)
    
    log.info("=" * 50)
    log.info("Matrix Communication Stack Ready!")
    log.info(f"  • Server: http://localhost:{matrix_config.get('port')}")
    log.info(f"  • Domain: {matrix_config.get('domain')}")
    log.info(f"  • Admin user: @{matrix_config.get('admin_user')}:{matrix_config.get('domain')}")
    log.info("=" * 50)
    
    return {
        "matrix_config": matrix_config,
        "credentials": credentials,
    }


def stop_matrix_stack():
    """Stop the complete Matrix communication stack."""
    stop_matrix_server()
