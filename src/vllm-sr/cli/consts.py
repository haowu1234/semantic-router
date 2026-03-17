"""Constants for vLLM Semantic Router CLI."""

from cli import __version__

# Docker image configuration
VLLM_SR_DOCKER_IMAGE_DEFAULT = "ghcr.io/vllm-project/semantic-router/extproc:latest"
VLLM_SR_DOCKER_IMAGE_ROCM = "ghcr.io/vllm-project/semantic-router/extproc-rocm:latest"
VLLM_SR_DASHBOARD_IMAGE_DEFAULT = (
    "ghcr.io/vllm-project/semantic-router/dashboard:latest"
)
VLLM_SR_ENVOY_IMAGE_DEFAULT = "envoyproxy/envoy:v1.34-latest"
VLLM_SR_DASHBOARD_DB_IMAGE_DEFAULT = "postgres:16-alpine"
VLLM_SR_MONOLITH_IMAGE_DEFAULT = "ghcr.io/vllm-project/semantic-router/vllm-sr:latest"
VLLM_SR_MONOLITH_IMAGE_ROCM = "ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest"
VLLM_SR_DOCKER_IMAGE_DEV = "ghcr.io/vllm-project/semantic-router/extproc:latest"
VLLM_SR_DASHBOARD_IMAGE_DEV = "ghcr.io/vllm-project/semantic-router/dashboard:latest"
VLLM_SR_DOCKER_IMAGE_RELEASE = (
    f"ghcr.io/vllm-project/semantic-router/extproc:{__version__}"
)

# Container names
VLLM_SR_DOCKER_NAME = "vllm-sr-container"
DEFAULT_STACK_NAME = "vllm-sr"
VLLM_SR_ROUTER_CONTAINER_NAME = "vllm-sr-router"
VLLM_SR_ENVOY_CONTAINER_NAME = "vllm-sr-envoy"
VLLM_SR_DASHBOARD_CONTAINER_NAME = "vllm-sr-dashboard"
VLLM_SR_DASHBOARD_DB_CONTAINER_NAME = "vllm-sr-dashboard-db"
VLLM_SR_NETWORK_NAME = "vllm-sr-network"

PLATFORM_AMD = "amd"

# Image pull policies
IMAGE_PULL_POLICY_ALWAYS = "always"
IMAGE_PULL_POLICY_IF_NOT_PRESENT = "ifnotpresent"
IMAGE_PULL_POLICY_NEVER = "never"
DEFAULT_IMAGE_PULL_POLICY = IMAGE_PULL_POLICY_ALWAYS

# Runtime topology
TOPOLOGY_MONOLITH = "monolith"
TOPOLOGY_SPLIT = "split"

# Service names
SERVICE_NAME_ALL = "all"
SERVICE_NAME_ROUTER = "router"
SERVICE_NAME_ENVOY = "envoy"
SERVICE_NAME_DASHBOARD = "dashboard"
SERVICE_NAME_DASHBOARD_DB = "dashboard-db"
STATUS_SERVICES = [
    SERVICE_NAME_ROUTER,
    SERVICE_NAME_ENVOY,
    SERVICE_NAME_DASHBOARD,
    SERVICE_NAME_DASHBOARD_DB,
    SERVICE_NAME_ALL,
]
LOG_SERVICES = [
    SERVICE_NAME_ROUTER,
    SERVICE_NAME_ENVOY,
    SERVICE_NAME_DASHBOARD,
    SERVICE_NAME_DASHBOARD_DB,
]

# Default ports
DEFAULT_ENVOY_PORT = 9901
DEFAULT_ROUTER_PORT = 50051
DEFAULT_API_PORT = 8080
DEFAULT_LISTENER_PORT = 8899
DEFAULT_DASHBOARD_PORT = 8700
DEFAULT_METRICS_PORT = 9190
DEFAULT_POSTGRES_PORT = 5432

# Health check
HEALTH_CHECK_TIMEOUT = 1800  # 5 minutes (increased for model loading)
HEALTH_CHECK_INTERVAL = 2

# Log prefixes
LOG_PREFIX_ROUTER = "[router]"
LOG_PREFIX_ENVOY = "[envoy]"
LOG_PREFIX_ACCESS = "[access_logs]"

# File descriptor limits
DEFAULT_NOFILE_LIMIT = 65536
MIN_NOFILE_LIMIT = 8192

# External API model formats (routed through Envoy to external API endpoints)
# These models don't require vLLM endpoints - they use external APIs like Anthropic
EXTERNAL_API_MODEL_FORMATS = ["anthropic"]
