#!/bin/sh
set -eu

if [ "${VLLM_SR_PRESERVE_RUN_USER:-}" = "1" ] || [ "${VLLM_SR_PRESERVE_RUN_USER:-}" = "true" ]; then
    exec "$@"
fi

if [ "$(id -u)" -ne 0 ]; then
    exec "$@"
fi

# Fix ownership and permissions for config directory before switching to nonroot user
# This allows the dashboard to write to the mounted config.yaml file
CONFIG_FILE_PATH=${ROUTER_CONFIG_PATH:-/app/config/config.yaml}
CONFIG_DIR=$(dirname "$CONFIG_FILE_PATH")

# Fix ownership for the entire config directory (needed when directory is mounted)
if [ -d "$CONFIG_DIR" ]; then
    chown -R nonroot:nonroot "$CONFIG_DIR" 2>/dev/null || {
        # Fallback: make directory and files group-writable
        chmod -R 775 "$CONFIG_DIR" 2>/dev/null || true
    }
elif [ -f "$CONFIG_FILE_PATH" ]; then
    # Fallback for single file mount
    chown nonroot:nonroot "$CONFIG_FILE_PATH" 2>/dev/null || {
        chmod 664 "$CONFIG_FILE_PATH" 2>/dev/null || true
    }
fi

# Also fix ownership for data directory if it exists
if [ -d "/app/data" ]; then
    chown -R nonroot:nonroot /app/data 2>/dev/null || {
        chmod -R 775 /app/data 2>/dev/null || true
    }
fi

# Switch to nonroot user and execute the dashboard backend
# Using exec to replace shell process with dashboard-backend
exec su-exec nonroot:nonroot "$@"
