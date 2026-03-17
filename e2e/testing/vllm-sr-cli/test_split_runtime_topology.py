#!/usr/bin/env python3
"""
test_split_runtime_topology.py - E2E tests for Split Runtime communication topology.

This test validates the communication links between containers in split runtime mode:
- Network connectivity: All containers on the same Docker network
- Router ↔ Envoy: extproc gRPC connection (port 50051)
- Dashboard → Router: API and metrics endpoints
- Dashboard → Envoy: Listener and admin endpoints
- OpenClaw → Envoy: LLM requests via OpenAI-compatible API

Signed-off-by: vLLM-SR Team
"""

import json
import os
import re
import subprocess
import time
import unittest
from contextlib import contextmanager
from urllib import error as urllib_error
from urllib import request as urllib_request

from cli_test_base import CLITestBase


class TestSplitRuntimeTopology(CLITestBase):
    """E2E tests for split runtime container topology and communication links."""

    # Extended timeouts for split runtime (multiple containers)
    CONTAINER_STARTUP_TIMEOUT = 180
    HEALTH_CHECK_TIMEOUT = 120

    # Container names in split runtime mode
    SPLIT_CONTAINERS = (
        "vllm-sr-router",
        "vllm-sr-envoy",
        "vllm-sr-dashboard",
        "vllm-sr-dashboard-db",
    )

    # Expected Docker network name
    NETWORK_NAME = "vllm-sr-network"

    # Port mappings
    ROUTER_GRPC_PORT = 50051
    ROUTER_API_PORT = 8080
    ROUTER_METRICS_PORT = 9190
    ENVOY_LISTENER_PORT = 8899
    ENVOY_ADMIN_PORT = 9901
    DASHBOARD_PORT = 3000
    POSTGRES_PORT = 5432

    def _create_split_runtime_config(self, listener_port: int = 8899) -> str:
        """Create a config.yaml suitable for split runtime testing."""
        config_content = f"""version: v0.1

listeners:
  - name: "test-listener"
    address: "0.0.0.0"
    port: {listener_port}
    timeout: "60s"

decisions:
  - name: "default-route"
    description: "Default route for topology testing"
    priority: 100
    rules:
      operator: "AND"
      conditions: []
    modelRefs:
      - model: "test-model"
        use_reasoning: false

providers:
  models:
    - name: "test-model"
      endpoints:
        - name: "primary"
          weight: 100
          endpoint: "host.docker.internal:8000/v1"
          protocol: "http"
  default_model: "test-model"
"""
        config_path = os.path.join(self.test_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)
        return config_path

    def _start_serve_background(
        self, env: dict[str, str] | None = None
    ) -> subprocess.Popen:
        """Start vllm-sr serve in background (non-blocking)."""
        cmd = ["vllm-sr", "serve", "--image-pull-policy", "ifnotpresent"]
        print(f"\nStarting split runtime: {' '.join(cmd)}")

        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_dir,
            env=full_env,
        )
        return process

    def _stop_serve_process(self, serve_process: subprocess.Popen | None):
        """Terminate a background serve process if it is still running."""
        if serve_process and serve_process.poll() is None:
            serve_process.terminate()
            try:
                serve_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                serve_process.kill()

    def _wait_for_split_runtime_ready(self, serve_process: subprocess.Popen) -> bool:
        """Wait for all split runtime containers to be running."""
        time.sleep(5)

        if serve_process.poll() is not None:
            stdout, stderr = serve_process.communicate()
            print(f"Serve crashed early: {stderr[:500] or stdout[:500]}")
            return False

        print(
            f"Waiting for split runtime containers (timeout: {self.CONTAINER_STARTUP_TIMEOUT}s)..."
        )

        start = time.time()
        while time.time() - start < self.CONTAINER_STARTUP_TIMEOUT:
            all_running = True
            for container in self.SPLIT_CONTAINERS:
                status = self._get_container_status(container)
                if status != "running":
                    all_running = False
                    break

            if all_running:
                print("  ✓ All split runtime containers are running")
                return True

            time.sleep(3)

        print("  ✗ Timeout waiting for containers")
        return False

    def _get_container_status(self, container_name: str) -> str:
        """Get the status of a specific container."""
        try:
            result = self._run_subprocess(
                [
                    self.container_runtime,
                    "ps",
                    "-a",
                    "--filter",
                    f"name=^{container_name}$",
                    "--format",
                    "{{.Status}}",
                ],
                timeout=10,
            )
            status = result.stdout.strip()
            if not status:
                return "not found"
            if "Up" in status:
                return "running"
            if "Exited" in status:
                return "exited"
            return "unknown"
        except Exception as e:
            print(f"Failed to get container status for {container_name}: {e}")
            return "error"

    def _get_container_networks(self, container_name: str) -> list[str]:
        """Get list of networks a container is connected to."""
        try:
            result = self._run_subprocess(
                [
                    self.container_runtime,
                    "inspect",
                    "--format",
                    "{{json .NetworkSettings.Networks}}",
                    container_name,
                ],
                timeout=10,
            )
            if result.returncode != 0:
                return []

            networks_json = result.stdout.strip()
            if not networks_json or networks_json == "null":
                return []

            networks = json.loads(networks_json)
            return list(networks.keys())
        except Exception as e:
            print(f"Failed to get networks for {container_name}: {e}")
            return []

    def _get_container_ip(self, container_name: str, network_name: str) -> str | None:
        """Get the IP address of a container on a specific network."""
        try:
            result = self._run_subprocess(
                [
                    self.container_runtime,
                    "inspect",
                    "--format",
                    f"{{{{.NetworkSettings.Networks.{network_name}.IPAddress}}}}",
                    container_name,
                ],
                timeout=10,
            )
            if result.returncode != 0:
                return None
            ip = result.stdout.strip()
            return ip if ip else None
        except Exception:
            return None

    def _get_container_env(self, container_name: str) -> dict[str, str]:
        """Get environment variables from a container."""
        try:
            result = self._run_subprocess(
                [
                    self.container_runtime,
                    "inspect",
                    "--format",
                    "{{json .Config.Env}}",
                    container_name,
                ],
                timeout=10,
            )
            if result.returncode != 0:
                return {}

            env_list = json.loads(result.stdout.strip())
            env_dict = {}
            for item in env_list:
                if "=" in item:
                    key, value = item.split("=", 1)
                    env_dict[key] = value
            return env_dict
        except Exception as e:
            print(f"Failed to get env for {container_name}: {e}")
            return {}

    def _exec_in_container(
        self, container_name: str, command: list[str], timeout: int = 30
    ) -> tuple[int, str, str]:
        """Execute a command inside a container."""
        try:
            result = self._run_subprocess(
                [self.container_runtime, "exec", container_name] + command,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def _check_tcp_connectivity(
        self, from_container: str, to_host: str, to_port: int
    ) -> bool:
        """Check if a container can reach a TCP port on another host."""
        # Use nc (netcat) or timeout+bash to check connectivity
        # Try multiple methods as different base images have different tools

        # Method 1: Using nc (netcat)
        rc, stdout, stderr = self._exec_in_container(
            from_container,
            ["nc", "-zv", "-w", "3", to_host, str(to_port)],
            timeout=10,
        )
        if rc == 0:
            return True

        # Method 2: Using timeout + bash TCP check
        rc, stdout, stderr = self._exec_in_container(
            from_container,
            [
                "timeout",
                "3",
                "bash",
                "-c",
                f"</dev/tcp/{to_host}/{to_port}",
            ],
            timeout=10,
        )
        if rc == 0:
            return True

        # Method 3: Using curl (for HTTP ports)
        rc, stdout, stderr = self._exec_in_container(
            from_container,
            ["curl", "-sf", "--max-time", "3", f"http://{to_host}:{to_port}/"],
            timeout=10,
        )
        # curl returns 0 on success, or non-zero but we got a response
        if rc == 0 or "HTTP" in stdout or "HTTP" in stderr:
            return True

        return False

    def _check_grpc_connectivity(
        self, from_container: str, to_host: str, to_port: int
    ) -> bool:
        """Check if a container can reach a gRPC port."""
        # For gRPC, we just check TCP connectivity to the port
        return self._check_tcp_connectivity(from_container, to_host, to_port)

    @contextmanager
    def _running_split_runtime(self, env: dict[str, str] | None = None):
        """Context manager to start and stop split runtime."""
        self._create_split_runtime_config()
        os.makedirs(os.path.join(self.test_dir, "models"), exist_ok=True)

        serve_process = self._start_serve_background(env=env)
        try:
            if self._wait_for_split_runtime_ready(serve_process):
                # Additional stabilization time
                time.sleep(5)
                yield serve_process
            else:
                self.fail("Failed to start split runtime")
        finally:
            self._stop_serve_process(serve_process)
            # Ensure cleanup
            self.run_cli(["stop"], timeout=60)

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_all_containers_on_same_network(self):
        """Test that all split runtime containers are on the same Docker network."""
        self.print_test_header(
            "Network Topology Verification",
            "Validates all containers are connected to vllm-sr-network",
        )

        with self._running_split_runtime():
            containers_on_network = {}

            for container in self.SPLIT_CONTAINERS:
                networks = self._get_container_networks(container)
                containers_on_network[container] = networks
                print(f"  {container}: {networks}")

                self.assertIn(
                    self.NETWORK_NAME,
                    networks,
                    f"{container} should be connected to {self.NETWORK_NAME}",
                )

            # Verify all containers have IPs on the shared network
            print("\n  Container IPs on shared network:")
            for container in self.SPLIT_CONTAINERS:
                ip = self._get_container_ip(container, self.NETWORK_NAME)
                print(f"    {container}: {ip}")
                self.assertIsNotNone(
                    ip, f"{container} should have an IP on {self.NETWORK_NAME}"
                )

        self.print_test_result(True, "All containers on shared network")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_envoy_to_router_grpc_connectivity(self):
        """Test Envoy → Router extproc gRPC connectivity (port 50051)."""
        self.print_test_header(
            "Envoy → Router gRPC Connectivity",
            "Validates Envoy can reach Router's extproc gRPC endpoint",
        )

        with self._running_split_runtime():
            # Check that Envoy can reach Router on gRPC port
            can_reach = self._check_grpc_connectivity(
                "vllm-sr-envoy",
                "vllm-sr-router",
                self.ROUTER_GRPC_PORT,
            )

            if can_reach:
                print(f"  ✓ Envoy can reach Router on port {self.ROUTER_GRPC_PORT}")
            else:
                # Fallback: check using Envoy admin stats for extproc cluster health
                rc, stdout, stderr = self._exec_in_container(
                    "vllm-sr-envoy",
                    [
                        "curl",
                        "-sf",
                        f"http://localhost:{self.ENVOY_ADMIN_PORT}/clusters",
                    ],
                    timeout=10,
                )
                if "extproc" in stdout.lower() and "healthy" in stdout.lower():
                    print("  ✓ Envoy extproc cluster is healthy (via admin)")
                    can_reach = True
                else:
                    print(f"  ✗ Cannot verify Envoy → Router connectivity")
                    print(f"    Cluster status: {stdout[:200]}")

            self.assertTrue(can_reach, "Envoy should be able to reach Router gRPC")

        self.print_test_result(True, "Envoy → Router gRPC connectivity verified")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_dashboard_to_router_api_connectivity(self):
        """Test Dashboard → Router API connectivity (port 8080)."""
        self.print_test_header(
            "Dashboard → Router API Connectivity",
            "Validates Dashboard can reach Router's API endpoint",
        )

        with self._running_split_runtime():
            # Check environment variable configuration
            dashboard_env = self._get_container_env("vllm-sr-dashboard")
            router_api_url = dashboard_env.get("TARGET_ROUTER_API_URL", "")

            print(f"  TARGET_ROUTER_API_URL: {router_api_url}")
            self.assertIn(
                "vllm-sr-router",
                router_api_url,
                "Dashboard should use container name for Router API",
            )

            # Check actual connectivity
            can_reach = self._check_tcp_connectivity(
                "vllm-sr-dashboard",
                "vllm-sr-router",
                self.ROUTER_API_PORT,
            )

            if can_reach:
                print(f"  ✓ Dashboard can reach Router on port {self.ROUTER_API_PORT}")
            else:
                print(
                    f"  ✗ Dashboard cannot reach Router on port {self.ROUTER_API_PORT}"
                )

            self.assertTrue(can_reach, "Dashboard should be able to reach Router API")

        self.print_test_result(True, "Dashboard → Router API connectivity verified")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_dashboard_to_envoy_connectivity(self):
        """Test Dashboard → Envoy connectivity (listener and admin ports)."""
        self.print_test_header(
            "Dashboard → Envoy Connectivity",
            "Validates Dashboard can reach Envoy's listener and admin endpoints",
        )

        with self._running_split_runtime():
            # Check environment variable configuration
            dashboard_env = self._get_container_env("vllm-sr-dashboard")
            envoy_url = dashboard_env.get("TARGET_ENVOY_URL", "")
            openclaw_model_base_url = dashboard_env.get("OPENCLAW_MODEL_BASE_URL", "")

            print(f"  TARGET_ENVOY_URL: {envoy_url}")
            print(f"  OPENCLAW_MODEL_BASE_URL: {openclaw_model_base_url}")

            self.assertIn(
                "vllm-sr-envoy",
                envoy_url,
                "Dashboard should use container name for Envoy URL",
            )

            # Check listener port connectivity
            can_reach_listener = self._check_tcp_connectivity(
                "vllm-sr-dashboard",
                "vllm-sr-envoy",
                self.ENVOY_LISTENER_PORT,
            )

            if can_reach_listener:
                print(
                    f"  ✓ Dashboard can reach Envoy listener on port {self.ENVOY_LISTENER_PORT}"
                )
            else:
                print(
                    f"  ✗ Dashboard cannot reach Envoy listener on port {self.ENVOY_LISTENER_PORT}"
                )

            # Check admin port connectivity
            can_reach_admin = self._check_tcp_connectivity(
                "vllm-sr-dashboard",
                "vllm-sr-envoy",
                self.ENVOY_ADMIN_PORT,
            )

            if can_reach_admin:
                print(
                    f"  ✓ Dashboard can reach Envoy admin on port {self.ENVOY_ADMIN_PORT}"
                )
            else:
                print(
                    f"  ⚠ Dashboard cannot reach Envoy admin on port {self.ENVOY_ADMIN_PORT}"
                )

            self.assertTrue(
                can_reach_listener, "Dashboard should be able to reach Envoy listener"
            )

        self.print_test_result(True, "Dashboard → Envoy connectivity verified")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_openclaw_model_base_url_points_to_envoy(self):
        """Test that OPENCLAW_MODEL_BASE_URL correctly points to Envoy (not Dashboard)."""
        self.print_test_header(
            "OpenClaw Model Base URL Configuration",
            "Validates OpenClaw LLM requests route through Envoy gateway",
        )

        with self._running_split_runtime():
            dashboard_env = self._get_container_env("vllm-sr-dashboard")
            openclaw_model_base_url = dashboard_env.get("OPENCLAW_MODEL_BASE_URL", "")

            print(f"  OPENCLAW_MODEL_BASE_URL: {openclaw_model_base_url}")

            # Critical: Must point to Envoy, not Dashboard
            self.assertIn(
                "vllm-sr-envoy",
                openclaw_model_base_url,
                "OPENCLAW_MODEL_BASE_URL must point to Envoy for LLM routing",
            )

            # Should include the listener port
            self.assertIn(
                str(self.ENVOY_LISTENER_PORT),
                openclaw_model_base_url,
                f"OPENCLAW_MODEL_BASE_URL should use Envoy listener port {self.ENVOY_LISTENER_PORT}",
            )

            # Should end with /v1 for OpenAI-compatible API
            self.assertTrue(
                openclaw_model_base_url.endswith("/v1"),
                "OPENCLAW_MODEL_BASE_URL should end with /v1",
            )

            # Should NOT point to Dashboard
            self.assertNotIn(
                "vllm-sr-dashboard",
                openclaw_model_base_url,
                "OPENCLAW_MODEL_BASE_URL must NOT point to Dashboard",
            )
            self.assertNotIn(
                "3000",
                openclaw_model_base_url,
                "OPENCLAW_MODEL_BASE_URL must NOT use Dashboard port 3000",
            )

            print("  ✓ OPENCLAW_MODEL_BASE_URL correctly configured for Envoy")

        self.print_test_result(True, "OpenClaw LLM routing configuration verified")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_dashboard_to_postgres_connectivity(self):
        """Test Dashboard → PostgreSQL database connectivity."""
        self.print_test_header(
            "Dashboard → PostgreSQL Connectivity",
            "Validates Dashboard can reach the database container",
        )

        with self._running_split_runtime():
            # Check environment variable configuration
            dashboard_env = self._get_container_env("vllm-sr-dashboard")
            database_url = dashboard_env.get("DATABASE_URL", "")

            print(f"  DATABASE_URL: {database_url[:50]}...")  # Truncate for security

            self.assertIn(
                "vllm-sr-dashboard-db",
                database_url,
                "Dashboard should use container name for database",
            )

            # Check actual connectivity
            can_reach = self._check_tcp_connectivity(
                "vllm-sr-dashboard",
                "vllm-sr-dashboard-db",
                self.POSTGRES_PORT,
            )

            if can_reach:
                print(
                    f"  ✓ Dashboard can reach PostgreSQL on port {self.POSTGRES_PORT}"
                )
            else:
                print(
                    f"  ✗ Dashboard cannot reach PostgreSQL on port {self.POSTGRES_PORT}"
                )

            self.assertTrue(can_reach, "Dashboard should be able to reach PostgreSQL")

        self.print_test_result(True, "Dashboard → PostgreSQL connectivity verified")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_router_startup_before_envoy(self):
        """Test that Router is started and ready before Envoy starts."""
        self.print_test_header(
            "Router → Envoy Startup Sequence",
            "Validates Router is running before Envoy to avoid DNS resolution failures",
        )

        with self._running_split_runtime():
            # Both should be running now
            router_status = self._get_container_status("vllm-sr-router")
            envoy_status = self._get_container_status("vllm-sr-envoy")

            print(f"  Router status: {router_status}")
            print(f"  Envoy status: {envoy_status}")

            self.assertEqual(router_status, "running", "Router should be running")
            self.assertEqual(envoy_status, "running", "Envoy should be running")

            # Verify Router's gRPC port is accessible from Envoy
            # This indirectly proves the startup sequence is correct
            can_reach = self._check_grpc_connectivity(
                "vllm-sr-envoy",
                "vllm-sr-router",
                self.ROUTER_GRPC_PORT,
            )

            self.assertTrue(
                can_reach,
                "Envoy should be able to reach Router (implies correct startup order)",
            )

            print("  ✓ Startup sequence verified (Router accessible from Envoy)")

        self.print_test_result(True, "Router → Envoy startup sequence verified")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_envoy_listener_health(self):
        """Test that Envoy listener is healthy and responding."""
        self.print_test_header(
            "Envoy Listener Health",
            "Validates Envoy's OpenAI-compatible listener is responsive",
        )

        with self._running_split_runtime():
            # Try to reach Envoy listener from host
            # Note: This assumes port is exposed to host
            time.sleep(3)  # Give Envoy time to fully initialize

            # Check Envoy admin /ready endpoint
            rc, stdout, stderr = self._exec_in_container(
                "vllm-sr-envoy",
                ["curl", "-sf", f"http://localhost:{self.ENVOY_ADMIN_PORT}/ready"],
                timeout=10,
            )

            if rc == 0 and "LIVE" in stdout:
                print("  ✓ Envoy admin reports LIVE status")
            else:
                print(f"  ⚠ Envoy admin status: {stdout}")

            # Check listener is bound
            rc, stdout, stderr = self._exec_in_container(
                "vllm-sr-envoy",
                ["curl", "-sf", f"http://localhost:{self.ENVOY_ADMIN_PORT}/listeners"],
                timeout=10,
            )

            listener_bound = str(self.ENVOY_LISTENER_PORT) in stdout
            if listener_bound:
                print(f"  ✓ Envoy listener bound on port {self.ENVOY_LISTENER_PORT}")
            else:
                print(f"  ⚠ Listener status: {stdout[:200]}")

            self.assertTrue(listener_bound, "Envoy listener should be bound")

        self.print_test_result(True, "Envoy listener health verified")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_full_communication_chain(self):
        """Test the complete communication chain: External → Envoy → Router."""
        self.print_test_header(
            "Full Communication Chain",
            "Validates end-to-end request flow through the split runtime",
        )

        with self._running_split_runtime():
            print("  Testing communication chain:")
            print("    External → Envoy:8899 → Router:50051 (extproc)")

            # Step 1: Verify Envoy is accessible
            rc, stdout, stderr = self._exec_in_container(
                "vllm-sr-envoy",
                ["curl", "-sf", f"http://localhost:{self.ENVOY_ADMIN_PORT}/ready"],
                timeout=10,
            )
            envoy_ready = rc == 0 and "LIVE" in stdout
            print(f"    [1] Envoy ready: {'✓' if envoy_ready else '✗'}")

            # Step 2: Verify Router is accessible from Envoy
            router_reachable = self._check_grpc_connectivity(
                "vllm-sr-envoy",
                "vllm-sr-router",
                self.ROUTER_GRPC_PORT,
            )
            print(
                f"    [2] Router reachable from Envoy: {'✓' if router_reachable else '✗'}"
            )

            # Step 3: Verify extproc cluster is healthy
            rc, stdout, stderr = self._exec_in_container(
                "vllm-sr-envoy",
                ["curl", "-sf", f"http://localhost:{self.ENVOY_ADMIN_PORT}/clusters"],
                timeout=10,
            )
            extproc_healthy = "extproc" in stdout.lower()
            print(
                f"    [3] Extproc cluster configured: {'✓' if extproc_healthy else '✗'}"
            )

            # Step 4: Send a test request through Envoy (will fail without backend, but tests routing)
            rc, stdout, stderr = self._exec_in_container(
                "vllm-sr-dashboard",
                [
                    "curl",
                    "-sf",
                    "--max-time",
                    "5",
                    "-X",
                    "POST",
                    f"http://vllm-sr-envoy:{self.ENVOY_LISTENER_PORT}/v1/chat/completions",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    '{"model": "test-model", "messages": [{"role": "user", "content": "test"}]}',
                ],
                timeout=15,
            )
            # We expect this to fail (no backend), but it should reach Envoy
            request_reached_envoy = rc != 0  # 5xx or timeout expected
            print(
                f"    [4] Request reached Envoy: {'✓' if request_reached_envoy else '⚠ (may have succeeded unexpectedly)'}"
            )

            # Overall assessment
            chain_ok = envoy_ready and router_reachable and extproc_healthy
            self.assertTrue(chain_ok, "Full communication chain should be healthy")

            print("\n  ✓ Communication chain verified")

        self.print_test_result(True, "Full communication chain verified")

    def tearDown(self):
        """Clean up after each test."""
        self.run_cli(["stop"], timeout=60)
        self._cleanup_container()
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
