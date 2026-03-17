package runtimecontrol

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultRegistryUsesSingleContainerFallback(t *testing.T) {
	t.Setenv("VLLM_SR_CONTAINER_NAME", "")
	t.Setenv("VLLM_SR_ROUTER_CONTAINER_NAME", "")
	t.Setenv("VLLM_SR_ENVOY_CONTAINER_NAME", "")
	t.Setenv("VLLM_SR_DASHBOARD_CONTAINER_NAME", "")
	t.Setenv("VLLM_SR_DASHBOARD_DB_CONTAINER_NAME", "")

	registry := DefaultRegistry()
	if !registry.SingleContainerTopology() {
		t.Fatalf("expected single-container topology")
	}
	if registry.DeploymentType() != "docker" {
		t.Fatalf("deployment type = %q, want docker", registry.DeploymentType())
	}

	router, ok := registry.Service(ServiceRouter)
	if !ok {
		t.Fatalf("expected router service")
	}
	if router.ContainerName != fallbackContainerName {
		t.Fatalf("router container = %q, want %q", router.ContainerName, fallbackContainerName)
	}
}

func TestDefaultRegistrySupportsSplitTopology(t *testing.T) {
	t.Setenv("VLLM_SR_CONTAINER_NAME", "legacy-monolith")
	t.Setenv("VLLM_SR_ROUTER_CONTAINER_NAME", "sr-router")
	t.Setenv("VLLM_SR_ENVOY_CONTAINER_NAME", "sr-envoy")
	t.Setenv("VLLM_SR_DASHBOARD_CONTAINER_NAME", "sr-dashboard")
	t.Setenv("VLLM_SR_DASHBOARD_DB_CONTAINER_NAME", "sr-dashboard-db")

	registry := DefaultRegistry()
	if registry.SingleContainerTopology() {
		t.Fatalf("expected split topology")
	}
	if registry.DeploymentType() != "docker-split" {
		t.Fatalf("deployment type = %q, want docker-split", registry.DeploymentType())
	}

	dashboardDB, ok := registry.Service(ServiceDashboardDB)
	if !ok {
		t.Fatalf("expected dashboard db service")
	}
	if dashboardDB.ContainerName != "sr-dashboard-db" {
		t.Fatalf("dashboard db container = %q, want %q", dashboardDB.ContainerName, "sr-dashboard-db")
	}
}

func TestControllerFetchLogsRejectsUnknownService(t *testing.T) {
	controller := NewControllerWithRegistry(NewServiceRegistryForContainer("vllm-sr-container"))

	if _, err := controller.FetchLogs("not-a-service", 50); err == nil {
		t.Fatalf("expected unknown service error")
	}
}

func TestFilterSharedContainerLogLines(t *testing.T) {
	lines := []string{
		"2026/03/15 Dashboard listening on :8700",
		`{"caller":"router.go","msg":"router ready"}`,
		"[2026-03-15 00:00:00][info][main] source/server/server.cc:100] initializing epoch 0",
	}

	routerLogs := filterSharedContainerLogLines(ServiceRouter, lines, 10)
	if len(routerLogs) != 1 {
		t.Fatalf("router logs len = %d, want 1", len(routerLogs))
	}

	envoyLogs := filterSharedContainerLogLines(ServiceEnvoy, lines, 10)
	if len(envoyLogs) != 1 {
		t.Fatalf("envoy logs len = %d, want 1", len(envoyLogs))
	}

	dashboardLogs := filterSharedContainerLogLines(ServiceDashboard, lines, 10)
	if len(dashboardLogs) != 1 {
		t.Fatalf("dashboard logs len = %d, want 1", len(dashboardLogs))
	}
}

func TestVerifySplitRuntimeServicesRunningReportsMissingService(t *testing.T) {
	runtimePath := filepath.Join(t.TempDir(), "docker")
	script := `#!/bin/sh
container="${4}"
if [ "${container}" = "sr-router" ]; then
  printf 'running\n'
  exit 0
fi
exit 1
`
	if err := os.WriteFile(runtimePath, []byte(script), 0o755); err != nil {
		t.Fatalf("write runtime shim: %v", err)
	}
	t.Setenv("VLLM_SR_CONTAINER_RUNTIME", runtimePath)

	controller := NewControllerWithRegistry(ServiceRegistry{
		primaryContainer: "legacy-monolith",
		services: []RuntimeService{
			{Name: ServiceRouter, ContainerName: "sr-router"},
			{Name: ServiceEnvoy, ContainerName: "sr-envoy"},
		},
	})

	err := controller.verifySplitRuntimeServicesRunning(ServiceRouter, ServiceEnvoy)
	if err == nil {
		t.Fatalf("expected missing envoy error")
	}
	if !strings.Contains(err.Error(), "service envoy is not found") {
		t.Fatalf("unexpected error: %v", err)
	}
}
