package handlers

import (
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/runtimecontrol"
)

func TestRuntimeControllerUsesSplitRegistryByDefault(t *testing.T) {
	t.Setenv("VLLM_SR_CONTAINER_NAME", "sr-router")
	t.Setenv("VLLM_SR_ROUTER_CONTAINER_NAME", "sr-router")
	t.Setenv("VLLM_SR_ENVOY_CONTAINER_NAME", "sr-envoy")
	t.Setenv("VLLM_SR_DASHBOARD_CONTAINER_NAME", "sr-dashboard")
	t.Setenv("VLLM_SR_DASHBOARD_DB_CONTAINER_NAME", "sr-dashboard-db")

	controller := runtimeController()
	if controller.SingleContainerTopology() {
		t.Fatalf("expected split topology controller")
	}

	service, ok := controller.Service(runtimecontrol.ServiceDashboardDB)
	if !ok {
		t.Fatalf("expected dashboard-db service")
	}
	if service.ContainerName != "sr-dashboard-db" {
		t.Fatalf("dashboard-db container = %q, want %q", service.ContainerName, "sr-dashboard-db")
	}
}
