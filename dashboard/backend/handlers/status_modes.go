package handlers

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/runtimecontrol"
)

// envoyListenerEndpoint returns the Envoy listener endpoint from
// the TARGET_ENVOY_URL environment variable, falling back to the
// conventional default.  Used for status page "endpoints" display.
func envoyListenerEndpoint() string {
	if u := strings.TrimSpace(os.Getenv("TARGET_ENVOY_URL")); u != "" {
		return u
	}
	return "http://localhost:8899"
}

// envoyAdminReadyURL returns the Envoy admin /ready endpoint.
// The admin port is typically 9901 (see envoy.template.yaml).
// We derive it from the ENVOY_ADMIN_URL env var when available.
func envoyAdminReadyURL() string {
	if u := strings.TrimSpace(os.Getenv("ENVOY_ADMIN_URL")); u != "" {
		return strings.TrimSuffix(u, "/") + "/ready"
	}
	return "http://localhost:9901/ready"
}

func detectSystemStatus(routerAPIURL, configDir string) SystemStatus {
	controller := runtimeController()
	runtimePath := filepath.Join(configDir, ".vllm-sr", "router-runtime.json")
	if isRunningInContainer() && controller.SingleContainerTopology() {
		return collectInContainerStatus(runtimePath, routerAPIURL)
	}

	return collectHostStatus(runtimePath, routerAPIURL)
}

func baseSystemStatus() SystemStatus {
	return SystemStatus{
		Overall:        "not_running",
		DeploymentType: "none",
		Services:       []ServiceStatus{},
		Version:        "v0.1.0",
	}
}

func collectInContainerStatus(runtimePath, routerAPIURL string) SystemStatus {
	controller := runtimeController()

	status := baseSystemStatus()
	status.DeploymentType = controller.DeploymentType()
	status.Overall = "healthy"
	status.Endpoints = []string{envoyListenerEndpoint()}

	routerHealthy, routerMsg := checkServiceFromContainerLogs(runtimecontrol.ServiceRouter)
	envoyHealthy, envoyMsg := checkServiceFromContainerLogs(runtimecontrol.ServiceEnvoy)
	dashboardHealthy := true
	dashboardMsg := "Running"

	status.RouterRuntime = resolveRouterRuntimeStatus(
		runtimePath,
		routerAPIURL,
		routerHealthy,
		readRouterLogContentInContainer(),
	)
	routerMsg = applyRuntimeMessage(routerMsg, status.RouterRuntime)
	status.Models = fetchModelsWhenReady(routerAPIURL, routerHealthy)
	status.Services = append(status.Services,
		buildManagedServiceStatus(runtimecontrol.ServiceRouter, boolToStatus(routerHealthy), routerHealthy, routerMsg),
		buildManagedServiceStatus(runtimecontrol.ServiceEnvoy, boolToStatus(envoyHealthy), envoyHealthy, envoyMsg),
		buildManagedServiceStatus(runtimecontrol.ServiceDashboard, boolToStatus(dashboardHealthy), dashboardHealthy, dashboardMsg),
	)
	setDegradedWhenUnhealthy(&status, routerHealthy, envoyHealthy, dashboardHealthy)

	return status
}

func collectHostStatus(runtimePath, routerAPIURL string) SystemStatus {
	serviceStatuses := runtimeController().ManagedServiceStatuses()

	switch {
	case anyServiceHasStatus(serviceStatuses, "running"):
		return collectRunningDockerStatus(runtimePath, routerAPIURL)
	case anyServiceHasStatus(serviceStatuses, "exited"):
		return exitedContainerStatus(serviceStatuses)
	case anyServiceHasStatus(serviceStatuses, "paused", "unknown", "error"):
		return unknownContainerStatus(serviceStatuses)
	default:
		if status, ok := collectDirectStatus(runtimePath, routerAPIURL); ok {
			return status
		}
		return baseSystemStatus()
	}
}

func collectRunningDockerStatus(runtimePath, routerAPIURL string) SystemStatus {
	controller := runtimeController()

	status := baseSystemStatus()
	status.DeploymentType = controller.DeploymentType()
	status.Overall = "healthy"
	status.Endpoints = []string{envoyListenerEndpoint()}

	routerHealthy, routerMsg, routerLogContent := detectManagedRouterHealth(controller, routerAPIURL)
	envoyHealthy, envoyMsg := detectManagedEnvoyHealth(controller)
	dashboardHealthy, dashboardMsg := detectManagedDashboardHealth(controller)

	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy, routerLogContent)
	routerMsg = applyRuntimeMessage(routerMsg, status.RouterRuntime)
	status.Models = fetchModelsWhenReady(routerAPIURL, routerHealthy)
	status.Services = append(status.Services,
		buildManagedServiceStatus(runtimecontrol.ServiceRouter, boolToStatus(routerHealthy), routerHealthy, routerMsg),
		buildManagedServiceStatus(runtimecontrol.ServiceEnvoy, boolToStatus(envoyHealthy), envoyHealthy, envoyMsg),
		buildManagedServiceStatus(runtimecontrol.ServiceDashboard, boolToStatus(dashboardHealthy), dashboardHealthy, dashboardMsg),
	)

	if service, ok := controller.Service(runtimecontrol.ServiceDashboardDB); ok {
		dbHealthy, dbMsg := detectContainerBackedServiceHealth(controller, service.Name)
		status.Services = append(
			status.Services,
			buildManagedServiceStatus(service.Name, boolToStatus(dbHealthy), dbHealthy, dbMsg),
		)
		setDegradedWhenUnhealthy(&status, dbHealthy)
	}

	setDegradedWhenUnhealthy(&status, routerHealthy, envoyHealthy, dashboardHealthy)
	return status
}

func exitedContainerStatus(serviceStatuses map[string]string) SystemStatus {
	status := baseSystemStatus()
	status.DeploymentType = runtimeController().DeploymentType()
	status.Overall = "stopped"

	for _, service := range runtimeController().Services() {
		componentStatus := serviceStatuses[service.Name]
		if componentStatus == "" || componentStatus == "not found" {
			continue
		}
		status.Services = append(status.Services, buildServiceStatus(
			service.DisplayName,
			componentStatus,
			false,
			"Managed service exited. Check logs with: vllm-sr logs "+service.Name,
			service.Component,
		))
	}

	return status
}

func unknownContainerStatus(serviceStatuses map[string]string) SystemStatus {
	status := baseSystemStatus()
	status.DeploymentType = runtimeController().DeploymentType()
	status.Overall = "degraded"

	for _, service := range runtimeController().Services() {
		componentStatus := serviceStatuses[service.Name]
		if componentStatus == "" || componentStatus == "not found" {
			continue
		}
		status.Services = append(status.Services, buildServiceStatus(
			service.DisplayName,
			componentStatus,
			false,
			containerStatusMessage(componentStatus),
			service.Component,
		))
	}

	return status
}

func collectDirectStatus(runtimePath, routerAPIURL string) (SystemStatus, bool) {
	if routerAPIURL == "" {
		return SystemStatus{}, false
	}

	routerHealthy, routerMsg := checkHTTPHealth(strings.TrimSuffix(routerAPIURL, "/") + "/health")
	if !routerHealthy {
		return SystemStatus{}, false
	}

	status := baseSystemStatus()
	status.DeploymentType = "local (direct)"
	status.Overall = "healthy"
	status.Endpoints = []string{routerAPIURL}
	status.RouterRuntime = resolveRouterRuntimeStatus(runtimePath, routerAPIURL, routerHealthy, "")
	routerMsg = applyRuntimeMessage(routerMsg, status.RouterRuntime)
	status.Models = fetchModelsWhenReady(routerAPIURL, true)
	status.Services = append(status.Services, buildServiceStatus("Router", "running", true, routerMsg, "process"))

	appendDirectEnvoyStatus(&status)
	status.Services = append(status.Services, buildServiceStatus("Dashboard", "running", true, "Running", "process"))

	return status, true
}

func appendDirectEnvoyStatus(status *SystemStatus) {
	envoyRunning, envoyHealthy, envoyMsg := checkEnvoyHealth(envoyAdminReadyURL())
	if !envoyRunning {
		return
	}

	status.Services = append(status.Services, buildServiceStatus("Envoy", boolToStatus(envoyHealthy), envoyHealthy, envoyMsg, "proxy"))
	if !envoyHealthy {
		status.Overall = "degraded"
	}
}

func buildManagedServiceStatus(serviceName, serviceStatus string, healthy bool, message string) ServiceStatus {
	service, ok := runtimeController().Service(serviceName)
	if !ok {
		return buildServiceStatus(serviceName, serviceStatus, healthy, message, "container")
	}
	return buildServiceStatus(service.DisplayName, serviceStatus, healthy, message, service.Component)
}

func setDegradedWhenUnhealthy(status *SystemStatus, checks ...bool) {
	for _, healthy := range checks {
		if !healthy {
			status.Overall = "degraded"
			return
		}
	}
}

func applyRuntimeMessage(message string, runtime *RouterRuntimeStatus) string {
	if runtime != nil && runtime.Message != "" {
		return runtime.Message
	}
	return message
}

func fetchModelsWhenReady(routerAPIURL string, routerHealthy bool) *RouterModelsInfo {
	if !routerHealthy {
		return nil
	}

	return fetchRouterModelsInfo(routerAPIURL)
}

func detectManagedRouterHealth(controller *runtimecontrol.Controller, routerAPIURL string) (bool, string, string) {
	logContent := controller.TailServiceLogs(runtimecontrol.ServiceRouter, 500)
	if routerAPIURL != "" {
		if healthy, msg := checkHTTPHealth(strings.TrimSuffix(routerAPIURL, "/") + "/health"); healthy {
			return true, msg, logContent
		}
	}

	if healthy, msg := checkServiceInLogContent(runtimecontrol.ServiceRouter, logContent); healthy {
		return true, msg, logContent
	}

	return false, containerStatusMessage(controller.ServiceContainerStatus(runtimecontrol.ServiceRouter)), logContent
}

func detectManagedEnvoyHealth(controller *runtimecontrol.Controller) (bool, string) {
	if running, healthy, msg := checkEnvoyHealth(envoyAdminReadyURL()); running {
		return healthy, msg
	}

	logContent := controller.TailServiceLogs(runtimecontrol.ServiceEnvoy, 500)
	if healthy, msg := checkServiceInLogContent(runtimecontrol.ServiceEnvoy, logContent); healthy {
		return true, msg
	}

	return false, containerStatusMessage(controller.ServiceContainerStatus(runtimecontrol.ServiceEnvoy))
}

func detectManagedDashboardHealth(controller *runtimecontrol.Controller) (bool, string) {
	logContent := controller.TailServiceLogs(runtimecontrol.ServiceDashboard, 500)
	if healthy, msg := checkServiceInLogContent(runtimecontrol.ServiceDashboard, logContent); healthy {
		return true, msg
	}

	status := controller.ServiceContainerStatus(runtimecontrol.ServiceDashboard)
	if status == "running" {
		return true, "Running"
	}

	return false, containerStatusMessage(status)
}

func detectContainerBackedServiceHealth(controller *runtimecontrol.Controller, serviceName string) (bool, string) {
	status := controller.ServiceContainerStatus(serviceName)
	if status == "running" {
		return true, "Running"
	}
	return false, containerStatusMessage(status)
}

func containerStatusMessage(containerStatus string) string {
	switch containerStatus {
	case "running":
		return "Container running"
	case "exited":
		return "Container exited"
	case "paused":
		return "Container paused"
	case "not found":
		return "Container not found"
	case "":
		return "Status unknown"
	default:
		return "Container status: " + containerStatus
	}
}

func anyServiceHasStatus(serviceStatuses map[string]string, statuses ...string) bool {
	if len(serviceStatuses) == 0 {
		return false
	}

	for _, actual := range serviceStatuses {
		for _, expected := range statuses {
			if actual == expected {
				return true
			}
		}
	}

	return false
}
