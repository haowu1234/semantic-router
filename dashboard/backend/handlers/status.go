package handlers

import (
	"encoding/json"
	"net/http"
	"time"
)

// ServiceStatus represents the status of a single service
type ServiceStatus struct {
	Name      string `json:"name"`
	Status    string `json:"status"`
	Healthy   bool   `json:"healthy"`
	Message   string `json:"message,omitempty"`
	Component string `json:"component,omitempty"`
}

// RouterRuntimeStatus captures router startup progress beyond process-level health.
type RouterRuntimeStatus struct {
	Phase            string   `json:"phase"`
	Ready            bool     `json:"ready"`
	Message          string   `json:"message,omitempty"`
	DownloadingModel string   `json:"downloading_model,omitempty"`
	PendingModels    []string `json:"pending_models,omitempty"`
	ReadyModels      int      `json:"ready_models,omitempty"`
	TotalModels      int      `json:"total_models,omitempty"`
}

// SystemStatus represents the overall system status
type SystemStatus struct {
	Overall        string               `json:"overall"`
	DeploymentType string               `json:"deployment_type"`
	Services       []ServiceStatus      `json:"services"`
	RouterRuntime  *RouterRuntimeStatus `json:"router_runtime,omitempty"`
	Models         *RouterModelsInfo    `json:"models,omitempty"`
	Endpoints      []string             `json:"endpoints,omitempty"`
	Version        string               `json:"version,omitempty"`
}

// vllmSrContainerName is the container name used by the Python vllm-sr CLI
// for the monolith container in single-container mode.
const vllmSrContainerName = "vllm-sr-container"

// StatusHandler returns the status of vLLM-SR services
// Aligns with the vllm-sr Python CLI by using the same Docker-based detection
func StatusHandler(routerAPIURL, configDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		status := detectSystemStatus(routerAPIURL, configDir)

		if err := json.NewEncoder(w).Encode(status); err != nil {
			http.Error(w, `{"error":"Failed to encode response"}`, http.StatusInternalServerError)
			return
		}
	}
}

// getDockerContainerStatus checks the status of a Docker container
// Returns: "running", "exited", "not found", or other Docker status
func getDockerContainerStatus(containerName string) string {
	return runtimeControllerForContainer(containerName).DockerContainerStatus()
}

// isRunningInContainer checks if the current process is running inside a Docker container
func isRunningInContainer() bool {
	return runtimeController().RunningInContainer()
}

// checkServiceFromContainerLogs checks service status from supervisorctl within the same container
func checkServiceFromContainerLogs(service string) (bool, string) {
	return runtimeController().SupervisorServiceStatus(service)
}

// boolToStatus converts a boolean to a status string
func boolToStatus(healthy bool) string {
	if healthy {
		return "running"
	}
	return "unknown"
}

func buildServiceStatus(name, serviceStatus string, healthy bool, message, component string) ServiceStatus {
	return ServiceStatus{
		Name:      name,
		Status:    serviceStatus,
		Healthy:   healthy,
		Message:   message,
		Component: component,
	}
}

// checkHTTPHealth performs an HTTP health check
func checkHTTPHealth(url string) (bool, string) {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false, ""
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return true, "HTTP health check OK"
	}
	return false, ""
}

// checkEnvoyHealth checks if Envoy is running and healthy
// Returns: (isRunning, isHealthy, message)
func checkEnvoyHealth(url string) (bool, bool, string) {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false, false, ""
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	// Envoy is running if we got ANY response
	isRunning := true

	// Healthy only if 200
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return isRunning, true, "Ready"
	}

	// Running but not healthy (e.g., 503 "no healthy upstream")
	return isRunning, false, "Running (upstream not ready)"
}
