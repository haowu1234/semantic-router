package handlers

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/runtimecontrol"
)

// LogEntry represents a single log entry
type LogEntry struct {
	Line    string `json:"line"`
	Service string `json:"service,omitempty"`
}

// LogsResponse represents the logs response
type LogsResponse struct {
	DeploymentType string     `json:"deployment_type"`
	Service        string     `json:"service"`
	Logs           []LogEntry `json:"logs"`
	Count          int        `json:"count"`
	Error          string     `json:"error,omitempty"`
	Message        string     `json:"message,omitempty"`
}

// LogsHandler returns logs from vLLM-SR services
// Aligns with the vllm-sr Python CLI by using the same Docker-based approach
func LogsHandler(routerAPIURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "application/json")

		// Parse query parameters
		component := r.URL.Query().Get("component")
		if component == "" {
			component = "router"
		}

		linesStr := r.URL.Query().Get("lines")
		lines := 100
		if linesStr != "" {
			if n, err := strconv.Atoi(linesStr); err == nil && n > 0 && n <= 1000 {
				lines = n
			}
		}

		response := LogsResponse{
			DeploymentType: "none",
			Service:        component,
			Logs:           []LogEntry{},
		}

		controller := runtimeController()
		if component == runtimecontrol.ServiceAll || controller.ServiceContainerStatus(component) != "not found" {
			response.DeploymentType = controller.DeploymentType()
			logs, err := fetchContainerLogs(component, lines)
			if err != nil {
				response.Error = err.Error()
			} else {
				for _, line := range logs {
					if line != "" {
						response.Logs = append(response.Logs, LogEntry{
							Line:    line,
							Service: component,
						})
					}
				}
			}
		} else if routerAPIURL != "" && checkRouterHealth(routerAPIURL) {
			response.DeploymentType = "local (direct)"
			response.Message = "Logs are available for Docker deployments started with 'vllm-sr serve'. " +
				"For the current deployment, logs are written to the process stdout/stderr."
		} else {
			response.Error = "No running deployment detected. Start with: vllm-sr serve"
		}

		response.Count = len(response.Logs)

		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, `{"error":"Failed to encode response"}`, http.StatusInternalServerError)
			return
		}
	}
}

// fetchContainerLogs gets logs from supervisor-managed services
// When running inside the container, use supervisorctl to get logs
// When running outside, use docker logs
func fetchContainerLogs(component string, lines int) ([]string, error) {
	return runtimeController().FetchLogs(component, lines)
}

// checkRouterHealth checks if router is accessible via HTTP
func checkRouterHealth(url string) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}
