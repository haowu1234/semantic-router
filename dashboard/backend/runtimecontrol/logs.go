package runtimecontrol

import (
	"bufio"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// FetchLogs returns logs for a runtime component, using the local container when available.
func (c *Controller) FetchLogs(component string, lines int) ([]string, error) {
	if component == ServiceAll {
		if c.SingleContainerTopology() {
			logs, err := fetchLogsFromSupervisor(c.Services(), lines)
			if err == nil && len(logs) > 0 {
				return logs, nil
			}
			return c.fetchSharedContainerLogs(lines)
		}
		return c.fetchSplitTopologyLogs(lines)
	}

	service, ok := c.Service(component)
	if !ok {
		return nil, fmt.Errorf("unknown runtime service %q", component)
	}

	if c.RunningInContainer() && c.SingleContainerTopology() && service.SupervisorService != "" {
		logs, err := fetchLogsFromSupervisor([]RuntimeService{service}, lines)
		if err == nil && len(logs) > 0 {
			return logs, nil
		}
	}

	return c.fetchLogsFromDocker(service, lines)
}

// TailContainerLogs returns the raw tail of managed-container logs.
func (c *Controller) TailContainerLogs(lines int) string {
	return c.tailDockerLogs(c.containerName, lines)
}

// TailServiceLogs returns the raw tail for one managed service.
func (c *Controller) TailServiceLogs(serviceName string, lines int) string {
	logs, err := c.FetchLogs(serviceName, lines)
	if err != nil || len(logs) == 0 {
		return ""
	}
	return strings.Join(logs, "\n")
}

func (c *Controller) tailDockerLogs(containerName string, lines int) string {
	if !isSafeManagedName(containerName) {
		return ""
	}
	tailArg := strconv.Itoa(lines)
	// #nosec G204 -- the managed container name is restricted to a safe character set.
	cmd := containerRuntimeExec("logs", "--tail", tailArg, containerName)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return ""
	}
	return string(output)
}

func fetchLogsFromSupervisor(services []RuntimeService, lines int) ([]string, error) {
	var result []string
	for _, service := range services {
		for _, logFile := range service.LogFiles {
			// #nosec G204 -- log file paths are static runtime definitions and line count is bounded by caller.
			cmd := exec.Command("tail", "-n", strconv.Itoa(lines), logFile)
			output, err := cmd.CombinedOutput()
			if err != nil {
				continue
			}
			result = append(result, splitLogLines(string(output))...)
		}
	}

	if len(result) > lines {
		return result[len(result)-lines:], nil
	}
	return result, nil
}

func (c *Controller) fetchSharedContainerLogs(lines int) ([]string, error) {
	tailArg := strconv.Itoa(lines * 2)
	// #nosec G204 -- the managed container name is restricted to a safe character set.
	cmd := containerRuntimeExec("logs", "--tail", tailArg, c.containerName)
	output, err := cmd.CombinedOutput()
	if err != nil && len(output) == 0 {
		return []string{}, nil
	}

	allLines := splitLogLines(string(output))
	if len(allLines) > lines {
		return allLines[len(allLines)-lines:], nil
	}
	return allLines, nil
}

func (c *Controller) fetchLogsFromDocker(service RuntimeService, lines int) ([]string, error) {
	containerName := strings.TrimSpace(service.ContainerName)
	if containerName == "" {
		return []string{}, nil
	}

	tailMultiplier := 1
	if c.registry.IsSharedContainerService(service.Name) {
		tailMultiplier = 2
	}

	tailArg := strconv.Itoa(lines * tailMultiplier)
	// #nosec G204 -- the managed container name is restricted to a safe character set.
	cmd := containerRuntimeExec("logs", "--tail", tailArg, containerName)
	output, err := cmd.CombinedOutput()
	if err != nil && len(output) == 0 {
		return []string{}, nil
	}

	allLines := splitLogLines(string(output))
	if !c.registry.IsSharedContainerService(service.Name) {
		if len(allLines) > lines {
			return allLines[len(allLines)-lines:], nil
		}
		return allLines, nil
	}

	return filterSharedContainerLogLines(service.Name, allLines, lines), nil
}

func (c *Controller) fetchSplitTopologyLogs(lines int) ([]string, error) {
	var combined []string
	for _, service := range c.Services() {
		serviceLogs, err := c.fetchLogsFromDocker(service, lines)
		if err != nil {
			continue
		}
		for _, line := range serviceLogs {
			combined = append(combined, "["+service.Name+"] "+line)
		}
	}
	if len(combined) > lines {
		return combined[len(combined)-lines:], nil
	}
	return combined, nil
}

func filterSharedContainerLogLines(component string, allLines []string, lines int) []string {
	var filtered []string
	for _, line := range allLines {
		lineLower := strings.ToLower(line)
		switch component {
		case ServiceRouter:
			if strings.Contains(line, `"caller"`) ||
				strings.Contains(line, "spawned: 'router'") ||
				strings.Contains(lineLower, "success: router") ||
				strings.Contains(lineLower, "router entered running") ||
				strings.Contains(line, "Starting router") ||
				strings.Contains(line, "Starting insecure LLM Router ExtProc server") {
				filtered = append(filtered, line)
			}
		case ServiceEnvoy:
			if (strings.Contains(line, "[20") && strings.Contains(line, "][")) ||
				strings.Contains(line, "spawned: 'envoy'") ||
				strings.Contains(lineLower, "success: envoy") ||
				strings.Contains(lineLower, "envoy entered running") ||
				strings.Contains(line, "Generating Envoy config") {
				filtered = append(filtered, line)
			}
		case ServiceDashboard:
			if (strings.Contains(line, "2026/") || strings.Contains(line, "2025/") || strings.Contains(line, "2027/")) ||
				strings.Contains(line, "spawned: 'dashboard'") ||
				strings.Contains(lineLower, "success: dashboard") ||
				strings.Contains(lineLower, "dashboard entered running") ||
				strings.Contains(line, "Starting dashboard") ||
				strings.Contains(line, "Dashboard listening") {
				filtered = append(filtered, line)
			}
		}
	}

	if len(filtered) > lines {
		return filtered[len(filtered)-lines:]
	}
	return filtered
}

func splitLogLines(output string) []string {
	var result []string
	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := scanner.Text()
		if line != "" {
			result = append(result, line)
		}
	}
	return result
}
