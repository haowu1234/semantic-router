package runtimecontrol

import (
	"os"
	"os/exec"
	"regexp"
	"strings"
)

const fallbackContainerName = "vllm-sr-container"

var managedNamePattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_.-]*$`)

// Controller centralizes runtime operations that used to be spread across dashboard handlers.
type Controller struct {
	registry      ServiceRegistry
	containerName string
}

// DefaultContainerName resolves the managed local runtime container name.
func DefaultContainerName() string {
	if name := strings.TrimSpace(os.Getenv("VLLM_SR_CONTAINER_NAME")); name != "" {
		return name
	}
	return fallbackContainerName
}

// NewController returns a controller for the default managed runtime container.
func NewController() *Controller {
	return NewControllerWithRegistry(DefaultRegistry())
}

// NewControllerForContainer returns a controller for a specific managed runtime container.
func NewControllerForContainer(containerName string) *Controller {
	return NewControllerWithRegistry(NewServiceRegistryForContainer(containerName))
}

// NewControllerWithRegistry returns a controller backed by a specific runtime registry.
func NewControllerWithRegistry(registry ServiceRegistry) *Controller {
	name := registry.PrimaryContainerName()
	if !isSafeManagedName(name) {
		name = fallbackContainerName
	}
	return &Controller{
		registry:      registry,
		containerName: name,
	}
}

// ContainerName returns the managed runtime container name used by this controller.
func (c *Controller) ContainerName() string {
	return c.containerName
}

// Services returns the managed service topology in stable display order.
func (c *Controller) Services() []RuntimeService {
	return c.registry.Services()
}

// Service resolves one managed runtime service.
func (c *Controller) Service(name string) (RuntimeService, bool) {
	return c.registry.Service(name)
}

// DeploymentType reports whether the controller is operating against monolithic or split Docker topology.
func (c *Controller) DeploymentType() string {
	return c.registry.DeploymentType()
}

// SingleContainerTopology reports whether router, Envoy, and dashboard still share one managed container.
func (c *Controller) SingleContainerTopology() bool {
	return c.registry.SingleContainerTopology()
}

// DockerContainerStatus returns the Docker status for the managed runtime container.
func (c *Controller) DockerContainerStatus() string {
	return c.dockerContainerStatus(c.containerName)
}

// ServiceContainerStatus returns the Docker status for one managed runtime service.
func (c *Controller) ServiceContainerStatus(serviceName string) string {
	service, ok := c.Service(serviceName)
	if !ok || strings.TrimSpace(service.ContainerName) == "" {
		return "not found"
	}
	return c.dockerContainerStatus(service.ContainerName)
}

func (c *Controller) dockerContainerStatus(containerName string) string {
	if !isSafeManagedName(containerName) {
		return "not found"
	}
	// #nosec G204 -- the managed container name is restricted to a safe character set.
	cmd := containerRuntimeExec("inspect", "-f", "{{.State.Status}}", containerName)
	output, err := cmd.Output()
	if err != nil {
		return "not found"
	}
	return strings.TrimSpace(string(output))
}

// ManagedServiceStatuses returns Docker status for each managed runtime service.
func (c *Controller) ManagedServiceStatuses() map[string]string {
	statuses := make(map[string]string, len(c.registry.services))
	for _, service := range c.Services() {
		statuses[service.Name] = c.ServiceContainerStatus(service.Name)
	}
	return statuses
}

// RunningInContainer reports whether the dashboard process is already inside a container.
func (c *Controller) RunningInContainer() bool {
	if _, err := os.Stat("/.dockerenv"); err == nil {
		return true
	}

	data, err := os.ReadFile("/proc/1/cgroup")
	if err != nil {
		return false
	}
	content := string(data)
	return strings.Contains(content, "docker") || strings.Contains(content, "containerd")
}

// SupervisorServiceStatus checks service state through supervisorctl in the current container.
func (c *Controller) SupervisorServiceStatus(service string) (bool, string) {
	if !isKnownSupervisorService(service) {
		return false, "Status unknown"
	}
	// #nosec G204 -- supervisor service names are restricted to the managed allowlist.
	cmd := exec.Command("supervisorctl", "status", service)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return false, "Status unknown"
	}

	status := string(output)
	switch {
	case strings.Contains(status, "RUNNING"):
		return true, "Running"
	case strings.Contains(status, "STOPPED"):
		return false, "Stopped"
	case strings.Contains(status, "FATAL"), strings.Contains(status, "EXITED"):
		return false, "Failed"
	case strings.Contains(status, "STARTING"):
		return false, "Starting"
	default:
		return false, "Status unknown"
	}
}

func isSafeManagedName(name string) bool {
	return managedNamePattern.MatchString(strings.TrimSpace(name))
}

func isKnownSupervisorService(service string) bool {
	switch strings.TrimSpace(service) {
	case "router", "envoy", "dashboard":
		return true
	default:
		return false
	}
}
