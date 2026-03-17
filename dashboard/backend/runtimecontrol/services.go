package runtimecontrol

import (
	"os"
	"strings"
)

const (
	ServiceAll         = "all"
	ServiceRouter      = "router"
	ServiceEnvoy       = "envoy"
	ServiceDashboard   = "dashboard"
	ServiceDashboardDB = "dashboard-db"
)

// RuntimeService describes one managed local-runtime service.
type RuntimeService struct {
	Name              string
	DisplayName       string
	ContainerName     string
	SupervisorService string
	LogFiles          []string
	Component         string
}

// ServiceRegistry captures the current local runtime topology.
type ServiceRegistry struct {
	primaryContainer string
	services         []RuntimeService
}

func DefaultRegistry() ServiceRegistry {
	primary := sanitizeManagedName(
		strings.TrimSpace(os.Getenv("VLLM_SR_CONTAINER_NAME")),
		fallbackContainerName,
	)

	routerContainer := sanitizeManagedName(
		firstNonEmpty(os.Getenv("VLLM_SR_ROUTER_CONTAINER_NAME"), primary),
		primary,
	)
	envoyContainer := sanitizeManagedName(
		firstNonEmpty(os.Getenv("VLLM_SR_ENVOY_CONTAINER_NAME"), primary),
		primary,
	)
	dashboardContainer := sanitizeManagedName(
		firstNonEmpty(os.Getenv("VLLM_SR_DASHBOARD_CONTAINER_NAME"), primary),
		primary,
	)
	dashboardDBContainer := sanitizeManagedName(
		firstNonEmpty(
			os.Getenv("VLLM_SR_DASHBOARD_DB_CONTAINER_NAME"),
			os.Getenv("VLLM_SR_DB_CONTAINER_NAME"),
		),
		"",
	)

	services := []RuntimeService{
		{
			Name:              ServiceRouter,
			DisplayName:       "Router",
			ContainerName:     routerContainer,
			SupervisorService: ServiceRouter,
			LogFiles: []string{
				"/var/log/supervisor/router.log",
				"/var/log/supervisor/router-error.log",
			},
			Component: "container",
		},
		{
			Name:              ServiceEnvoy,
			DisplayName:       "Envoy",
			ContainerName:     envoyContainer,
			SupervisorService: ServiceEnvoy,
			LogFiles: []string{
				"/var/log/supervisor/envoy.log",
				"/var/log/supervisor/envoy-error.log",
			},
			Component: "proxy",
		},
		{
			Name:              ServiceDashboard,
			DisplayName:       "Dashboard",
			ContainerName:     dashboardContainer,
			SupervisorService: ServiceDashboard,
			LogFiles: []string{
				"/var/log/supervisor/dashboard.log",
				"/var/log/supervisor/dashboard-error.log",
			},
			Component: "container",
		},
	}

	if dashboardDBContainer != "" {
		services = append(services, RuntimeService{
			Name:          ServiceDashboardDB,
			DisplayName:   "Dashboard DB",
			ContainerName: dashboardDBContainer,
			Component:     "database",
		})
	}

	return ServiceRegistry{
		primaryContainer: primary,
		services:         services,
	}
}

func NewServiceRegistryForContainer(containerName string) ServiceRegistry {
	primary := sanitizeManagedName(strings.TrimSpace(containerName), fallbackContainerName)
	return ServiceRegistry{
		primaryContainer: primary,
		services: []RuntimeService{
			{
				Name:              ServiceRouter,
				DisplayName:       "Router",
				ContainerName:     primary,
				SupervisorService: ServiceRouter,
				LogFiles: []string{
					"/var/log/supervisor/router.log",
					"/var/log/supervisor/router-error.log",
				},
				Component: "container",
			},
			{
				Name:              ServiceEnvoy,
				DisplayName:       "Envoy",
				ContainerName:     primary,
				SupervisorService: ServiceEnvoy,
				LogFiles: []string{
					"/var/log/supervisor/envoy.log",
					"/var/log/supervisor/envoy-error.log",
				},
				Component: "proxy",
			},
			{
				Name:              ServiceDashboard,
				DisplayName:       "Dashboard",
				ContainerName:     primary,
				SupervisorService: ServiceDashboard,
				LogFiles: []string{
					"/var/log/supervisor/dashboard.log",
					"/var/log/supervisor/dashboard-error.log",
				},
				Component: "container",
			},
		},
	}
}

func (r ServiceRegistry) PrimaryContainerName() string {
	return r.primaryContainer
}

func (r ServiceRegistry) Services() []RuntimeService {
	out := make([]RuntimeService, 0, len(r.services))
	out = append(out, r.services...)
	return out
}

func (r ServiceRegistry) Service(name string) (RuntimeService, bool) {
	normalized := strings.TrimSpace(name)
	for _, service := range r.services {
		if service.Name == normalized {
			return service, true
		}
	}
	return RuntimeService{}, false
}

func (r ServiceRegistry) SingleContainerTopology() bool {
	containerNames := map[string]struct{}{}
	for _, service := range r.services {
		if service.Name == ServiceDashboardDB || strings.TrimSpace(service.ContainerName) == "" {
			continue
		}
		containerNames[service.ContainerName] = struct{}{}
	}
	return len(containerNames) <= 1
}

func (r ServiceRegistry) DeploymentType() string {
	if r.SingleContainerTopology() {
		return "docker"
	}
	return "docker-split"
}

func (r ServiceRegistry) IsSharedContainerService(name string) bool {
	service, ok := r.Service(name)
	if !ok {
		return false
	}
	return service.ContainerName != "" && service.ContainerName == r.primaryContainer
}

func sanitizeManagedName(name string, fallback string) string {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return fallback
	}
	if !isSafeManagedName(trimmed) {
		return fallback
	}
	return trimmed
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}
