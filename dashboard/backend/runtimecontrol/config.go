package runtimecontrol

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const defaultEnvoyConfigPath = "/etc/envoy/envoy.yaml"

// PropagateConfig recompiles and applies runtime config using the current runtime topology.
func (c *Controller) PropagateConfig(configPath string, configDir string) error {
	if c.RunningInContainer() && c.SingleContainerTopology() && isManagedContainerConfigPath(configPath) {
		if err := c.regenerateRouterConfigSync(configPath, configDir); err != nil {
			return err
		}
		return c.regenerateAndReloadEnvoyLocally(configPath)
	}

	if c.SingleContainerTopology() && c.DockerContainerStatus() == "running" {
		return c.propagateConfigToManagedContainer()
	}

	if err := c.regenerateRouterConfigSync(configPath, configDir); err != nil {
		return err
	}

	if c.SingleContainerTopology() {
		return nil
	}

	if err := c.regenerateEnvoyConfigLocally(configPath); err != nil {
		return err
	}

	return c.restartManagedServiceContainers(ServiceRouter, ServiceEnvoy)
}

// RestartLocalServices restarts supervisor-managed services when the dashboard is colocated with them.
func (c *Controller) RestartLocalServices(services ...string) error {
	if !c.SingleContainerTopology() {
		return c.restartManagedServiceContainers(services...)
	}

	if _, err := exec.LookPath("supervisorctl"); err != nil {
		return nil
	}

	for _, service := range services {
		if err := restartOrStartSupervisorService(service, 20*time.Second); err != nil {
			return err
		}
	}
	return nil
}

// GenerateRouterConfigWithPython recompiles router-config.yaml from the user-facing config.yaml.
func GenerateRouterConfigWithPython(configPath string, outputDir string) (string, error) {
	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return "SKIP: Python CLI not available, skipping router config regeneration", nil
	}

	pythonScript := fmt.Sprintf(`
import sys
sys.path.insert(0, %q)
try:
    from cli.commands.serve import generate_router_config
    result = generate_router_config(%q, %q, force=True)
    print(f"Regenerated router config: {result}")
except ImportError:
    print("SKIP: Python CLI not available, skipping router config regeneration")
except Exception as e:
    print(f"ERROR: Failed to regenerate router config: {e}", file=sys.stderr)
    sys.exit(1)
`, cliRoot, configPath, outputDir)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "python3", "-c", pythonScript)
	cmd.Dir = filepath.Dir(configPath)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func resumeSplitRuntimeAfterSetupWithPython(configPath string) (string, error) {
	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return "SKIP: Python CLI not available, skipping split-runtime resume", nil
	}

	resumeConfigPath := resolveSplitRuntimeResumeConfigPath(configPath)

	pythonScript := fmt.Sprintf(`
import os
import sys
sys.path.insert(0, %q)
try:
    from cli.runtime_stack import resume_split_runtime_after_setup
    resume_split_runtime_after_setup(
        config_file=%q,
        env_vars=dict(os.environ),
        image=os.getenv("VLLM_SR_ROUTER_IMAGE"),
        pull_policy=os.getenv("VLLM_SR_IMAGE_PULL_POLICY"),
        network_name=os.getenv("VLLM_SR_RUNTIME_NETWORK_NAME"),
        openclaw_network_name=os.getenv("VLLM_SR_OPENCLAW_NETWORK_NAME"),
    )
    print("Split runtime resumed after setup activation")
except ImportError:
    print("SKIP: Python CLI not available, skipping split-runtime resume")
except Exception as e:
    print(f"ERROR: Failed to resume split runtime: {e}", file=sys.stderr)
    sys.exit(1)
`, cliRoot, resumeConfigPath)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "python3", "-c", pythonScript)
	cmd.Dir = filepath.Dir(configPath)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func resolveSplitRuntimeResumeConfigPath(configPath string) string {
	if hostConfigPath := strings.TrimSpace(os.Getenv("VLLM_SR_HOST_CONFIG_PATH")); hostConfigPath != "" {
		return hostConfigPath
	}
	return configPath
}

func generateEnvoyConfigWithPython(configPath string, outputPath string) (string, error) {
	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return "SKIP: Python CLI not available, skipping Envoy config regeneration", nil
	}

	pythonScript := fmt.Sprintf(`
import sys
sys.path.insert(0, %q)
try:
    from cli.config_generator import generate_envoy_config_from_user_config
    from cli.parser import parse_user_config
    user_config = parse_user_config(%q)
    generate_envoy_config_from_user_config(user_config, %q)
    print("Regenerated Envoy config: %s")
except ImportError:
    print("SKIP: Python CLI not available, skipping Envoy config regeneration")
except Exception as e:
    print(f"ERROR: Failed to regenerate Envoy config: {e}", file=sys.stderr)
    sys.exit(1)
`, cliRoot, configPath, outputPath, outputPath)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "python3", "-c", pythonScript)
	cmd.Dir = filepath.Dir(configPath)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func (c *Controller) regenerateRouterConfigSync(configPath string, configDir string) error {
	outputDir := filepath.Join(configDir, ".vllm-sr")
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		log.Printf("Config propagation: .vllm-sr directory not found at %s, skipping router config regeneration (dev mode?)", outputDir)
		return nil
	}

	output, err := GenerateRouterConfigWithPython(configPath, outputDir)
	if err != nil {
		return fmt.Errorf("failed to regenerate router config: %w (output: %s)", err, strings.TrimSpace(output))
	}

	log.Printf("Config propagation: %s", strings.TrimSpace(output))
	return nil
}

func (c *Controller) regenerateAndReloadEnvoyLocally(configPath string) error {
	if err := c.regenerateEnvoyConfigLocally(configPath); err != nil {
		return err
	}

	if _, err := exec.LookPath("supervisorctl"); err != nil {
		log.Printf("Config propagation: supervisorctl not available, skipping managed Envoy reload")
		return nil
	}

	if err := restartOrStartSupervisorService("envoy", 20*time.Second); err != nil {
		return fmt.Errorf("failed to restart Envoy: %w", err)
	}
	return nil
}

func (c *Controller) regenerateEnvoyConfigLocally(configPath string) error {
	envoyConfigPath := detectEnvoyConfigPath()
	if envoyConfigPath == "" {
		log.Printf("Config propagation: Envoy config path not found, skipping managed Envoy reload")
		return nil
	}

	output, err := generateEnvoyConfigWithPython(configPath, envoyConfigPath)
	if err != nil {
		return fmt.Errorf("failed to regenerate Envoy config: %w (output: %s)", err, strings.TrimSpace(output))
	}
	log.Printf("Config propagation: %s", strings.TrimSpace(output))
	return nil
}

func (c *Controller) propagateConfigToManagedContainer() error {
	containerName := c.ContainerName()

	if output, err := c.generateRouterConfigInManagedContainer(); err != nil {
		return fmt.Errorf("failed to regenerate router config in %s: %w (output: %s)", containerName, err, strings.TrimSpace(output))
	} else {
		log.Printf("Config propagation: %s", strings.TrimSpace(output))
	}

	if output, err := c.generateEnvoyConfigInManagedContainer(); err != nil {
		return fmt.Errorf("failed to regenerate Envoy config in %s: %w (output: %s)", containerName, err, strings.TrimSpace(output))
	} else {
		log.Printf("Config propagation: %s", strings.TrimSpace(output))
	}

	if err := c.restartOrStartManagedContainerService("envoy", 20*time.Second); err != nil {
		return fmt.Errorf("failed to restart Envoy in %s: %w", containerName, err)
	}
	return nil
}

func (c *Controller) generateRouterConfigInManagedContainer() (string, error) {
	pythonScript := `
from cli.commands.serve import generate_router_config
result = generate_router_config("/app/config.yaml", "/app/.vllm-sr", force=True)
print(f"Regenerated router config: {result}")
`
	return c.execInManagedContainer(30*time.Second, "python3", "-c", pythonScript)
}

func (c *Controller) generateEnvoyConfigInManagedContainer() (string, error) {
	return c.execInManagedContainer(30*time.Second, "python3", "-m", "cli.config_generator", "/app/config.yaml", defaultEnvoyConfigPath)
}

func (c *Controller) execInManagedContainer(timeout time.Duration, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	if err := validateManagedContainerExecArgs(args); err != nil {
		return "", err
	}

	commandArgs := append([]string{"exec", c.containerName}, args...)
	// #nosec G204 -- command args are validated above and the container name is controller-scoped.
	cmd := containerRuntimeExecContext(ctx, commandArgs...)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func (c *Controller) restartOrStartManagedContainerService(service string, timeout time.Duration) error {
	if output, err := c.execInManagedContainer(15*time.Second, "supervisorctl", "restart", service); err != nil {
		startOutput, startErr := c.execInManagedContainer(15*time.Second, "supervisorctl", "start", service)
		if startErr != nil {
			return fmt.Errorf("%s restart failed: %s / start failed: %s", service, strings.TrimSpace(output), strings.TrimSpace(startOutput))
		}
	}

	return c.waitForManagedContainerService(service, timeout)
}

func (c *Controller) waitForManagedContainerService(service string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	lastStatus := ""

	for time.Now().Before(deadline) {
		output, err := c.execInManagedContainer(10*time.Second, "supervisorctl", "status", service)
		lastStatus = strings.TrimSpace(output)
		if err == nil && strings.Contains(lastStatus, "RUNNING") {
			return nil
		}
		if strings.Contains(lastStatus, "FATAL") || strings.Contains(lastStatus, "EXITED") || strings.Contains(lastStatus, "BACKOFF") {
			return fmt.Errorf("%s failed to start: %s", service, lastStatus)
		}
		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("timed out waiting for %s in %s to become RUNNING (last status: %s)", service, c.containerName, lastStatus)
}

func (c *Controller) restartManagedServiceContainers(services ...string) error {
	restartedContainers := map[string]bool{}
	for _, serviceName := range services {
		service, ok := c.Service(serviceName)
		if !ok {
			continue
		}
		containerName := strings.TrimSpace(service.ContainerName)
		if containerName == "" || restartedContainers[containerName] {
			continue
		}
		if c.registry.IsSharedContainerService(serviceName) {
			continue
		}

		status := c.ServiceContainerStatus(serviceName)
		if status == "not found" {
			continue
		}

		if err := c.restartManagedContainer(containerName, 30*time.Second); err != nil {
			return fmt.Errorf("failed to restart %s container %s: %w", serviceName, containerName, err)
		}
		restartedContainers[containerName] = true
	}
	return nil
}

func (c *Controller) ResumeSplitSetupRuntime(configPath string) error {
	if c.SingleContainerTopology() {
		return nil
	}

	output, err := resumeSplitRuntimeAfterSetupWithPython(configPath)
	if err != nil {
		return fmt.Errorf("failed to resume split runtime after setup activation: %w (output: %s)", err, strings.TrimSpace(output))
	}

	if err := c.verifySplitRuntimeServicesRunning(ServiceRouter, ServiceEnvoy); err != nil {
		return err
	}

	log.Printf("Setup activation: %s", strings.TrimSpace(output))
	return nil
}

func (c *Controller) verifySplitRuntimeServicesRunning(serviceNames ...string) error {
	for _, serviceName := range serviceNames {
		status := c.ServiceContainerStatus(serviceName)
		if status != "running" {
			return fmt.Errorf(
				"split runtime service %s is %s after setup activation",
				serviceName,
				status,
			)
		}
	}
	return nil
}

func (c *Controller) restartManagedContainer(containerName string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	if !isSafeManagedName(containerName) {
		return fmt.Errorf("unsupported container name %q", containerName)
	}

	// #nosec G204 -- the managed container name is restricted to a safe character set.
	output, err := containerRuntimeExecContext(ctx, "restart", containerName).CombinedOutput()
	if err != nil {
		return fmt.Errorf("container restart failed: %s", strings.TrimSpace(string(output)))
	}

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if c.dockerContainerStatus(containerName) == "running" {
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("timed out waiting for container %s to become running", containerName)
}

func restartOrStartSupervisorService(service string, timeout time.Duration) error {
	if !isKnownSupervisorService(service) {
		return fmt.Errorf("unsupported supervisor service: %s", service)
	}

	// #nosec G204 -- supervisor service names are restricted to the managed allowlist.
	cmd := exec.Command("supervisorctl", "restart", service)
	if output, err := cmd.CombinedOutput(); err != nil {
		// #nosec G204 -- supervisor service names are restricted to the managed allowlist.
		startCmd := exec.Command("supervisorctl", "start", service)
		if startOutput, startErr := startCmd.CombinedOutput(); startErr != nil {
			return fmt.Errorf("%s restart failed: %s / start failed: %s", service, strings.TrimSpace(string(output)), strings.TrimSpace(string(startOutput)))
		}
	}

	return waitForSupervisorService(service, timeout)
}

func waitForSupervisorService(service string, timeout time.Duration) error {
	if !isKnownSupervisorService(service) {
		return fmt.Errorf("unsupported supervisor service: %s", service)
	}

	deadline := time.Now().Add(timeout)
	lastStatus := ""

	for time.Now().Before(deadline) {
		// #nosec G204 -- supervisor service names are restricted to the managed allowlist.
		output, err := exec.Command("supervisorctl", "status", service).CombinedOutput()
		lastStatus = strings.TrimSpace(string(output))
		if err == nil && strings.Contains(lastStatus, "RUNNING") {
			return nil
		}
		if strings.Contains(lastStatus, "FATAL") || strings.Contains(lastStatus, "EXITED") || strings.Contains(lastStatus, "BACKOFF") {
			return fmt.Errorf("%s failed to start: %s", service, lastStatus)
		}
		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("timed out waiting for %s to become RUNNING (last status: %s)", service, lastStatus)
}

func validateManagedContainerExecArgs(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("managed container command is required")
	}

	switch args[0] {
	case "python3":
		return validateManagedContainerPythonArgs(args)
	case "supervisorctl":
		return validateManagedContainerSupervisorArgs(args)
	default:
		return fmt.Errorf("unsupported managed container command: %s", args[0])
	}
}

func validateManagedContainerPythonArgs(args []string) error {
	if len(args) == 3 && args[1] == "-c" {
		return nil
	}

	if len(args) == 5 &&
		args[1] == "-m" &&
		args[2] == "cli.config_generator" &&
		args[3] == "/app/config.yaml" &&
		args[4] == defaultEnvoyConfigPath {
		return nil
	}

	return fmt.Errorf("unsupported python3 invocation in managed container")
}

func validateManagedContainerSupervisorArgs(args []string) error {
	if len(args) != 3 {
		return fmt.Errorf("unsupported supervisorctl invocation in managed container")
	}

	switch args[1] {
	case "restart", "start", "status":
	default:
		return fmt.Errorf("unsupported supervisorctl action: %s", args[1])
	}

	switch args[2] {
	case "router", "envoy", "dashboard":
		return nil
	default:
		return fmt.Errorf("unsupported supervisorctl service: %s", args[2])
	}
}

func detectEnvoyConfigPath() string {
	var candidates []string
	if envPath := strings.TrimSpace(os.Getenv("VLLM_SR_ENVOY_CONFIG_PATH")); envPath != "" {
		candidates = append(candidates, envPath)
	}
	candidates = append(candidates, defaultEnvoyConfigPath)

	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if info, err := os.Stat(filepath.Dir(candidate)); err == nil && info.IsDir() {
			return candidate
		}
	}

	return ""
}

func detectPythonCLIRoot() string {
	var candidates []string
	if envPath := strings.TrimSpace(os.Getenv("VLLM_SR_CLI_PATH")); envPath != "" {
		candidates = append(candidates, envPath)
	}
	candidates = append(candidates, "/app")

	if wd, err := os.Getwd(); err == nil {
		candidates = append(
			candidates,
			filepath.Clean(filepath.Join(wd, "..", "..", "..", "src", "vllm-sr")),
			filepath.Clean(filepath.Join(wd, "..", "..", "src", "vllm-sr")),
			filepath.Clean(filepath.Join(wd, "src", "vllm-sr")),
		)
	}
	if _, thisFile, _, ok := runtime.Caller(0); ok {
		candidates = append(
			candidates,
			filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", "..", "src", "vllm-sr")),
		)
	}

	seen := map[string]bool{}
	for _, candidate := range candidates {
		if candidate == "" || seen[candidate] {
			continue
		}
		seen[candidate] = true
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate
		}
	}

	return ""
}

func isManagedContainerConfigPath(configPath string) bool {
	return filepath.Clean(configPath) == "/app/config.yaml"
}
