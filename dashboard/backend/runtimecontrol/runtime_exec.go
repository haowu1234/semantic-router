package runtimecontrol

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func containerRuntimeCommand() string {
	candidates := []string{
		strings.TrimSpace(os.Getenv("VLLM_SR_CONTAINER_RUNTIME")),
		strings.TrimSpace(os.Getenv("OPENCLAW_CONTAINER_RUNTIME")),
		strings.TrimSpace(os.Getenv("CONTAINER_RUNTIME")),
		"docker",
		"podman",
	}

	seen := make(map[string]bool, len(candidates))
	for _, candidate := range candidates {
		if candidate == "" || seen[candidate] {
			continue
		}
		seen[candidate] = true

		if filepath.IsAbs(candidate) {
			if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
				return candidate
			}
			continue
		}

		if resolved, err := exec.LookPath(candidate); err == nil {
			return resolved
		}
	}

	return "docker"
}

func containerRuntimeExec(args ...string) *exec.Cmd {
	return exec.Command(containerRuntimeCommand(), args...)
}

func containerRuntimeExecContext(ctx context.Context, args ...string) *exec.Cmd {
	return exec.CommandContext(ctx, containerRuntimeCommand(), args...)
}
