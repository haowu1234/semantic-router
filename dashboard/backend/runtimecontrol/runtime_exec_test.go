package runtimecontrol

import (
	"os"
	"path/filepath"
	"testing"
)

func TestContainerRuntimeCommandPrefersExplicitPath(t *testing.T) {
	runtimePath := filepath.Join(t.TempDir(), "docker")
	if err := os.WriteFile(runtimePath, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatalf("write runtime shim: %v", err)
	}

	t.Setenv("VLLM_SR_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", "")
	t.Setenv("CONTAINER_RUNTIME", "")

	if got := containerRuntimeCommand(); got != runtimePath {
		t.Fatalf("containerRuntimeCommand() = %q, want %q", got, runtimePath)
	}
}

func TestContainerRuntimeCommandFallsBackToOpenClawRuntime(t *testing.T) {
	runtimePath := filepath.Join(t.TempDir(), "docker")
	if err := os.WriteFile(runtimePath, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatalf("write runtime shim: %v", err)
	}

	t.Setenv("VLLM_SR_CONTAINER_RUNTIME", "")
	t.Setenv("OPENCLAW_CONTAINER_RUNTIME", runtimePath)
	t.Setenv("CONTAINER_RUNTIME", "")

	if got := containerRuntimeCommand(); got != runtimePath {
		t.Fatalf("containerRuntimeCommand() = %q, want %q", got, runtimePath)
	}
}
