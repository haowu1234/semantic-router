package runtimecontrol

import "testing"

func TestResolveSplitRuntimeResumeConfigPathPrefersHostPath(t *testing.T) {
	t.Setenv("VLLM_SR_HOST_CONFIG_PATH", "/host/workspace/config.yaml")

	got := resolveSplitRuntimeResumeConfigPath("/app/workspace/config.yaml")
	if got != "/host/workspace/config.yaml" {
		t.Fatalf("resume config path = %q, want %q", got, "/host/workspace/config.yaml")
	}
}

func TestResolveSplitRuntimeResumeConfigPathFallsBackToContainerPath(t *testing.T) {
	t.Setenv("VLLM_SR_HOST_CONFIG_PATH", "")

	got := resolveSplitRuntimeResumeConfigPath("/app/workspace/config.yaml")
	if got != "/app/workspace/config.yaml" {
		t.Fatalf("resume config path = %q, want %q", got, "/app/workspace/config.yaml")
	}
}
