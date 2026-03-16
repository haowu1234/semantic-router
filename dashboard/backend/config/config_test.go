package config

import (
	"flag"
	"os"
	"testing"
)

func TestLoadConfigAppliesNLPlannerFlags(t *testing.T) {
	t.Parallel()

	oldCommandLine := flag.CommandLine
	oldArgs := os.Args
	t.Cleanup(func() {
		flag.CommandLine = oldCommandLine
		os.Args = oldArgs
	})

	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	os.Args = []string{
		"dashboard-backend",
		"-nl-planner-backend=structured-llm",
		"-nl-planner-provider=openai-compatible",
		"-nl-planner-base-url=http://planner.local/v1",
		"-nl-planner-api-key=test-key",
		"-nl-planner-model=gpt-test",
		"-nl-planner-timeout-ms=22000",
		"-nl-planner-max-output-tokens=2048",
		"-nl-planner-tool-budget=6",
		"-nl-planner-allow-web-tools=true",
		"-nl-planner-allow-mcp-tools=true",
		"-nl-planner-allowed-mcp-tools=tool_a,tool_b",
	}

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig error = %v", err)
	}

	if cfg.NLPlannerBackend != "structured-llm" {
		t.Fatalf("NLPlannerBackend = %q, want structured-llm", cfg.NLPlannerBackend)
	}
	if cfg.NLPlannerProvider != "openai-compatible" {
		t.Fatalf("NLPlannerProvider = %q, want openai-compatible", cfg.NLPlannerProvider)
	}
	if cfg.NLPlannerBaseURL != "http://planner.local/v1" {
		t.Fatalf("NLPlannerBaseURL = %q, want http://planner.local/v1", cfg.NLPlannerBaseURL)
	}
	if cfg.NLPlannerAPIKey != "test-key" {
		t.Fatalf("NLPlannerAPIKey = %q, want test-key", cfg.NLPlannerAPIKey)
	}
	if cfg.NLPlannerModel != "gpt-test" {
		t.Fatalf("NLPlannerModel = %q, want gpt-test", cfg.NLPlannerModel)
	}
	if cfg.NLPlannerTimeoutMs != 22000 {
		t.Fatalf("NLPlannerTimeoutMs = %d, want 22000", cfg.NLPlannerTimeoutMs)
	}
	if cfg.NLPlannerMaxOutputTokens != 2048 {
		t.Fatalf("NLPlannerMaxOutputTokens = %d, want 2048", cfg.NLPlannerMaxOutputTokens)
	}
	if cfg.NLPlannerToolBudget != 6 {
		t.Fatalf("NLPlannerToolBudget = %d, want 6", cfg.NLPlannerToolBudget)
	}
	if !cfg.NLPlannerAllowWebTools {
		t.Fatal("NLPlannerAllowWebTools = false, want true")
	}
	if !cfg.NLPlannerAllowMCPTools {
		t.Fatal("NLPlannerAllowMCPTools = false, want true")
	}
	if cfg.NLPlannerAllowedMCPTools != "tool_a,tool_b" {
		t.Fatalf("NLPlannerAllowedMCPTools = %q, want tool_a,tool_b", cfg.NLPlannerAllowedMCPTools)
	}
}
