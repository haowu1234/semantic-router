package router

import (
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestNewNLAuthoringServiceUsesConfiguredStructuredPlanner(t *testing.T) {
	t.Parallel()

	service := newNLAuthoringService(&config.Config{
		NLPlannerBackend:         "structured-llm",
		NLPlannerProvider:        "openai-compatible",
		NLPlannerBaseURL:         "http://planner.local/v1",
		NLPlannerModel:           "gpt-test",
		NLPlannerTimeoutMs:       18000,
		NLPlannerMaxOutputTokens: 1500,
	}, nil)

	capabilities := service.Capabilities(false)
	if capabilities.PlannerBackend != "structured-llm" {
		t.Fatalf("plannerBackend = %q, want structured-llm", capabilities.PlannerBackend)
	}
	if !capabilities.PlannerAvailable {
		t.Fatalf("plannerAvailable = false, want true")
	}
}
