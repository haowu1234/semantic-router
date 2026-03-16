package nlauthor

import (
	"context"
	"testing"
)

type stubProvider struct {
	response StructuredGenerationResponse
	err      error
}

func (stubProvider) Name() string {
	return "stub-provider"
}

func (stubProvider) Available() bool {
	return true
}

func (s stubProvider) GenerateStructured(_ context.Context, _ StructuredGenerationRequest) (StructuredGenerationResponse, error) {
	return s.response, s.err
}

func (s stubProvider) GenerateToolCalls(_ context.Context, _ ToolCallingRequest) (ToolCallingResponse, error) {
	return ToolCallingResponse{Content: s.response.Content}, s.err
}

func TestStructuredLLMPlannerParsesProviderJSON(t *testing.T) {
	t.Parallel()

	planner := NewStructuredLLMPlanner(DefaultSchemaManifest(), stubProvider{
		response: StructuredGenerationResponse{
			Content: `{"status":"ready","explanation":"Create a keyword signal.","intentIr":{"version":"1.0","operation":"generate","intents":[{"type":"signal","signal_type":"keyword","name":"urgent_signal","fields":{"keywords":["urgent","asap"],"operator":"any"}}]}}`,
		},
	}, RuntimeConfig{
		Backend: PlannerBackendStructuredLLM,
		Model:   "gpt-test",
	})

	result, err := planner.Plan(t.Context(), Session{}, TurnRequest{Prompt: "Create a keyword signal"})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}
	if result.IntentIR == nil || len(result.IntentIR.Intents) != 1 {
		t.Fatalf("intentIr = %+v, want one intent", result.IntentIR)
	}
}

func TestStructuredLLMPlannerRecordsObserverEvent(t *testing.T) {
	t.Parallel()

	observer := &recordingPlannerObserver{}
	planner := newStructuredLLMPlannerWithObserver(DefaultSchemaManifest(), stubProvider{
		response: StructuredGenerationResponse{
			Content: `{"status":"ready","explanation":"Create a keyword signal.","warnings":[{"code":"default_keyword_operator","message":"Using operator any."}],"intentIr":{"version":"1.0","operation":"generate","intents":[{"type":"signal","signal_type":"keyword","name":"urgent_signal","fields":{"keywords":["urgent","asap"],"operator":"any"}}]}}`,
		},
	}, RuntimeConfig{
		Backend: PlannerBackendStructuredLLM,
		Model:   "gpt-test",
	}, observer)

	result, err := planner.Plan(t.Context(), Session{}, TurnRequest{
		Prompt:   "Create a keyword signal",
		ModeHint: OperationGenerate,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}

	turns, toolCalls := observer.snapshot()
	if len(turns) != 1 {
		t.Fatalf("turn events = %d, want 1", len(turns))
	}
	if len(toolCalls) != 0 {
		t.Fatalf("tool events = %d, want 0", len(toolCalls))
	}
	if turns[0].Backend != string(PlannerBackendStructuredLLM) {
		t.Fatalf("backend = %q, want %q", turns[0].Backend, PlannerBackendStructuredLLM)
	}
	if turns[0].WarningCount != 1 {
		t.Fatalf("warningCount = %d, want 1", turns[0].WarningCount)
	}
	if turns[0].PromptChars == 0 {
		t.Fatal("promptChars = 0, want non-zero")
	}
}

func TestStructuredLLMPlannerRejectsMissingIntentIR(t *testing.T) {
	t.Parallel()

	planner := NewStructuredLLMPlanner(DefaultSchemaManifest(), stubProvider{
		response: StructuredGenerationResponse{
			Content: `{"status":"ready","explanation":"Create a keyword signal.","warnings":[{"code":"default_keyword_operator","message":"Using operator any."}]}`,
		},
	}, RuntimeConfig{
		Backend: PlannerBackendStructuredLLM,
		Model:   "gpt-test",
	})

	result, err := planner.Plan(t.Context(), Session{}, TurnRequest{Prompt: "Create a keyword signal"})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusError {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusError)
	}
	if !containsWarningCode(result.Warnings, "invalid_planner_result") {
		t.Fatalf("warnings = %+v, want invalid_planner_result", result.Warnings)
	}
	if result.Error == "" {
		t.Fatal("error is empty, want structured validation error")
	}
}

func TestNewPlannerFromRuntimeConfigMarksMissingStructuredProviderUnavailable(t *testing.T) {
	t.Parallel()

	planner := NewPlannerFromRuntimeConfig(DefaultSchemaManifest(), RuntimeConfig{
		Backend: PlannerBackendStructuredLLM,
		Model:   "gpt-test",
	})

	if planner.BackendName() != string(PlannerBackendStructuredLLM) {
		t.Fatalf("backend = %q, want %q", planner.BackendName(), PlannerBackendStructuredLLM)
	}
	if planner.Available() {
		t.Fatal("planner available = true, want false")
	}

	result, err := planner.Plan(t.Context(), Session{}, TurnRequest{Prompt: "Create a route"})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusUnsupported {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusUnsupported)
	}
}

func containsWarningCode(warnings []PlannerWarning, code string) bool {
	for _, warning := range warnings {
		if warning.Code == code {
			return true
		}
	}
	return false
}
