package nlauthor

import (
	"context"
	"testing"
)

type stagedProvider struct {
	steps []ToolCallingResponse
	index int
}

func (stagedProvider) Name() string {
	return "staged-provider"
}

func (stagedProvider) Available() bool {
	return true
}

func (stagedProvider) GenerateStructured(_ context.Context, _ StructuredGenerationRequest) (StructuredGenerationResponse, error) {
	return StructuredGenerationResponse{}, nil
}

func (s *stagedProvider) GenerateToolCalls(_ context.Context, _ ToolCallingRequest) (ToolCallingResponse, error) {
	if s.index >= len(s.steps) {
		return ToolCallingResponse{}, nil
	}
	step := s.steps[s.index]
	s.index++
	return step, nil
}

func TestToolCallingPlannerUsesRegistryThenParsesFinalJSON(t *testing.T) {
	t.Parallel()

	provider := &stagedProvider{
		steps: []ToolCallingResponse{
			{
				ToolCalls: []ProviderToolCall{
					{ID: "call_1", Name: "list_symbols", Arguments: `{}`},
				},
			},
			{
				Content: `{"status":"ready","explanation":"Create a route.","intentIr":{"version":"1.0","operation":"generate","intents":[{"type":"route","name":"support_route","priority":100,"models":[{"model":"gpt-4o-mini"}],"plugins":[]}]}}`,
			},
		},
	}
	registry := NewPlannerToolRegistry(
		NewBuiltinBuilderToolSource(DefaultSchemaManifest(), NewPreviewPlanner(DefaultSchemaManifest()).Support()),
	)
	planner := NewToolCallingPlanner(DefaultSchemaManifest(), provider, registry, RuntimeConfig{
		Backend:    PlannerBackendToolCallingLLM,
		Model:      "gpt-test",
		ToolBudget: 2,
	})

	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			Symbols: &SymbolSnapshot{Models: []string{"gpt-4o-mini"}},
		},
	}, TurnRequest{Prompt: "Create route support_route for model gpt-4o-mini"})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}
	if provider.index != 2 {
		t.Fatalf("provider steps used = %d, want 2", provider.index)
	}
}

func TestToolCallingPlannerRecordsToolObserverEvents(t *testing.T) {
	t.Parallel()

	provider := &stagedProvider{
		steps: []ToolCallingResponse{
			{
				ToolCalls: []ProviderToolCall{
					{ID: "call_1", Name: "list_symbols", Arguments: `{}`},
				},
			},
			{
				Content: `{"status":"ready","explanation":"Create a route.","intentIr":{"version":"1.0","operation":"generate","intents":[{"type":"route","name":"support_route","priority":100,"models":[{"model":"gpt-4o-mini"}],"plugins":[]}]}}`,
			},
		},
	}
	registry := NewPlannerToolRegistry(
		NewBuiltinBuilderToolSource(DefaultSchemaManifest(), NewPreviewPlanner(DefaultSchemaManifest()).Support()),
	)
	observer := &recordingPlannerObserver{}
	planner := newToolCallingPlannerWithObserver(DefaultSchemaManifest(), provider, registry, RuntimeConfig{
		Backend:    PlannerBackendToolCallingLLM,
		Model:      "gpt-test",
		ToolBudget: 2,
	}, observer)

	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			Symbols: &SymbolSnapshot{Models: []string{"gpt-4o-mini"}},
		},
	}, TurnRequest{Prompt: "Create route support_route for model gpt-4o-mini", ModeHint: OperationGenerate})
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
	if len(toolCalls) != 1 {
		t.Fatalf("tool events = %d, want 1", len(toolCalls))
	}
	if turns[0].ToolCalls != 1 {
		t.Fatalf("toolCalls = %d, want 1", turns[0].ToolCalls)
	}
	if toolCalls[0].Source != PlannerToolSourceBuiltinBuilder {
		t.Fatalf("tool source = %q, want %q", toolCalls[0].Source, PlannerToolSourceBuiltinBuilder)
	}
}

func TestToolCallingPlannerRepairsInvalidFinalPlannerJSON(t *testing.T) {
	t.Parallel()

	provider := &stagedProvider{
		steps: []ToolCallingResponse{
			{
				ToolCalls: []ProviderToolCall{
					{ID: "call_1", Name: "list_symbols", Arguments: `{}`},
				},
			},
			{
				Content: `{"status":"ready","explanation":"Create a route.","warnings":[{"code":"no_plugins","message":"No plugins attached to the route."}]}`,
			},
			{
				Content: `{"status":"ready","explanation":"Create a route.","warnings":[{"code":"no_plugins","message":"No plugins attached to the route."}],"intentIr":{"version":"1.0","operation":"generate","intents":[{"type":"route","name":"support_route","priority":100,"models":[{"model":"gpt-4o-mini"}],"plugins":[]}]}}`,
			},
		},
	}
	registry := NewPlannerToolRegistry(
		NewBuiltinBuilderToolSource(DefaultSchemaManifest(), NewPreviewPlanner(DefaultSchemaManifest()).Support()),
	)
	planner := NewToolCallingPlanner(DefaultSchemaManifest(), provider, registry, RuntimeConfig{
		Backend:    PlannerBackendToolCallingLLM,
		Model:      "gpt-test",
		ToolBudget: 2,
	})

	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			Symbols: &SymbolSnapshot{Models: []string{"gpt-4o-mini"}},
		},
	}, TurnRequest{Prompt: "Create route support_route for model gpt-4o-mini"})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}
	if provider.index != 3 {
		t.Fatalf("provider steps used = %d, want 3", provider.index)
	}
	if result.IntentIR == nil || len(result.IntentIR.Intents) != 1 {
		t.Fatalf("intentIr = %+v, want one repaired intent", result.IntentIR)
	}
}
