package nlauthor

import "testing"

func TestStructuredLLMPlannerFixtures(t *testing.T) {
	t.Parallel()

	fixtures := []struct {
		name        string
		prompt      string
		response    string
		wantStatus  PlannerResultStatus
		wantWarning bool
	}{
		{
			name:       "create keyword signal",
			prompt:     `Create a keyword signal named urgent_signal with keywords "urgent", "asap"`,
			response:   `{"status":"ready","explanation":"Create a keyword signal.","intentIr":{"version":"1.0","operation":"generate","intents":[{"type":"signal","signal_type":"keyword","name":"urgent_signal","fields":{"keywords":["urgent","asap"],"operator":"any"}}]}}`,
			wantStatus: PlannerStatusReady,
		},
		{
			name:       "clarification required",
			prompt:     `Create a signal for urgent requests`,
			response:   `{"status":"needs_clarification","explanation":"Signal type is ambiguous.","clarification":{"question":"What kind of signal should be created?","options":[{"id":"keyword","label":"Keyword","description":"Match explicit keywords."},{"id":"embedding","label":"Embedding","description":"Use semantic similarity."}]}}`,
			wantStatus: PlannerStatusNeedsClarification,
		},
		{
			name:       "unsupported request",
			prompt:     `Create a pii signal`,
			response:   `{"status":"unsupported","explanation":"PII signals are not yet supported by this planner.","error":"unsupported signal type pii"}`,
			wantStatus: PlannerStatusUnsupported,
		},
	}

	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			t.Parallel()

			planner := NewStructuredLLMPlanner(DefaultSchemaManifest(), stubProvider{
				response: StructuredGenerationResponse{Content: fixture.response},
			}, RuntimeConfig{
				Backend: PlannerBackendStructuredLLM,
				Model:   "gpt-test",
			})

			result, err := planner.Plan(t.Context(), Session{}, TurnRequest{Prompt: fixture.prompt, ModeHint: OperationGenerate})
			if err != nil {
				t.Fatalf("Plan error = %v", err)
			}
			if result.Status != fixture.wantStatus {
				t.Fatalf("status = %q, want %q", result.Status, fixture.wantStatus)
			}
		})
	}
}

func TestToolCallingPlannerFixtures(t *testing.T) {
	t.Parallel()

	t.Run("modify route model after symbol lookup", func(t *testing.T) {
		t.Parallel()

		provider := &stagedProvider{
			steps: []ToolCallingResponse{
				{
					ToolCalls: []ProviderToolCall{
						{ID: "call_1", Name: "get_route", Arguments: `{"name":"support_route"}`},
					},
				},
				{
					Content: `{"status":"ready","explanation":"Update route support_route to use model gpt-4.1-mini.","intentIr":{"version":"1.0","operation":"modify","intents":[{"type":"modify","action":"update","target_construct":"route","target_name":"support_route","changes":{"models":[{"model":"gpt-4.1-mini"}]}}]}}`,
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
				BaseDSL: `ROUTE support_route {
  MODEL "gpt-4o-mini"
}`,
				Symbols: &SymbolSnapshot{
					Routes: []string{"support_route"},
					Models: []string{"gpt-4o-mini", "gpt-4.1-mini"},
				},
			},
		}, TurnRequest{Prompt: "Update support_route to use model gpt-4.1-mini", ModeHint: OperationModify})
		if err != nil {
			t.Fatalf("Plan error = %v", err)
		}
		if result.Status != PlannerStatusReady {
			t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
		}
	})

	t.Run("fix invalid threshold after diagnostics lookup", func(t *testing.T) {
		t.Parallel()

		provider := &stagedProvider{
			steps: []ToolCallingResponse{
				{
					ToolCalls: []ProviderToolCall{
						{ID: "call_1", Name: "get_diagnostics", Arguments: `{}`},
					},
				},
				{
					Content: `{"status":"needs_clarification","explanation":"The validator flagged an invalid threshold value.","clarification":{"question":"What threshold should replace the invalid value?","options":[{"id":"0.7","label":"0.7","description":"Balanced threshold."},{"id":"0.8","label":"0.8","description":"Stricter threshold."}]}}`,
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
				Diagnostics: []DiagnosticSnapshot{
					{Level: "error", Message: "threshold must be between 0 and 1", Line: 4},
				},
			},
		}, TurnRequest{Prompt: "Fix the invalid prompt_guard threshold", ModeHint: OperationFix})
		if err != nil {
			t.Fatalf("Plan error = %v", err)
		}
		if result.Status != PlannerStatusNeedsClarification {
			t.Fatalf("status = %q, want %q", result.Status, PlannerStatusNeedsClarification)
		}
		if result.Clarification == nil || len(result.Clarification.Options) != 2 {
			t.Fatalf("clarification = %+v, want threshold options", result.Clarification)
		}
	})
}
