package nlauthor

import "testing"

func TestPreviewPlannerClarifiesAmbiguousPrompt(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{}, TurnRequest{Prompt: "help"})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusNeedsClarification {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusNeedsClarification)
	}
	if result.Clarification == nil || len(result.Clarification.Options) == 0 {
		t.Fatal("clarification is empty")
	}
}

func TestPreviewPlannerCreatesKeywordSignalIntent(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{}, TurnRequest{
		Prompt: `Create a keyword signal named urgent_signal with keywords "urgent", "asap"`,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}
	if result.IntentIR == nil || len(result.IntentIR.Intents) != 1 {
		t.Fatalf("intent count = %d, want 1", len(result.IntentIR.Intents))
	}
	intent := result.IntentIR.Intents[0]
	if intent["type"] != "signal" {
		t.Fatalf("intent type = %v, want signal", intent["type"])
	}
	if intent["signal_type"] != "keyword" {
		t.Fatalf("signal_type = %v, want keyword", intent["signal_type"])
	}
}

func TestPreviewPlannerCreatesRouteIntent(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			Symbols: &SymbolSnapshot{Models: []string{"gpt-4o-mini"}},
		},
	}, TurnRequest{
		Prompt: `Create route support_route for model gpt-4o-mini`,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}
	intent := result.IntentIR.Intents[0]
	if intent["type"] != "route" {
		t.Fatalf("intent type = %v, want route", intent["type"])
	}
	if intent["name"] != "support_route" {
		t.Fatalf("route name = %v, want support_route", intent["name"])
	}
	if _, ok := intent["condition"]; ok {
		t.Fatalf("route condition unexpectedly present: %#v", intent["condition"])
	}
}

func TestPreviewPlannerCreatesRouteIntentWithSignalCondition(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			Symbols: &SymbolSnapshot{
				Models:  []string{"gpt-4o-mini"},
				Signals: []SymbolInfo{{Name: "urgent_signal", Type: "keyword"}},
			},
		},
	}, TurnRequest{
		Prompt: `Create route support_route for model gpt-4o-mini when signal urgent_signal`,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}
	intent := result.IntentIR.Intents[0]
	condition := intent["condition"].(map[string]interface{})
	if condition["signal_name"] != "urgent_signal" {
		t.Fatalf("signal_name = %v, want urgent_signal", condition["signal_name"])
	}
}

func TestPreviewPlannerDeletesExistingBackend(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			Symbols: &SymbolSnapshot{
				Backends: []SymbolInfo{{Name: "fast_api", Type: "response_api"}},
			},
		},
	}, TurnRequest{
		Prompt: `Delete backend fast_api`,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}
	intent := result.IntentIR.Intents[0]
	if intent["type"] != "modify" {
		t.Fatalf("intent type = %v, want modify", intent["type"])
	}
	if intent["action"] != "delete" {
		t.Fatalf("action = %v, want delete", intent["action"])
	}
}

func TestPreviewPlannerUpdatesRouteModel(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			BaseDSL: "ROUTE support_route {}",
			Symbols: &SymbolSnapshot{
				Routes: []string{"support_route"},
				Models: []string{"gpt-4o-mini", "gpt-4.1-mini"},
			},
		},
	}, TurnRequest{
		Prompt: `Update route support_route to use model gpt-4.1-mini`,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}
	intent := result.IntentIR.Intents[0]
	if intent["type"] != "modify" {
		t.Fatalf("intent type = %v, want modify", intent["type"])
	}
	changes := intent["changes"].(map[string]interface{})
	models := changes["models"].([]map[string]interface{})
	if models[0]["model"] != "gpt-4.1-mini" {
		t.Fatalf("model = %v, want gpt-4.1-mini", models[0]["model"])
	}
}

func TestPreviewPlannerUpdatesRouteWithPluginRef(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			BaseDSL: "ROUTE support_route {}",
			Symbols: &SymbolSnapshot{
				Routes:  []string{"support_route"},
				Plugins: []string{"blocker"},
			},
		},
	}, TurnRequest{
		Prompt: `Update route support_route to use plugin blocker`,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}
	intent := result.IntentIR.Intents[0]
	changes := intent["changes"].(map[string]interface{})
	plugins := changes["plugins"].([]map[string]interface{})
	if plugins[0]["name"] != "blocker" {
		t.Fatalf("plugin name = %v, want blocker", plugins[0]["name"])
	}
}

func TestPreviewPlannerClarifiesMultiConstructPrompt(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{}, TurnRequest{
		Prompt: `Add pii protection to the math route`,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusNeedsClarification {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusNeedsClarification)
	}
}

func TestPreviewPlannerAdvertisesOnlySupportedSignalClarifications(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{}, TurnRequest{
		Prompt: `Create a signal for urgent requests`,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusNeedsClarification {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusNeedsClarification)
	}
	if result.Clarification == nil || len(result.Clarification.Options) == 0 {
		t.Fatal("clarification options are empty")
	}
	if containsClarificationOption(result.Clarification.Options, "pii") {
		t.Fatalf("clarification options unexpectedly contain unsupported pii type: %+v", result.Clarification.Options)
	}
	if !containsClarificationOption(result.Clarification.Options, "keyword") {
		t.Fatalf("clarification options = %+v, want keyword support option", result.Clarification.Options)
	}
}

func TestPreviewPlannerRepairClarifiesUndefinedSignalReference(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			Symbols: &SymbolSnapshot{
				Signals: []SymbolInfo{{Name: "math", Type: "domain"}},
			},
			Diagnostics: []DiagnosticSnapshot{
				{
					Level:   "warning",
					Message: `signal reference "mth" is not defined`,
				},
			},
		},
	}, TurnRequest{
		Prompt:   "Fix the invalid route draft",
		ModeHint: OperationFix,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusNeedsClarification {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusNeedsClarification)
	}
	if result.Clarification == nil || result.Clarification.Question == "" {
		t.Fatal("clarification is empty")
	}
	if len(result.Clarification.Options) == 0 {
		t.Fatal("clarification options are empty")
	}
	if result.Clarification.Options[0].Label != "math" {
		t.Fatalf("first option = %q, want math", result.Clarification.Options[0].Label)
	}
}

func TestPreviewPlannerRepairClarifiesThresholdConstraint(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			Diagnostics: []DiagnosticSnapshot{
				{
					Level:   "constraint",
					Message: "prompt_guard threshold must be between 0 and 1",
				},
			},
		},
	}, TurnRequest{
		Prompt:   "Repair the invalid threshold",
		ModeHint: OperationFix,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusNeedsClarification {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusNeedsClarification)
	}
	if result.Clarification == nil || len(result.Clarification.Options) < 3 {
		t.Fatal("threshold clarification options are empty")
	}
}

func TestPreviewPlannerRepairFixesThresholdWhenChoiceProvided(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			BaseDSL: `GLOBAL {
  prompt_guard: {
    enabled: true
    threshold: 1.5
  }
}`,
			Diagnostics: []DiagnosticSnapshot{
				{
					Level:   "constraint",
					Message: "prompt_guard threshold must be between 0 and 1",
				},
			},
		},
	}, TurnRequest{
		Prompt:   "Repair the invalid threshold. Use threshold 0.7.",
		ModeHint: OperationFix,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}
	if result.IntentIR == nil || len(result.IntentIR.Intents) != 1 {
		t.Fatal("threshold repair intent IR is empty")
	}
	fields := result.IntentIR.Intents[0]["fields"].(map[string]interface{})
	promptGuard := fields["prompt_guard"].(map[string]interface{})
	if promptGuard["threshold"] != 0.7 {
		t.Fatalf("threshold = %v, want 0.7", promptGuard["threshold"])
	}
}

func TestPreviewPlannerRepairClarifiesUnknownAlgorithmType(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			Diagnostics: []DiagnosticSnapshot{
				{
					Level:   "error",
					Message: `route test: unknown algorithm type "nonexistent_algo"`,
				},
			},
		},
	}, TurnRequest{
		Prompt:   "Repair the unknown algorithm",
		ModeHint: OperationFix,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusNeedsClarification {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusNeedsClarification)
	}
	if result.Clarification == nil || len(result.Clarification.Options) == 0 {
		t.Fatal("algorithm clarification options are empty")
	}
}

func TestPreviewPlannerRepairFixesUndefinedSignalWhenChoiceProvided(t *testing.T) {
	t.Parallel()

	planner := NewPreviewPlanner(DefaultSchemaManifest())
	result, err := planner.Plan(t.Context(), Session{
		Context: SessionContext{
			BaseDSL: `ROUTE support_route {
  PRIORITY 100

  WHEN domain("mth")

  MODEL "gpt-4o-mini"
}`,
			Symbols: &SymbolSnapshot{
				Signals: []SymbolInfo{{Name: "math", Type: "domain"}},
			},
			Diagnostics: []DiagnosticSnapshot{
				{
					Level:   "warning",
					Message: `signal reference "mth" is not defined`,
				},
			},
		},
	}, TurnRequest{
		Prompt:   "Fix the invalid route. Use signal math.",
		ModeHint: OperationFix,
	})
	if err != nil {
		t.Fatalf("Plan error = %v", err)
	}
	if result.Status != PlannerStatusReady {
		t.Fatalf("status = %q, want %q", result.Status, PlannerStatusReady)
	}
	if result.IntentIR == nil || len(result.IntentIR.Intents) != 1 {
		t.Fatal("signal repair intent IR is empty")
	}
	intent := result.IntentIR.Intents[0]
	if intent["type"] != "route" {
		t.Fatalf("intent type = %v, want route", intent["type"])
	}
	condition := intent["condition"].(map[string]interface{})
	if condition["signal_name"] != "math" {
		t.Fatalf("signal_name = %v, want math", condition["signal_name"])
	}
}

func containsClarificationOption(options []ClarificationOption, id string) bool {
	for _, option := range options {
		if option.ID == id {
			return true
		}
	}
	return false
}
