package sessionaffinity

import "testing"

func TestManagerEvaluateNewSessionBindsSelectorModel(t *testing.T) {
	store := NewMemoryStore()
	manager := NewManager(Config{Enabled: true}, store)

	eval, err := manager.Evaluate(Request{
		UserID:        "user-1",
		SessionID:     "session-1",
		SelectorModel: "model-b",
		Candidates: []Candidate{
			{Model: "model-a", Score: 0.3},
			{Model: "model-b", Score: 0.5},
		},
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if eval.Action != ActionBind {
		t.Fatalf("Action = %s, want %s", eval.Action, ActionBind)
	}
	if eval.Reason != ReasonNewSession {
		t.Fatalf("Reason = %s, want %s", eval.Reason, ReasonNewSession)
	}
	if len(eval.PreferredModels) == 0 || eval.PreferredModels[0] != "model-b" {
		t.Fatalf("PreferredModels = %v, want model-b first", eval.PreferredModels)
	}
}

func TestManagerEvaluateMomentumHoldThenRelease(t *testing.T) {
	store := NewMemoryStore()
	manager := NewManager(Config{
		Enabled:                   true,
		ReleaseAfterPendingTurns:  2,
		ImmediateUpgradeThreshold: 0.2,
	}, store)

	initialEval := &Evaluation{
		Enabled:       true,
		Key:           "user-1:session-1",
		UserID:        "user-1",
		SessionID:     "session-1",
		SelectorModel: "model-b",
		PriorModel:    "model-a",
		Action:        ActionStick,
		Reason:        ReasonMomentumHold,
		Request: Request{
			UserID:        "user-1",
			SessionID:     "session-1",
			SelectorModel: "model-b",
			DecisionName:  "chat",
		},
	}
	if err := manager.Commit(initialEval, "model-a", ActionStick, ReasonMomentumHold); err != nil {
		t.Fatalf("Commit() error = %v", err)
	}

	firstEval, err := manager.Evaluate(Request{
		UserID:        "user-1",
		SessionID:     "session-1",
		SelectorModel: "model-b",
		Candidates: []Candidate{
			{Model: "model-a", Score: 0.40},
			{Model: "model-b", Score: 0.45},
		},
	})
	if err != nil {
		t.Fatalf("first Evaluate() error = %v", err)
	}
	if firstEval.Action != ActionSwitch {
		t.Fatalf("first Action = %s, want %s", firstEval.Action, ActionSwitch)
	}
	if firstEval.Reason != ReasonMomentumRelease {
		t.Fatalf("first Reason = %s, want %s", firstEval.Reason, ReasonMomentumRelease)
	}
}

func TestManagerEvaluateNegativeFeedbackSwitches(t *testing.T) {
	store := NewMemoryStore()
	manager := NewManager(Config{Enabled: true}, store)

	if err := store.Put(&State{
		Key:        "user-1:session-1",
		UserID:     "user-1",
		SessionID:  "session-1",
		BoundModel: "model-a",
	}, 0); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	eval, err := manager.Evaluate(Request{
		UserID:        "user-1",
		SessionID:     "session-1",
		SelectorModel: "model-b",
		Candidates: []Candidate{
			{Model: "model-a", Score: 0.3},
			{Model: "model-b", Score: 0.4},
		},
		MatchedFeedbackSignal: []string{"wrong_answer"},
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if eval.Action != ActionSwitch {
		t.Fatalf("Action = %s, want %s", eval.Action, ActionSwitch)
	}
	if eval.Reason != ReasonNegativeFeedback {
		t.Fatalf("Reason = %s, want %s", eval.Reason, ReasonNegativeFeedback)
	}
	if len(eval.PreferredModels) == 0 || eval.PreferredModels[0] != "model-b" {
		t.Fatalf("PreferredModels = %v, want model-b first", eval.PreferredModels)
	}
}

func TestManagerEvaluateUpgradeThresholdSwitches(t *testing.T) {
	store := NewMemoryStore()
	manager := NewManager(Config{
		Enabled:                   true,
		ImmediateUpgradeThreshold: 0.15,
	}, store)

	if err := store.Put(&State{
		Key:        "user-1:session-1",
		UserID:     "user-1",
		SessionID:  "session-1",
		BoundModel: "model-a",
	}, 0); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	eval, err := manager.Evaluate(Request{
		UserID:        "user-1",
		SessionID:     "session-1",
		SelectorModel: "model-b",
		Candidates: []Candidate{
			{Model: "model-a", Score: 0.2},
			{Model: "model-b", Score: 0.5},
		},
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if eval.Action != ActionSwitch {
		t.Fatalf("Action = %s, want %s", eval.Action, ActionSwitch)
	}
	if eval.Reason != ReasonUpgradeThreshold {
		t.Fatalf("Reason = %s, want %s", eval.Reason, ReasonUpgradeThreshold)
	}
}

func TestManagerDisablesWithoutTrustedUser(t *testing.T) {
	store := NewMemoryStore()
	manager := NewManager(Config{
		Enabled:            true,
		RequireTrustedUser: true,
	}, store)

	eval, err := manager.Evaluate(Request{
		SessionID:     "session-1",
		SelectorModel: "model-a",
		Candidates: []Candidate{
			{Model: "model-a", Score: 1},
		},
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if eval.Action != ActionDisabled {
		t.Fatalf("Action = %s, want %s", eval.Action, ActionDisabled)
	}
	if eval.Reason != ReasonDisabledMissingUser {
		t.Fatalf("Reason = %s, want %s", eval.Reason, ReasonDisabledMissingUser)
	}
}
