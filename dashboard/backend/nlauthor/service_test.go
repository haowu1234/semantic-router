package nlauthor

import (
	"context"
	"errors"
	"testing"
	"time"
)

type stubPlanner struct {
	result PlannerResult
}

func (stubPlanner) BackendName() string {
	return "stub"
}

func (stubPlanner) Available() bool {
	return true
}

func (stubPlanner) SupportsClarification() bool {
	return true
}

func (stubPlanner) Support() PlannerSupport {
	return PlannerSupport{
		Operations:  []OperationMode{OperationGenerate},
		Constructs:  []ConstructKind{ConstructSignal},
		SignalTypes: []string{"keyword"},
	}
}

func (s stubPlanner) Plan(_ context.Context, _ Session, _ TurnRequest) (PlannerResult, error) {
	return s.result, nil
}

func TestServiceRejectsUnsupportedSchemaVersion(t *testing.T) {
	t.Parallel()

	service := NewService(DefaultSchemaManifest(), NewInMemorySessionStore(), UnavailablePlanner{}, time.Minute)

	_, err := service.CreateSession(SessionCreateRequest{SchemaVersion: "v9"}, false, "user-1")
	if err == nil {
		t.Fatal("CreateSession error = nil, want schema version error")
	}

	var serviceErr *ServiceError
	if !errors.As(err, &serviceErr) {
		t.Fatalf("CreateSession error = %T, want *ServiceError", err)
	}
	if serviceErr.Code != ErrorCodeInvalidArgument {
		t.Fatalf("service error code = %q, want %q", serviceErr.Code, ErrorCodeInvalidArgument)
	}
}

func TestServiceExpiresSessions(t *testing.T) {
	t.Parallel()

	service := NewService(DefaultSchemaManifest(), NewInMemorySessionStore(), UnavailablePlanner{}, time.Minute)
	baseTime := time.Date(2026, time.March, 13, 10, 0, 0, 0, time.UTC)
	service.now = func() time.Time {
		return baseTime
	}

	session, err := service.CreateSession(SessionCreateRequest{}, false, "user-1")
	if err != nil {
		t.Fatalf("CreateSession error = %v", err)
	}

	service.now = func() time.Time {
		return baseTime.Add(2 * time.Minute)
	}

	_, err = service.RunTurn(context.Background(), session.SessionID, TurnRequest{Prompt: "add pii"}, false, "user-1")
	if err == nil {
		t.Fatal("RunTurn error = nil, want expired session error")
	}

	var serviceErr *ServiceError
	if !errors.As(err, &serviceErr) {
		t.Fatalf("RunTurn error = %T, want *ServiceError", err)
	}
	if serviceErr.Code != ErrorCodeNotFound {
		t.Fatalf("service error code = %q, want %q", serviceErr.Code, ErrorCodeNotFound)
	}
}

func TestServiceNormalizesInvalidPlannerResult(t *testing.T) {
	t.Parallel()

	planner := stubPlanner{
		result: PlannerResult{
			Status:      PlannerStatusReady,
			Explanation: "Create a keyword signal.",
			IntentIR: &IntentIR{
				Version:   "1.0",
				Operation: OperationGenerate,
				Intents: []map[string]interface{}{
					{
						"type":        "signal",
						"signal_type": "not_in_manifest",
						"name":        "bad_signal",
						"fields": map[string]interface{}{
							"keywords": []interface{}{"urgent"},
						},
					},
				},
			},
		},
	}

	service := NewService(DefaultSchemaManifest(), NewInMemorySessionStore(), planner, time.Minute)
	session, err := service.CreateSession(SessionCreateRequest{}, false, "user-1")
	if err != nil {
		t.Fatalf("CreateSession error = %v", err)
	}

	response, err := service.RunTurn(context.Background(), session.SessionID, TurnRequest{
		Prompt: "Create a signal",
	}, false, "user-1")
	if err != nil {
		t.Fatalf("RunTurn error = %v", err)
	}

	if response.Result.Status != PlannerStatusError {
		t.Fatalf("planner status = %q, want %q", response.Result.Status, PlannerStatusError)
	}
	if response.Result.Error == "" {
		t.Fatal("planner error is empty, want normalized validation error")
	}
	if len(response.Result.Warnings) == 0 || response.Result.Warnings[len(response.Result.Warnings)-1].Code != "invalid_planner_result" {
		t.Fatalf("planner warnings = %+v, want invalid_planner_result warning", response.Result.Warnings)
	}
}

func TestServiceTreatsTypedNilRouteConditionAsAbsent(t *testing.T) {
	t.Parallel()

	planner := stubPlanner{
		result: PlannerResult{
			Status:      PlannerStatusReady,
			Explanation: "Create a route without a condition.",
			IntentIR: &IntentIR{
				Version:   "1.0",
				Operation: OperationGenerate,
				Intents: []map[string]interface{}{
					{
						"type":      "route",
						"name":      "support_route",
						"priority":  100,
						"condition": map[string]interface{}(nil),
						"models": []map[string]interface{}{
							{"model": "gpt-4o-mini"},
						},
						"plugins": []map[string]interface{}{},
					},
				},
			},
		},
	}

	service := NewService(DefaultSchemaManifest(), NewInMemorySessionStore(), planner, time.Minute)
	session, err := service.CreateSession(SessionCreateRequest{}, false, "user-1")
	if err != nil {
		t.Fatalf("CreateSession error = %v", err)
	}

	response, err := service.RunTurn(context.Background(), session.SessionID, TurnRequest{
		Prompt: "Create a route",
	}, false, "user-1")
	if err != nil {
		t.Fatalf("RunTurn error = %v", err)
	}

	if response.Result.Status != PlannerStatusReady {
		t.Fatalf("planner status = %q, want %q", response.Result.Status, PlannerStatusReady)
	}
	if response.Result.Error != "" {
		t.Fatalf("planner error = %q, want empty", response.Result.Error)
	}
}
