package nlauthor

import (
	"sync"
	"testing"
	"time"
)

type recordingPlannerObserver struct {
	mu        sync.Mutex
	turns     []PlannerTurnObservation
	toolCalls []PlannerToolObservation
}

func (o *recordingPlannerObserver) ObserveTurn(observation PlannerTurnObservation) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.turns = append(o.turns, observation)
}

func (o *recordingPlannerObserver) ObserveToolCall(observation PlannerToolObservation) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.toolCalls = append(o.toolCalls, observation)
}

func (o *recordingPlannerObserver) snapshot() ([]PlannerTurnObservation, []PlannerToolObservation) {
	o.mu.Lock()
	defer o.mu.Unlock()
	turns := append([]PlannerTurnObservation(nil), o.turns...)
	toolCalls := append([]PlannerToolObservation(nil), o.toolCalls...)
	return turns, toolCalls
}

func TestLogPlannerObserverFallbackValue(t *testing.T) {
	t.Parallel()

	if got := fallbackPlannerObservationValue("  ", "fallback"); got != "fallback" {
		t.Fatalf("fallbackPlannerObservationValue = %q, want fallback", got)
	}
	if got := fallbackPlannerObservationValue("structured-llm", "fallback"); got != "structured-llm" {
		t.Fatalf("fallbackPlannerObservationValue = %q, want structured-llm", got)
	}
}

func TestRecordingPlannerObserverCollectsEvents(t *testing.T) {
	t.Parallel()

	observer := &recordingPlannerObserver{}
	observer.ObserveTurn(PlannerTurnObservation{Backend: "structured-llm", Status: PlannerStatusReady, Duration: time.Second})
	observer.ObserveToolCall(PlannerToolObservation{Backend: "tool-calling-llm", Name: "list_symbols", Success: true})

	turns, toolCalls := observer.snapshot()
	if len(turns) != 1 {
		t.Fatalf("turns = %d, want 1", len(turns))
	}
	if len(toolCalls) != 1 {
		t.Fatalf("toolCalls = %d, want 1", len(toolCalls))
	}
}
