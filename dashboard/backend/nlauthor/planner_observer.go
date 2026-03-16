package nlauthor

import (
	"log"
	"strings"
	"time"
)

type PlannerTurnObservation struct {
	Backend          string
	Provider         string
	Model            string
	ModeHint         OperationMode
	Status           PlannerResultStatus
	Duration         time.Duration
	PromptChars      int
	WarningCount     int
	HasClarification bool
	ToolCalls        int
	ToolErrors       int
	Error            string
}

type PlannerToolObservation struct {
	Backend  string
	Provider string
	Model    string
	Name     string
	Source   string
	Duration time.Duration
	Success  bool
	Error    string
}

type PlannerObserver interface {
	ObserveTurn(PlannerTurnObservation)
	ObserveToolCall(PlannerToolObservation)
}

type noopPlannerObserver struct{}

func (noopPlannerObserver) ObserveTurn(PlannerTurnObservation) {}

func (noopPlannerObserver) ObserveToolCall(PlannerToolObservation) {}

type logPlannerObserver struct{}

func newDefaultPlannerObserver() PlannerObserver {
	return logPlannerObserver{}
}

func (logPlannerObserver) ObserveTurn(observation PlannerTurnObservation) {
	log.Printf(
		"[NL-Planner] backend=%s provider=%s model=%s mode=%s status=%s duration_ms=%d prompt_chars=%d warnings=%d clarification=%t tool_calls=%d tool_errors=%d error=%q",
		fallbackPlannerObservationValue(observation.Backend, "unknown"),
		fallbackPlannerObservationValue(observation.Provider, "unknown"),
		fallbackPlannerObservationValue(observation.Model, "unconfigured"),
		fallbackPlannerObservationValue(string(observation.ModeHint), "generate"),
		fallbackPlannerObservationValue(string(observation.Status), "unknown"),
		observation.Duration.Milliseconds(),
		observation.PromptChars,
		observation.WarningCount,
		observation.HasClarification,
		observation.ToolCalls,
		observation.ToolErrors,
		strings.TrimSpace(observation.Error),
	)
}

func (logPlannerObserver) ObserveToolCall(observation PlannerToolObservation) {
	log.Printf(
		"[NL-Planner-Tool] backend=%s provider=%s model=%s tool=%s source=%s success=%t duration_ms=%d error=%q",
		fallbackPlannerObservationValue(observation.Backend, "unknown"),
		fallbackPlannerObservationValue(observation.Provider, "unknown"),
		fallbackPlannerObservationValue(observation.Model, "unconfigured"),
		fallbackPlannerObservationValue(observation.Name, "unknown"),
		fallbackPlannerObservationValue(observation.Source, "unknown"),
		observation.Success,
		observation.Duration.Milliseconds(),
		strings.TrimSpace(observation.Error),
	)
}

func fallbackPlannerObservationValue(value string, fallback string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return fallback
	}
	return trimmed
}
