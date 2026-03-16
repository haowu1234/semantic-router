package nlauthor

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
)

const plannerInvalidOutputLogCharBudget = 1600

func decodeValidatedPlannerResult(
	raw string,
	manifest SchemaManifest,
	backend string,
	provider string,
	model string,
	decodeExplanation string,
	decodeError string,
) PlannerResult {
	result := PlannerResult{}
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		logInvalidPlannerOutput(backend, provider, model, raw, err)
		return PlannerResult{
			Status:      PlannerStatusError,
			Explanation: decodeExplanation,
			Warnings: []PlannerWarning{
				{
					Code:    "planner_provider_output_invalid",
					Message: err.Error(),
				},
			},
			Error: decodeError,
		}
	}

	if err := validatePlannerResult(result, manifest); err != nil {
		logInvalidPlannerOutput(backend, provider, model, raw, err)
		warnings := append([]PlannerWarning{}, result.Warnings...)
		warnings = append(warnings, PlannerWarning{
			Code:    "invalid_planner_result",
			Message: "Planner output did not match the backend NL schema contract.",
		})
		return PlannerResult{
			Status:      PlannerStatusError,
			Explanation: fallbackPlannerExplanation(result.Explanation, decodeExplanation),
			Warnings:    warnings,
			Error:       fmt.Sprintf("Planner returned invalid structured output: %v", err),
		}
	}

	return result
}

func buildPlannerResultRepairMessage(validationErr error) ProviderMessage {
	return ProviderMessage{
		Role: "user",
		Content: strings.Join([]string{
			"Your previous PlannerResult JSON was invalid.",
			fmt.Sprintf("Validation error: %s.", strings.TrimSpace(validationErr.Error())),
			"Return corrected PlannerResult JSON only.",
			"If status is ready, include intentIr.version=\"1.0\", intentIr.operation, and at least one intent.",
			"If you cannot produce a valid intentIr, return needs_clarification or unsupported instead of ready.",
		}, " "),
	}
}

func validatePlannerResultRaw(raw string, manifest SchemaManifest) error {
	result := PlannerResult{}
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		return err
	}
	return validatePlannerResult(result, manifest)
}

func fallbackPlannerExplanation(value string, fallback string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return fallback
	}
	return trimmed
}

func logInvalidPlannerOutput(backend string, provider string, model string, raw string, err error) {
	excerpt := strings.TrimSpace(raw)
	if len(excerpt) > plannerInvalidOutputLogCharBudget {
		excerpt = excerpt[:plannerInvalidOutputLogCharBudget] + "... [truncated]"
	}
	log.Printf(
		"[NL-Planner-Invalid] backend=%s provider=%s model=%s error=%q raw=%q",
		fallbackPlannerObservationValue(backend, "unknown"),
		fallbackPlannerObservationValue(provider, "unknown"),
		fallbackPlannerObservationValue(model, "unconfigured"),
		strings.TrimSpace(err.Error()),
		excerpt,
	)
}
