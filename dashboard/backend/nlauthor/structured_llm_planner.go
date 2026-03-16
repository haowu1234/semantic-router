package nlauthor

import (
	"context"
	"encoding/json"
	"strings"
	"time"
)

// StructuredLLMPlanner is the first model-backed backend behind the stable planner seam.
type StructuredLLMPlanner struct {
	manifest         SchemaManifest
	provider         PlannerProvider
	model            string
	maxOutputTokens  int
	supportProfile   PlannerSupport
	unavailableProxy UnavailablePlanner
	observer         PlannerObserver
}

func NewStructuredLLMPlanner(manifest SchemaManifest, provider PlannerProvider, config RuntimeConfig) StructuredLLMPlanner {
	return newStructuredLLMPlannerWithObserver(manifest, provider, config, newDefaultPlannerObserver())
}

func newStructuredLLMPlannerWithObserver(
	manifest SchemaManifest,
	provider PlannerProvider,
	config RuntimeConfig,
	observer PlannerObserver,
) StructuredLLMPlanner {
	normalized := NormalizeRuntimeConfig(config)
	if manifest.Version == "" {
		manifest = DefaultSchemaManifest()
	}
	if observer == nil {
		observer = noopPlannerObserver{}
	}

	unavailable := NewUnavailablePlanner(
		string(PlannerBackendStructuredLLM),
		"The structured LLM planner is configured but the provider settings are incomplete.",
		"planner_unavailable",
		"Structured planner turns require a server-owned provider base URL and model configuration.",
	)

	return StructuredLLMPlanner{
		manifest:         manifest,
		provider:         provider,
		model:            normalized.Model,
		maxOutputTokens:  normalized.MaxOutputTokens,
		supportProfile:   NewPreviewPlanner(manifest).Support(),
		unavailableProxy: unavailable,
		observer:         observer,
	}
}

func (p StructuredLLMPlanner) BackendName() string {
	return string(PlannerBackendStructuredLLM)
}

func (p StructuredLLMPlanner) Available() bool {
	return p.provider != nil && p.provider.Available() && strings.TrimSpace(p.model) != ""
}

func (p StructuredLLMPlanner) SupportsClarification() bool {
	return true
}

func (p StructuredLLMPlanner) Support() PlannerSupport {
	return p.supportProfile
}

func (p StructuredLLMPlanner) Plan(ctx context.Context, session Session, request TurnRequest) (result PlannerResult, err error) {
	startedAt := time.Now()
	defer func() {
		p.observer.ObserveTurn(PlannerTurnObservation{
			Backend:          p.BackendName(),
			Provider:         p.providerName(),
			Model:            p.model,
			ModeHint:         request.ModeHint,
			Status:           result.Status,
			Duration:         time.Since(startedAt),
			PromptChars:      len(strings.TrimSpace(request.Prompt)),
			WarningCount:     len(result.Warnings),
			HasClarification: result.Clarification != nil,
			Error:            result.Error,
		})
	}()

	if !p.Available() {
		return p.unavailableProxy.Plan(ctx, session, request)
	}

	providerResponse, err := p.provider.GenerateStructured(ctx, StructuredGenerationRequest{
		Model:           p.model,
		Messages:        buildStructuredPlannerMessages(p.manifest, p.supportProfile, session, request),
		ResponseSchema:  StructuredOutputSchema{Name: "planner_result", Schema: plannerResultJSONSchema(), Strict: true},
		MaxOutputTokens: p.maxOutputTokens,
	})
	if err != nil {
		return PlannerResult{
			Status:      PlannerStatusError,
			Explanation: "The structured LLM planner could not produce a reviewable draft.",
			Warnings: []PlannerWarning{
				{
					Code:    "planner_provider_error",
					Message: err.Error(),
				},
			},
			Error: "structured planner request failed",
		}, nil
	}

	result = PlannerResult{}
	if err := json.Unmarshal([]byte(providerResponse.Content), &result); err != nil {
		return PlannerResult{
			Status:      PlannerStatusError,
			Explanation: "The structured LLM planner returned invalid JSON.",
			Warnings: []PlannerWarning{
				{
					Code:    "planner_provider_output_invalid",
					Message: err.Error(),
				},
			},
			Error: "invalid structured planner output",
		}, nil
	}

	return result, nil
}

func (p StructuredLLMPlanner) providerName() string {
	if p.provider == nil {
		return ""
	}
	return p.provider.Name()
}
