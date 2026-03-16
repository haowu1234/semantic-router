package nlauthor

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ToolCallingPlanner adds bounded readonly tool access behind the stable planner seam.
type ToolCallingPlanner struct {
	manifest         SchemaManifest
	provider         PlannerProvider
	registry         PlannerToolRegistry
	model            string
	maxOutputTokens  int
	toolPolicy       ToolPolicy
	supportProfile   PlannerSupport
	unavailableProxy UnavailablePlanner
	observer         PlannerObserver
}

func NewToolCallingPlanner(
	manifest SchemaManifest,
	provider PlannerProvider,
	registry PlannerToolRegistry,
	config RuntimeConfig,
) ToolCallingPlanner {
	return newToolCallingPlannerWithObserver(manifest, provider, registry, config, newDefaultPlannerObserver())
}

func newToolCallingPlannerWithObserver(
	manifest SchemaManifest,
	provider PlannerProvider,
	registry PlannerToolRegistry,
	config RuntimeConfig,
	observer PlannerObserver,
) ToolCallingPlanner {
	normalized := NormalizeRuntimeConfig(config)
	if manifest.Version == "" {
		manifest = DefaultSchemaManifest()
	}
	if observer == nil {
		observer = noopPlannerObserver{}
	}

	policy := DefaultToolPolicy()
	policy.MaxCalls = normalized.ToolBudget
	if normalized.AllowWebTools {
		policy.AllowedSources = append(policy.AllowedSources, PlannerToolSourceBuiltinWeb)
	}
	if normalized.AllowMCPTools {
		policy.AllowedSources = append(policy.AllowedSources, PlannerToolSourceMCP)
	}

	return ToolCallingPlanner{
		manifest:        manifest,
		provider:        provider,
		registry:        registry,
		model:           normalized.Model,
		maxOutputTokens: normalized.MaxOutputTokens,
		toolPolicy:      policy,
		supportProfile:  NewPreviewPlanner(manifest).Support(),
		unavailableProxy: NewUnavailablePlanner(
			string(PlannerBackendToolCallingLLM),
			"The tool-calling LLM planner is configured but the provider settings are incomplete.",
			"planner_unavailable",
			"Tool-calling planner turns require a server-owned provider base URL and model configuration.",
		),
		observer: observer,
	}
}

func (p ToolCallingPlanner) BackendName() string {
	return string(PlannerBackendToolCallingLLM)
}

func (p ToolCallingPlanner) Available() bool {
	return p.provider != nil && p.provider.Available() && p.model != ""
}

func (p ToolCallingPlanner) SupportsClarification() bool {
	return true
}

func (p ToolCallingPlanner) Support() PlannerSupport {
	return p.supportProfile
}

func (p ToolCallingPlanner) Plan(ctx context.Context, session Session, request TurnRequest) (result PlannerResult, err error) {
	startedAt := time.Now()
	toolCallsUsed := 0
	toolErrors := 0
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
			ToolCalls:        toolCallsUsed,
			ToolErrors:       toolErrors,
			Error:            result.Error,
		})
	}()

	if !p.Available() {
		return p.unavailableProxy.Plan(ctx, session, request)
	}

	messages := buildToolCallingPlannerMessages(p.manifest, p.supportProfile, session, request)
	toolDefinitions := definitionsToProviderTools(p.registry.List(session, request, p.toolPolicy))
	validationRepairAttempted := false

	for iteration := 0; iteration <= p.toolPolicy.MaxCalls; iteration++ {
		response, err := p.provider.GenerateToolCalls(ctx, ToolCallingRequest{
			Model:           p.model,
			Messages:        messages,
			Tools:           toolDefinitions,
			ResponseSchema:  StructuredOutputSchema{Name: "planner_result", Schema: plannerResultJSONSchema(), Strict: true},
			MaxOutputTokens: p.maxOutputTokens,
		})
		if err != nil {
			return PlannerResult{
				Status:      PlannerStatusError,
				Explanation: "The tool-calling planner could not produce a reviewable draft.",
				Warnings: []PlannerWarning{
					{Code: "planner_provider_error", Message: err.Error()},
				},
				Error: "tool-calling planner request failed",
			}, nil
		}

		if len(response.ToolCalls) == 0 {
			if validationErr := validatePlannerResultRaw(response.Content, p.manifest); validationErr != nil && !validationRepairAttempted {
				validationRepairAttempted = true
				messages = append(messages, ProviderMessage{
					Role:    "assistant",
					Content: strings.TrimSpace(response.Content),
				})
				messages = append(messages, buildPlannerResultRepairMessage(validationErr))
				continue
			}

			result = decodeValidatedPlannerResult(
				response.Content,
				p.manifest,
				p.BackendName(),
				p.providerName(),
				p.model,
				"The tool-calling planner returned invalid JSON.",
				"invalid tool-calling planner output",
			)
			return result, nil
		}

		if iteration == p.toolPolicy.MaxCalls {
			return PlannerResult{
				Status:      PlannerStatusError,
				Explanation: "The tool-calling planner exceeded its tool budget before producing a final draft.",
				Error:       "planner tool budget exceeded",
			}, nil
		}

		assistantToolMessage := ProviderMessage{Role: "assistant", ToolCalls: response.ToolCalls}
		messages = append(messages, assistantToolMessage)
		for _, toolCall := range response.ToolCalls {
			toolCallsUsed++
			definition, _ := p.registry.Definition(session, request, toolCall.Name, p.toolPolicy)
			toolStartedAt := time.Now()
			result, err := p.registry.Invoke(ctx, session, request, toolCall.Name, json.RawMessage(toolCall.Arguments), p.toolPolicy)
			toolContent := result.Content
			if err != nil {
				toolErrors++
				toolContent = fmt.Sprintf(`{"error":%q}`, err.Error())
			}
			p.observer.ObserveToolCall(PlannerToolObservation{
				Backend:  p.BackendName(),
				Provider: p.providerName(),
				Model:    p.model,
				Name:     toolCall.Name,
				Source:   definition.Source,
				Duration: time.Since(toolStartedAt),
				Success:  err == nil,
				Error:    stringifyError(err),
			})
			messages = append(messages, ProviderMessage{
				Role:       "tool",
				ToolCallID: toolCall.ID,
				Content:    toolContent,
			})
		}
	}

	return PlannerResult{
		Status:      PlannerStatusError,
		Explanation: "The tool-calling planner did not finish within its execution loop.",
		Error:       "planner tool loop exhausted",
	}, nil
}

func definitionsToProviderTools(definitions []PlannerToolDefinition) []ProviderToolDefinition {
	tools := make([]ProviderToolDefinition, 0, len(definitions))
	for _, definition := range definitions {
		tools = append(tools, ProviderToolDefinition{
			Name:        definition.Name,
			Description: definition.Description,
			InputSchema: definition.InputSchema,
		})
	}
	return tools
}

func (p ToolCallingPlanner) providerName() string {
	if p.provider == nil {
		return ""
	}
	return p.provider.Name()
}

func stringifyError(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
