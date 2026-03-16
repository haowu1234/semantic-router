package nlauthor

import (
	"encoding/json"
	"strings"
)

func buildToolCallingPlannerMessages(manifest SchemaManifest, support PlannerSupport, session Session, request TurnRequest) []ProviderMessage {
	contextPayload := map[string]any{
		"schemaVersion":  manifest.Version,
		"modeHint":       request.ModeHint,
		"plannerSupport": support,
		"manifestSubset": manifestSubsetForSupport(manifest, support),
		"currentContext": summarizeSessionContextForPrompt(session.Context),
		"userPrompt":     strings.TrimSpace(request.Prompt),
		"responseRules": []string{
			"Use tools when you need current Builder symbols, diagnostics, or DSL snippets before planning.",
			"You must end with strict PlannerResult JSON only.",
			"Do not call tools that are not necessary to answer the current request.",
			"If the request is ambiguous, return needs_clarification instead of guessing.",
			"If status is ready, include intentIr.version=\"1.0\", intentIr.operation, and at least one intent object.",
			"If status is needs_clarification, include clarification.question and 2-4 options.",
			"If you cannot produce a valid intentIr, return needs_clarification or unsupported instead of ready.",
		},
	}

	contextJSON, err := json.MarshalIndent(contextPayload, "", "  ")
	if err != nil {
		contextJSON = []byte(`{"error":"failed to encode planner context"}`)
	}

	return []ProviderMessage{
		{
			Role: "system",
			Content: strings.Join([]string{
				"You are the dashboard Builder NL planner with bounded readonly tools.",
				"Use tools only to inspect Builder context and then return strict PlannerResult JSON.",
				"Never emit DSL, YAML, markdown, or prose outside the final JSON response.",
			}, " "),
		},
		{
			Role:    "user",
			Content: string(contextJSON),
		},
	}
}
