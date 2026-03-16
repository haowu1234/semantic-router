package nlauthor

import (
	"encoding/json"
	"fmt"
	"strings"
)

const structuredPlannerPromptDSLCharBudget = 6000

func plannerResultJSONSchema() json.RawMessage {
	schema := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []string{"status"},
		"allOf": []map[string]any{
			{
				"if": map[string]any{
					"properties": map[string]any{
						"status": map[string]any{"const": string(PlannerStatusReady)},
					},
				},
				"then": map[string]any{
					"required": []string{"intentIr"},
				},
			},
			{
				"if": map[string]any{
					"properties": map[string]any{
						"status": map[string]any{"const": string(PlannerStatusNeedsClarification)},
					},
				},
				"then": map[string]any{
					"required": []string{"clarification"},
				},
			},
		},
		"properties": map[string]any{
			"status": map[string]any{
				"type": "string",
				"enum": []string{
					string(PlannerStatusReady),
					string(PlannerStatusNeedsClarification),
					string(PlannerStatusUnsupported),
					string(PlannerStatusError),
				},
			},
			"explanation": map[string]any{"type": "string"},
			"error":       map[string]any{"type": "string"},
			"warnings": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type":                 "object",
					"additionalProperties": false,
					"required":             []string{"code", "message"},
					"properties": map[string]any{
						"code":    map[string]any{"type": "string"},
						"message": map[string]any{"type": "string"},
					},
				},
			},
			"clarification": map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []string{"question", "options"},
				"properties": map[string]any{
					"question": map[string]any{"type": "string"},
					"options": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type":                 "object",
							"additionalProperties": false,
							"required":             []string{"id", "label"},
							"properties": map[string]any{
								"id":          map[string]any{"type": "string"},
								"label":       map[string]any{"type": "string"},
								"description": map[string]any{"type": "string"},
							},
						},
					},
				},
			},
			"intentIr": map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             []string{"version", "operation", "intents"},
				"properties": map[string]any{
					"version": map[string]any{"type": "string"},
					"operation": map[string]any{
						"type": "string",
						"enum": []string{
							string(OperationGenerate),
							string(OperationModify),
							string(OperationFix),
						},
					},
					"intents": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type":                 "object",
							"additionalProperties": true,
						},
					},
				},
			},
		},
	}

	raw, err := json.Marshal(schema)
	if err != nil {
		panic(fmt.Sprintf("marshal planner result schema: %v", err))
	}
	return raw
}

func buildStructuredPlannerMessages(manifest SchemaManifest, support PlannerSupport, session Session, request TurnRequest) []ProviderMessage {
	contextPayload := map[string]any{
		"schemaVersion":  manifest.Version,
		"modeHint":       request.ModeHint,
		"plannerSupport": support,
		"manifestSubset": manifestSubsetForSupport(manifest, support),
		"currentContext": summarizeSessionContextForPrompt(session.Context),
		"userPrompt":     strings.TrimSpace(request.Prompt),
		"responseRules": []string{
			"Return strict JSON only. Do not wrap the JSON in markdown.",
			"If the request is ambiguous or missing required details, return status needs_clarification with a question and 2-4 options.",
			"If the request is outside the supported planner subset, return status unsupported with a brief explanation and error.",
			"If status is ready, include intentIr.version=\"1.0\", intentIr.operation, and at least one intent object.",
			"Do not invent unsupported signal, plugin, backend, or algorithm types beyond the provided support subset.",
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
				"You are the dashboard Builder NL planner.",
				"You must produce structured PlannerResult JSON for the Builder review/apply flow.",
				"Stay on the canonical DSL contract. Do not output DSL text, YAML, markdown, or prose outside the JSON response.",
			}, " "),
		},
		{
			Role:    "user",
			Content: string(contextJSON),
		},
	}
}

func manifestSubsetForSupport(manifest SchemaManifest, support PlannerSupport) map[string]any {
	return map[string]any{
		"routes":     routeSchemaEntries("", support),
		"signals":    filterManifestEntries(manifest.Signals, support.SignalTypes),
		"plugins":    filterManifestEntries(manifest.Plugins, support.PluginTypes),
		"algorithms": filterManifestEntries(manifest.Algorithms, support.AlgorithmTypes),
		"backends":   filterManifestEntries(manifest.Backends, support.BackendTypes),
	}
}

func filterManifestEntries(entries []TypeSchemaEntry, allowed []string) []TypeSchemaEntry {
	if len(allowed) == 0 {
		return []TypeSchemaEntry{}
	}

	allowedSet := make(map[string]struct{}, len(allowed))
	for _, value := range allowed {
		allowedSet[value] = struct{}{}
	}

	filtered := make([]TypeSchemaEntry, 0, len(entries))
	for _, entry := range entries {
		if _, ok := allowedSet[entry.TypeName]; ok {
			filtered = append(filtered, entry)
		}
	}
	return filtered
}

func summarizeSessionContextForPrompt(context SessionContext) map[string]any {
	summary := map[string]any{}
	if context.Symbols != nil {
		summary["symbols"] = context.Symbols
	}
	if len(context.Diagnostics) > 0 {
		summary["diagnostics"] = context.Diagnostics
	}

	baseDSL := strings.TrimSpace(context.BaseDSL)
	if baseDSL != "" {
		if len(baseDSL) > structuredPlannerPromptDSLCharBudget {
			baseDSL = baseDSL[:structuredPlannerPromptDSLCharBudget] + "\n... [truncated]"
		}
		summary["baseDsl"] = baseDSL
	}
	return summary
}
