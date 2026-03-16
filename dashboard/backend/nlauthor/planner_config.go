package nlauthor

import (
	"strings"
	"time"
)

type PlannerBackend string

const (
	PlannerBackendPreviewRulebased PlannerBackend = "preview-rulebased"
	PlannerBackendStructuredLLM    PlannerBackend = "structured-llm"
	PlannerBackendToolCallingLLM   PlannerBackend = "tool-calling-llm"
)

type PlannerProviderKind string

const (
	PlannerProviderOpenAICompatible PlannerProviderKind = "openai-compatible"
)

const (
	defaultPlannerTimeout         = 15 * time.Second
	defaultPlannerMaxOutputTokens = 1800
)

// RuntimeConfig describes the server-owned planner runtime selection and provider settings.
type RuntimeConfig struct {
	Backend         PlannerBackend
	Provider        PlannerProviderKind
	BaseURL         string
	APIKey          string
	Model           string
	Timeout         time.Duration
	MaxOutputTokens int
	ToolBudget      int
	AllowWebTools   bool
	AllowMCPTools   bool
	AllowedMCPTools []string
}

func NormalizeRuntimeConfig(config RuntimeConfig) RuntimeConfig {
	config.Backend = PlannerBackend(strings.TrimSpace(string(config.Backend)))
	if config.Backend == "" {
		config.Backend = PlannerBackendPreviewRulebased
	}
	config.Provider = PlannerProviderKind(strings.TrimSpace(string(config.Provider)))
	if config.Provider == "" {
		config.Provider = PlannerProviderOpenAICompatible
	}
	config.BaseURL = strings.TrimSpace(config.BaseURL)
	config.APIKey = strings.TrimSpace(config.APIKey)
	config.Model = strings.TrimSpace(config.Model)
	config.AllowedMCPTools = normalizeCommaSeparatedValues(config.AllowedMCPTools)
	if config.Timeout <= 0 {
		config.Timeout = defaultPlannerTimeout
	}
	if config.MaxOutputTokens <= 0 {
		config.MaxOutputTokens = defaultPlannerMaxOutputTokens
	}
	if config.ToolBudget <= 0 {
		config.ToolBudget = 4
	}
	return config
}

func normalizeCommaSeparatedValues(values []string) []string {
	normalized := make([]string, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		for _, item := range strings.Split(value, ",") {
			trimmed := strings.TrimSpace(item)
			if trimmed == "" {
				continue
			}
			if _, ok := seen[trimmed]; ok {
				continue
			}
			seen[trimmed] = struct{}{}
			normalized = append(normalized, trimmed)
		}
	}
	return normalized
}
