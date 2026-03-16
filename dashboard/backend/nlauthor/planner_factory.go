package nlauthor

import "fmt"

func NewPlannerFromRuntimeConfig(manifest SchemaManifest, config RuntimeConfig, extraSources ...PlannerToolSource) Planner {
	normalized := NormalizeRuntimeConfig(config)

	switch normalized.Backend {
	case PlannerBackendPreviewRulebased:
		return NewPreviewPlanner(manifest)
	case PlannerBackendStructuredLLM:
		if normalized.Provider != PlannerProviderOpenAICompatible {
			return NewUnavailablePlanner(
				string(PlannerBackendStructuredLLM),
				"The structured LLM planner provider is not supported by this dashboard build.",
				"planner_provider_unsupported",
				fmt.Sprintf("Planner provider %q is not supported in this build.", normalized.Provider),
			)
		}
		return NewStructuredLLMPlanner(manifest, NewOpenAICompatibleProvider(normalized), normalized)
	case PlannerBackendToolCallingLLM:
		if normalized.Provider != PlannerProviderOpenAICompatible {
			return NewUnavailablePlanner(
				string(PlannerBackendToolCallingLLM),
				"The tool-calling LLM planner provider is not supported by this dashboard build.",
				"planner_provider_unsupported",
				fmt.Sprintf("Planner provider %q is not supported in this build.", normalized.Provider),
			)
		}
		sources := []PlannerToolSource{
			NewBuiltinBuilderToolSource(manifest, NewPreviewPlanner(manifest).Support()),
		}
		if normalized.AllowWebTools {
			sources = append(sources, NewBuiltinWebToolSource())
		}
		sources = append(sources, extraSources...)
		registry := NewPlannerToolRegistry(sources...)
		return NewToolCallingPlanner(manifest, NewOpenAICompatibleProvider(normalized), registry, normalized)
	default:
		return NewUnavailablePlanner(
			string(normalized.Backend),
			"The configured NL planner backend is not supported by this dashboard build.",
			"planner_backend_unsupported",
			fmt.Sprintf("Planner backend %q is not supported in this build.", normalized.Backend),
		)
	}
}

func NewServiceFromRuntimeConfig(manifest SchemaManifest, config RuntimeConfig, extraSources ...PlannerToolSource) *Service {
	planner := NewPlannerFromRuntimeConfig(manifest, config, extraSources...)
	return NewService(manifest, NewInMemorySessionStore(), planner, defaultSessionTTL)
}
