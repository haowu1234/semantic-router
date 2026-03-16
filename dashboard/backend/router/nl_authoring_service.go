package router

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
	"github.com/vllm-project/semantic-router/dashboard/backend/nlauthor"
)

func newNLAuthoringService(cfg *config.Config, mcpManager *mcp.Manager) *nlauthor.Service {
	manifest := nlauthor.DefaultSchemaManifest()
	runtimeConfig := nlauthor.RuntimeConfig{
		Backend:         nlauthor.PlannerBackend(cfg.NLPlannerBackend),
		Provider:        nlauthor.PlannerProviderKind(cfg.NLPlannerProvider),
		BaseURL:         cfg.NLPlannerBaseURL,
		APIKey:          cfg.NLPlannerAPIKey,
		Model:           cfg.NLPlannerModel,
		Timeout:         time.Duration(cfg.NLPlannerTimeoutMs) * time.Millisecond,
		MaxOutputTokens: cfg.NLPlannerMaxOutputTokens,
		ToolBudget:      cfg.NLPlannerToolBudget,
		AllowWebTools:   cfg.NLPlannerAllowWebTools,
		AllowMCPTools:   cfg.NLPlannerAllowMCPTools,
		AllowedMCPTools: []string{cfg.NLPlannerAllowedMCPTools},
	}
	extraSources := make([]nlauthor.PlannerToolSource, 0, 2)
	if runtimeConfig.AllowMCPTools && mcpManager != nil {
		extraSources = append(extraSources, nlauthor.NewMCPPlannerToolSource(mcpManager, strings.TrimSpace(cfg.NLPlannerAllowedMCPTools)))
	}
	return nlauthor.NewServiceFromRuntimeConfig(manifest, runtimeConfig, extraSources...)
}
