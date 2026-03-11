package router

import (
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

// registerNLRoutes registers NL-to-DSL LLM proxy endpoints
func registerNLRoutes(mux *http.ServeMux, cfg *config.Config) {
	nlCfg := handlers.LoadNLConfig()
	handlers.LoadNLConfigFromYAML(nlCfg, cfg.AbsConfigPath)

	mux.HandleFunc("/api/nl/generate", handlers.NLGenerateHandler(nlCfg))
	mux.HandleFunc("/api/nl/explain", handlers.NLExplainHandler(nlCfg))
	mux.HandleFunc("/api/nl/config", handlers.NLConfigHandler(nlCfg))

	log.Printf("NL API endpoints registered: /api/nl/generate, /api/nl/explain, /api/nl/config")
	if nlCfg.DefaultEndpoint != "" {
		log.Printf("NL LLM endpoint: %s (model=%s, server-key=%v, models=%d)",
			nlCfg.DefaultEndpoint, nlCfg.DefaultModel, nlCfg.DefaultAPIKey != "", len(nlCfg.AvailableModels))
	} else {
		log.Printf("NL LLM endpoint not configured (frontend must provide endpoint)")
	}
}
