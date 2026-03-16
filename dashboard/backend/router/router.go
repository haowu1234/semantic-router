package router

import (
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

// Setup configures all routes and returns the configured mux.
func Setup(cfg *config.Config) *http.ServeMux {
	mux := http.NewServeMux()
	authSvc := setupAuthRoutes(mux, cfg)
	openClawHandler := newOpenClawHandler(cfg)
	mcpManager := SetupMCP(mux, cfg, openClawHandler)

	registerCoreRoutes(mux, cfg)
	registerNLAuthoringRoutes(mux, cfg, mcpManager)
	registerEvaluationRoutes(mux, cfg)
	registerMLPipelineRoutes(mux, cfg)
	registerOpenClawRoutes(mux, cfg, openClawHandler)
	registerProxyRoutes(mux, cfg)

	// Static frontend must be registered last.
	mux.Handle("/", handlers.StaticFileServer(cfg.StaticDir))
	return wrapWithAuth(mux, authSvc)
}
